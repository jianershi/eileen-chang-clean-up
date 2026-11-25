# SPDX-License-Identifier: MIT
"""Multi-pass RE-USE ensemble, speech-gated archival residual fusion, and linear mastering."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .chunks import join_chunks, plan_chunks
from .common import check_output, probe, require_program, run, sha256, write_json
from .export import export_m4a
from .profile import (
    ENSEMBLE_AUX_SAMPLE_RATE,
    ENSEMBLE_OUTPUT_STEM,
    ENSEMBLE_TITLE,
    LAUGHTER_MODEL_REPOSITORY,
    LAUGHTER_MODEL_REVISION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SAMPLE_RATE,
    SOURCE_NOTICE,
)
from .restore import bind_run, enhance_chunks, valid_checkpoint
from .verify import verify_audio

ENSEMBLE_PIPELINE_VERSION = 1


def prepare_ensemble_inputs(source, work_dir):
    """Create 44.1 kHz raw audio plus 60 Hz high-pass 44.1 kHz and 16 kHz inputs."""
    import soxr
    from scipy import signal

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw44 = work_dir / "raw_44k.wav"
    prep44 = work_dir / "prepared_44k_hpf60.wav"
    prep16 = work_dir / "prepared_16k_hpf60.wav"

    if not raw44.exists():
        tmp = work_dir / "raw_44k.partial.wav"
        run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source,
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", tmp])
        os.replace(tmp, raw44)

    if not prep44.exists():
        tmp = work_dir / "prepared_44k_hpf60.partial.wav"
        run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source,
             "-af", "highpass=f=60", "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-c:a", "pcm_f32le", tmp])
        os.replace(tmp, prep44)

    if not prep16.exists():
        audio44, rate = sf.read(raw44, dtype="float32")
        if rate != SAMPLE_RATE:
            raise ValueError("Prepared 44.1 kHz audio has an unexpected sample rate.")
        audio16 = soxr.resample(audio44, SAMPLE_RATE, ENSEMBLE_AUX_SAMPLE_RATE, quality="VHQ").astype(np.float32)
        sos = signal.butter(4, 60.0, btype="highpass", fs=ENSEMBLE_AUX_SAMPLE_RATE, output="sos")
        audio16 = signal.sosfiltfilt(sos, audio16).astype(np.float32)
        tmp = work_dir / "prepared_16k_hpf60.partial.wav"
        sf.write(tmp, audio16, ENSEMBLE_AUX_SAMPLE_RATE, subtype="FLOAT", format="WAV")
        os.replace(tmp, prep16)

    return raw44, prep44, prep16


def run_resumable_pass(prepared_wav, pass_dir, sample_rate, repository, revision,
                       backend="auto", core_seconds=20.0, context_seconds=2.0,
                       overlap_seconds=1.0, shift_seconds=0.0,
                       model_root=None, enhancer_factory=None, progress=None):
    """Run one resumable, checksum-verified enhancement pass with optional front shift."""
    pass_dir = Path(pass_dir)
    pass_dir.mkdir(parents=True, exist_ok=True)
    joined_path = pass_dir / "joined.wav"
    if joined_path.exists():
        info = sf.info(joined_path)
        orig_info = sf.info(prepared_wav)
        if info.samplerate == sample_rate and info.frames == orig_info.frames:
            return joined_path, {"reused": True, "sample_rate": sample_rate}

    orig_info = sf.info(prepared_wav)
    if orig_info.samplerate != sample_rate or orig_info.channels != 1:
        raise ValueError(f"Prepared audio mismatch for {pass_dir.name}.")

    pad_front = round(shift_seconds * sample_rate)
    if pad_front > 0:
        shifted_input = pass_dir / "shifted_input.wav"
        if not shifted_input.exists():
            data, _ = sf.read(prepared_wav, dtype="float32")
            padded = np.pad(data, (pad_front, 0), mode="reflect").astype(np.float32)
            tmp = pass_dir / "shifted_input.partial.wav"
            sf.write(tmp, padded, sample_rate, subtype="FLOAT", format="WAV")
            os.replace(tmp, shifted_input)
        run_input = shifted_input
        total_samples = orig_info.frames + pad_front
    else:
        run_input = Path(prepared_wav)
        total_samples = orig_info.frames

    chunks = plan_chunks(total_samples, sample_rate, core_seconds, context_seconds, overlap_seconds)
    checkpoint_dir = pass_dir / "chunks"
    pending = any(valid_checkpoint(checkpoint_dir, chunk, sample_rate) is None for chunk in chunks)

    enhancer = None
    env_info = {}
    if pending:
        if enhancer_factory is not None:
            enhancer = enhancer_factory(repository, revision, sample_rate)
            env_info = {"device": getattr(enhancer, "device_name", "custom"),
                        "runtime": getattr(enhancer, "torch_version", "custom")}
        else:
            from .model import download_model, load_enhancer
            cache_dir = Path(model_root or (pass_dir.parent / "models")) / f"{repository.replace('/', '__')}_{revision[:7]}"
            model_dir, weights_hash = download_model(cache_dir, revision=revision, repository=repository)
            enhancer = load_enhancer(model_dir, sample_rate, backend=backend)
            env_info = {"device": enhancer.device_name, "runtime": enhancer.torch_version,
                        "weights_sha256": weights_hash}

    rows = enhance_chunks(run_input, chunks, checkpoint_dir, sample_rate, enhancer, progress=progress)
    del enhancer

    joined_full = pass_dir / "joined_full.partial.wav"
    join_chunks(rows, checkpoint_dir, joined_full, total_samples, sample_rate, overlap_seconds)
    if pad_front > 0:
        full_audio, _ = sf.read(joined_full, dtype="float32")
        trimmed = full_audio[pad_front:pad_front + orig_info.frames]
        tmp = pass_dir / "joined.partial.wav"
        sf.write(tmp, trimmed, sample_rate, subtype="FLOAT", format="WAV")
        os.replace(tmp, joined_path)
        joined_full.unlink(missing_ok=True)
    else:
        os.replace(joined_full, joined_path)

    return joined_path, {**env_info, "reused": False, "sample_rate": sample_rate, "shift_seconds": shift_seconds}


def gentle_wiener_blockwise(audio, sample_rate=SAMPLE_RATE, block_seconds=60.0,
                            overlap_seconds=5.0, max_attenuation_db=8.0):
    """Apply bounded STFT Wiener denoising (8 dB max cut) in overlapping blocks."""
    from scipy import ndimage, signal

    audio = np.asarray(audio, dtype=np.float32)
    total = len(audio)
    if total == 0:
        return audio.copy()
    block_len = min(total, max(2048, round(block_seconds * sample_rate)))
    overlap = min(block_len // 4, round(overlap_seconds * sample_rate))
    step = max(1, block_len - overlap)
    nfft = 2048 if sample_rate >= 40_000 else 512
    nfft = min(nfft, block_len)
    hop = max(1, nfft // 4)
    gain_floor = 10.0 ** (-max_attenuation_db / 20.0)

    out = np.zeros(total, dtype=np.float32)
    weights = np.zeros(total, dtype=np.float32)
    pos = 0
    while pos < total:
        end = min(total, pos + block_len)
        seg = audio[pos:end]
        if len(seg) < nfft:
            clean = seg.copy()
        else:
            freqs, _, spec = signal.stft(seg, fs=sample_rate, nperseg=nfft, noverlap=nfft - hop)
            power = np.abs(spec) ** 2
            band = (freqs >= 250) & (freqs <= 3400)
            band_energy = power[band].sum(axis=0) if np.any(band) else power.sum(axis=0)
            level = ndimage.uniform_filter1d(band_energy, size=min(15, len(band_energy)))
            quiet = level <= np.quantile(level, 0.12)
            noise = np.mean(power[:, quiet], axis=1, keepdims=True)
            smooth_power = ndimage.uniform_filter1d(power, size=min(5, power.shape[1]), axis=1)
            mask = np.sqrt(np.maximum(gain_floor ** 2, 1.0 - noise / np.maximum(smooth_power, 1e-18)))
            mask = ndimage.gaussian_filter(mask, sigma=(1.0, 1.5))
            _, clean = signal.istft(spec * mask, fs=sample_rate, nperseg=nfft, noverlap=nfft - hop)
            clean = np.pad(clean, (0, max(0, len(seg) - len(clean))))[:len(seg)].astype(np.float32)

        weight = np.ones(len(seg), dtype=np.float32)
        if pos > 0 and overlap > 0:
            n_fade = min(overlap, len(seg))
            phase = np.linspace(0.0, np.pi, n_fade, dtype=np.float64)
            weight[:n_fade] *= (0.5 - 0.5 * np.cos(phase)).astype(np.float32)
        if end < total and overlap > 0:
            n_fade = min(overlap, len(seg))
            phase = np.linspace(0.0, np.pi, n_fade, dtype=np.float64)
            weight[-n_fade:] *= (0.5 + 0.5 * np.cos(phase)).astype(np.float32)

        out[pos:end] += clean * weight
        weights[pos:end] += weight
        if end == total:
            break
        pos += step
    return (out / np.maximum(weights, 1e-8)).astype(np.float32)


def _active_speech_rms(audio, sample_rate):
    frame_len = max(1, int(0.05 * sample_rate))
    n_frames = len(audio) // frame_len
    if n_frames < 4:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) + 1e-12
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return float(np.percentile(rms, 75)) + 1e-12


def _estimate_lag(reference, candidate, sample_rate, max_lag_ms=20.0):
    from scipy import signal

    n = min(len(reference), len(candidate))
    window_len = min(n, int(30 * sample_rate))
    start = min(int(200 * sample_rate), max(0, (n - window_len) // 2))
    end = start + window_len
    max_lag = min(window_len // 4, int(max_lag_ms * sample_rate / 1000.0))
    if max_lag < 1:
        return 0
    corr = signal.correlate(reference[start:end].astype(np.float64),
                            candidate[start:end].astype(np.float64), mode="full")
    mid = len(corr) // 2
    return int(np.argmax(corr[mid - max_lag:mid + max_lag + 1])) - max_lag


def _apply_lag(audio, lag):
    if lag > 0:
        return np.pad(audio[:-lag], (lag, 0))
    if lag < 0:
        return np.pad(audio[-lag:], (0, -lag))
    return audio


def normalize_linear_master(source, output, target_lufs=-20.0, overwrite=False, title=ENSEMBLE_TITLE):
    """Apply one constant measured linear gain to target LUFS plus a -3 dBTP limiter (no dynaudnorm)."""
    check_output(output, overwrite, source)
    measured = run(["ffmpeg", "-nostdin", "-hide_banner", "-i", source,
                    "-af", f"loudnorm=I={target_lufs}:TP=-3:LRA=11:print_format=json",
                    "-f", "null", "-"], capture_output=True, text=True)
    text = measured.stderr
    values = json.loads(text[text.rfind("{"):text.rfind("}") + 1])
    input_i = float(values["input_i"])
    if not math.isfinite(input_i):
        raise ValueError("Cannot normalize silent or non-finite audio.")
    gain_db = target_lufs - input_i
    output = Path(output)
    temporary = output.with_name(output.stem + ".partial.flac")
    run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source,
         "-af", f"volume={gain_db:.4f}dB,alimiter=limit=0.708:attack=5:release=50:level=false:latency=true",
         "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "flac", "-sample_fmt", "s32",
         "-metadata", f"title={title}", "-metadata", f"comment={SOURCE_NOTICE}", temporary])
    os.replace(temporary, output)
    return {"pre_master": values, "linear_gain_db": gain_db}


def fuse_ensemble_tracks(pass1_44k, pass2_16k, pass3_16k, raw_44k, output_wav, residual_alpha=0.20):
    """Align, mix (50/30/20 below 7.2 kHz, 100% Pass 1 above), and apply speech-gated Wiener residual."""
    import soxr
    from scipy import ndimage, signal

    lau44, rate1 = sf.read(pass1_44k, dtype="float32")
    if rate1 != SAMPLE_RATE:
        raise ValueError("Pass 1 must be 44.1 kHz.")
    n_total = len(lau44)

    lau16s, rate2 = sf.read(pass2_16k, dtype="float32")
    std16, rate3 = sf.read(pass3_16k, dtype="float32")
    raw44, rate_raw = sf.read(raw_44k, dtype="float32")
    if rate_raw != SAMPLE_RATE:
        raise ValueError("Raw archival track must be 44.1 kHz.")
    raw44 = np.pad(raw44, (0, max(0, n_total - len(raw44))))[:n_total]

    lau16s_44 = soxr.resample(lau16s, rate2, SAMPLE_RATE, quality="VHQ").astype(np.float32)
    std16_44 = soxr.resample(std16, rate3, SAMPLE_RATE, quality="VHQ").astype(np.float32)
    lau16s_44 = np.pad(lau16s_44, (0, max(0, n_total - len(lau16s_44))))[:n_total]
    std16_44 = np.pad(std16_44, (0, max(0, n_total - len(std16_44))))[:n_total]

    sos_sb = signal.butter(4, [300.0, 3500.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    ref_sb = signal.sosfiltfilt(sos_sb, lau44).astype(np.float32)
    ref_rms = _active_speech_rms(ref_sb, SAMPLE_RATE)

    alignments = {}
    aligned = []
    for name, track in (("pass2_16k_shift10s", lau16s_44), ("pass3_16k_standard", std16_44)):
        sb = signal.sosfiltfilt(sos_sb, track).astype(np.float32)
        lag = _estimate_lag(ref_sb, sb, SAMPLE_RATE)
        scale = ref_rms / _active_speech_rms(sb, SAMPLE_RATE)
        alignments[name] = {"lag_samples": lag, "scale": scale}
        aligned.append(_apply_lag(track, lag) * scale)
    lau16s_44, std16_44 = aligned

    # 7.2 kHz crossover: 50/30/20 mix below 7.2 kHz, 100% Pass 1 above 7.2 kHz.
    sos_lp = signal.butter(4, 7200.0, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    lau44_lo = signal.sosfiltfilt(sos_lp, lau44).astype(np.float32)
    lau44_hi = lau44 - lau44_lo
    lau16s_lo = signal.sosfiltfilt(sos_lp, lau16s_44).astype(np.float32)
    std16_lo = signal.sosfiltfilt(sos_lp, std16_44).astype(np.float32)
    ensemble = (0.50 * lau44_lo + 0.30 * lau16s_lo + 0.20 * std16_lo + lau44_hi).astype(np.float32)

    if residual_alpha > 0.0:
        sos_bp = signal.butter(4, [180.0, 3600.0], btype="bandpass", fs=SAMPLE_RATE, output="sos")
        raw_bp = signal.sosfiltfilt(sos_bp, raw44).astype(np.float32)
        ens_bp = signal.sosfiltfilt(sos_bp, ensemble).astype(np.float32)
        lag_raw = _estimate_lag(ens_bp, raw_bp, SAMPLE_RATE)
        scale_raw = _active_speech_rms(ens_bp, SAMPLE_RATE) / _active_speech_rms(raw_bp, SAMPLE_RATE)
        alignments["raw_180_3600hz"] = {"lag_samples": lag_raw, "scale": scale_raw}
        raw_bp = _apply_lag(raw_bp, lag_raw) * scale_raw
        raw_clean = gentle_wiener_blockwise(raw_bp, sample_rate=SAMPLE_RATE,
                                            block_seconds=60.0, overlap_seconds=5.0,
                                            max_attenuation_db=8.0)
        env = np.sqrt(np.maximum(
            ndimage.uniform_filter1d(ensemble.astype(np.float64) ** 2, size=max(1, int(0.04 * SAMPLE_RATE))),
            0.0,
        ))
        speech_ref = float(np.percentile(env, 85)) + 1e-12
        ratio = env / (0.06 * speech_ref)
        gate = (ratio ** 2 / (1.0 + ratio ** 2)).astype(np.float32)
        gate = ndimage.uniform_filter1d(gate, size=max(1, int(0.08 * SAMPLE_RATE))).astype(np.float32)
        fused = (ensemble * (1.0 - residual_alpha * gate) + raw_clean * (residual_alpha * gate)).astype(np.float32)
    else:
        fused = ensemble

    if len(fused) != n_total or not np.isfinite(fused).all():
        raise ValueError("Ensemble fusion produced invalid length or non-finite samples.")
    output_wav = Path(output_wav)
    tmp = output_wav.with_name(output_wav.stem + ".partial.wav")
    sf.write(tmp, fused, SAMPLE_RATE, subtype="FLOAT", format="WAV")
    os.replace(tmp, output_wav)
    return alignments, n_total


def restore_ensemble(source, output_dir, work_dir, backend="auto",
                     core_seconds=20.0, context_seconds=2.0, overlap_seconds=1.0,
                     shift_seconds=10.0, residual_alpha=0.20,
                     pass1_joined=None, expected_duration=None, overwrite=False,
                     enhancer_factory=None):
    """Run the 3-pass multi-resolution RE-USE ensemble + gated archival residual + linear mastering."""
    require_program("ffmpeg")
    source = Path(source).resolve()
    output_dir, work_dir = Path(output_dir), Path(work_dir)
    flac_out = output_dir / f"{ENSEMBLE_OUTPUT_STEM}.flac"
    m4a_out = output_dir / f"{ENSEMBLE_OUTPUT_STEM}.m4a"
    check_output(flac_out, overwrite, source)
    check_output(m4a_out, overwrite, source)

    source_hash = sha256(source)
    duration = float(probe(source)["format"]["duration"])
    if expected_duration is not None and abs(duration - expected_duration) > 1:
        raise ValueError("The input duration does not match the expected recording.")

    configuration = {
        "pipeline_version": ENSEMBLE_PIPELINE_VERSION,
        "mode": "multi-pass-ensemble",
        "source_name": source.name,
        "source_sha256": source_hash,
        "sample_rate": SAMPLE_RATE,
        "aux_sample_rate": ENSEMBLE_AUX_SAMPLE_RATE,
        "highpass_hz": 60,
        "core_seconds": core_seconds,
        "context_seconds": context_seconds,
        "overlap_seconds": overlap_seconds,
        "shift_seconds": shift_seconds,
        "residual_alpha": residual_alpha,
        "laughter_model": LAUGHTER_MODEL_REPOSITORY,
        "laughter_revision": LAUGHTER_MODEL_REVISION,
        "standard_model": MODEL_REPOSITORY,
        "standard_revision": MODEL_REVISION,
    }
    bind_run(work_dir, configuration)

    started = time.monotonic()
    raw44, prep44, prep16 = prepare_ensemble_inputs(source, work_dir)
    progress = lambda row: print(json.dumps(row), flush=True)

    passes = {}
    if pass1_joined is not None:
        pass1_path = Path(pass1_joined).resolve()
        info1 = sf.info(pass1_path)
        info_ref = sf.info(prep44)
        if info1.samplerate != SAMPLE_RATE or info1.frames != info_ref.frames:
            raise ValueError("Supplied --pass1-joined WAV does not match the 44.1 kHz prepared sample count.")
        passes["pass1_44k_laughter"] = {"external_joined": str(pass1_path), "sha256": sha256(pass1_path)}
    else:
        pass1_path, passes["pass1_44k_laughter"] = run_resumable_pass(
            prep44, work_dir / "pass1_44k_laughter", SAMPLE_RATE,
            LAUGHTER_MODEL_REPOSITORY, LAUGHTER_MODEL_REVISION,
            backend=backend, core_seconds=core_seconds, context_seconds=context_seconds,
            overlap_seconds=overlap_seconds, shift_seconds=0.0,
            model_root=work_dir / "models", enhancer_factory=enhancer_factory, progress=progress,
        )

    pass2_path, passes["pass2_16k_laughter_shift"] = run_resumable_pass(
        prep16, work_dir / "pass2_16k_laughter_shift", ENSEMBLE_AUX_SAMPLE_RATE,
        LAUGHTER_MODEL_REPOSITORY, LAUGHTER_MODEL_REVISION,
        backend=backend, core_seconds=core_seconds, context_seconds=context_seconds,
        overlap_seconds=overlap_seconds, shift_seconds=shift_seconds,
        model_root=work_dir / "models", enhancer_factory=enhancer_factory, progress=progress,
    )

    pass3_path, passes["pass3_16k_standard"] = run_resumable_pass(
        prep16, work_dir / "pass3_16k_standard", ENSEMBLE_AUX_SAMPLE_RATE,
        MODEL_REPOSITORY, MODEL_REVISION,
        backend=backend, core_seconds=core_seconds, context_seconds=context_seconds,
        overlap_seconds=overlap_seconds, shift_seconds=0.0,
        model_root=work_dir / "models", enhancer_factory=enhancer_factory, progress=progress,
    )

    fused_wav = work_dir / "ensemble_pre_master.wav"
    alignments, total_samples = fuse_ensemble_tracks(
        pass1_path, pass2_path, pass3_path, raw44, fused_wav, residual_alpha=residual_alpha
    )
    norm_report = normalize_linear_master(fused_wav, flac_out, target_lufs=-20.0, overwrite=overwrite)
    checks = verify_audio(flac_out)
    if checks["samples"] != total_samples:
        raise ValueError("Export changed the recording's sample count.")
    playback = export_m4a(flac_out, m4a_out, overwrite=overwrite)

    report = {
        **configuration,
        "total_samples": total_samples,
        "duration_seconds": total_samples / SAMPLE_RATE,
        "passes": passes,
        "alignments": alignments,
        "normalization": norm_report,
        "verification": checks,
        "m4a_verification": playback,
        "elapsed_seconds_this_run": time.monotonic() - started,
        "method": (
            "Float32 3-pass RE-USE ensemble (50% 44.1 kHz laughter + 30% 16 kHz +10s staggered laughter "
            "+ 20% 16 kHz standard below 7.2 kHz, 100% 44.1 kHz laughter above 7.2 kHz); "
            "20% speech-gated 180-3600 Hz gentle Wiener archival residual; "
            "constant linear -20 LUFS mastering with -3 dBTP limiter (no dynaudnorm)"
        ),
        "notice": SOURCE_NOTICE,
    }
    write_json(output_dir / f"{ENSEMBLE_OUTPUT_STEM}_metadata.json", report)
    print(f"Restored ensemble master: {flac_out}", flush=True)
    return report
