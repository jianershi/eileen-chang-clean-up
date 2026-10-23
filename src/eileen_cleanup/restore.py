# SPDX-License-Identifier: MIT
"""Resumable inference, bounded-memory joining and a lossless normalized master."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .chunks import Chunk, checkpoint_paths, join_chunks, plan_chunks
from .common import check_output, probe, require_program, run, sha256, write_json
from .profile import MODEL_REPOSITORY, MODEL_REVISION, SAMPLE_RATE, SOURCE_NOTICE, TITLE

PIPELINE_VERSION = 1


def bind_run(directory, configuration):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != configuration:
            raise ValueError("This work directory belongs to a different source or configuration. Use a new --work-dir.")
    else:
        # An interrupted or manually copied cache without its run identity is
        # not safe to attach to a new recording.
        if any(directory.glob("chunks/*.wav")) or (directory / "prepared.wav").exists():
            raise ValueError("The work directory contains audio without a run identity. Use a new --work-dir.")
        write_json(path, configuration)


def valid_checkpoint(directory, chunk, rate):
    audio, metadata = checkpoint_paths(directory, chunk.index)
    if not (audio.exists() and metadata.exists()):
        return None
    try:
        row = json.loads(metadata.read_text())
        if any(row.get(key) != value for key, value in chunk.to_dict().items()):
            return None
        info = sf.info(audio)
        if (row["sample_rate"] != rate or info.samplerate != rate or info.channels != 1
                or info.frames != chunk.read_end - chunk.read_start
                or not math.isfinite(row["gain"]) or row["gain"] <= 0
                or row["sha256"] != sha256(audio)):
            return None
        return row
    except (ValueError, OSError, KeyError, TypeError, RuntimeError):
        return None


def enhance_chunks(prepared, chunks, directory, rate, enhancer, progress=None):
    """An injectable enhancer keeps checkpoint and join tests independent of CUDA."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with sf.SoundFile(prepared) as source:
        for chunk in chunks:
            row = valid_checkpoint(directory, chunk, rate)
            reused = row is not None
            if row is None:
                if enhancer is None:
                    raise ValueError(f"Missing or invalid enhancement checkpoint: {chunk.index}")
                source.seek(chunk.read_start)
                audio = source.read(chunk.read_end - chunk.read_start, dtype="float32")
                peak = float(np.max(np.abs(audio)))
                gain = min(20.0, 0.85 / max(peak, 1e-6))
                clean = np.asarray(enhancer(audio * gain), dtype=np.float32)
                if clean.ndim != 1 or not np.isfinite(clean).all():
                    raise ValueError("An enhancement window has invalid shape or non-finite samples.")
                predicted_length = len(clean)
                clean = np.pad(clean, (0, max(0, len(audio) - len(clean))))[:len(audio)]
                path, metadata = checkpoint_paths(directory, chunk.index)
                temporary = path.with_name(path.stem + ".partial.wav")
                # Float checkpoints avoid the legacy notebook's additional
                # PCM16 quantization and clipping before gain is undone.
                sf.write(temporary, clean, rate, subtype="FLOAT", format="WAV")
                os.replace(temporary, path)
                row = {**chunk.to_dict(), "sample_rate": rate, "gain": gain,
                       "predicted_samples": predicted_length, "sha256": sha256(path)}
                write_json(metadata, row)
            rows.append(row)
            if progress:
                progress({"chunk": chunk.index + 1, "chunks": len(chunks),
                          "completed_seconds": chunk.end / rate, "resumed": reused})
    return rows


def normalize_master(source, output, overwrite=False):
    check_output(output, overwrite, source)
    level = "dynaudnorm=f=500:g=31:p=0.85:m=8:r=0.08:t=0.0001"
    target = "loudnorm=I=-20:TP=-3:LRA=9"
    measured = run(["ffmpeg", "-nostdin", "-hide_banner", "-i", source,
                    "-af", level + "," + target + ":print_format=json",
                    "-f", "null", "-"], capture_output=True, text=True)
    text = measured.stderr
    values = json.loads(text[text.rfind("{"):text.rfind("}") + 1])
    keys = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if not all(math.isfinite(float(values[key])) for key in keys):
        raise ValueError("Cannot normalize silent or non-finite audio.")
    filters = (level + "," + target +
               ":measured_I={input_i}:measured_TP={input_tp}:measured_LRA={input_lra}"
               ":measured_thresh={input_thresh}:offset={target_offset}:linear=true").format(**values)
    filters += ",aresample=44100,alimiter=limit=0.708:attack=5:release=50:level=false:latency=true"
    output = Path(output)
    temporary = output.with_name(output.stem + ".partial.flac")
    run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source,
         "-af", filters, "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "flac",
         "-sample_fmt", "s32", "-metadata", f"title={TITLE}",
         "-metadata", f"comment={SOURCE_NOTICE}", temporary])
    os.replace(temporary, output)
    return values


def restore_audio(source, output_dir, work_dir, core_seconds=20.0, context_seconds=2.0,
                  overlap_seconds=1.0, revision=MODEL_REVISION,
                  expected_duration=None, overwrite=False):
    from .model import ReuseEnhancer, download_model

    require_program("ffmpeg")
    source, output_dir, work_dir = Path(source).resolve(), Path(output_dir), Path(work_dir)
    output = output_dir / "NVIDIA_REUSE_full_recording.flac"
    check_output(output, overwrite, source)
    check_output(output_dir / "NVIDIA_REUSE_full_recording.m4a", overwrite, source)
    source_hash = sha256(source)
    duration = float(probe(source)["format"]["duration"])
    if expected_duration is not None and abs(duration - expected_duration) > 1:
        raise ValueError("The input duration does not match the expected recording.")
    configuration = {"pipeline_version": PIPELINE_VERSION, "source_name": source.name,
                     "source_sha256": source_hash, "sample_rate": SAMPLE_RATE,
                     "highpass_hz": 60, "core_seconds": core_seconds,
                     "context_seconds": context_seconds, "overlap_seconds": overlap_seconds,
                     "model": MODEL_REPOSITORY, "model_revision": revision,
                     "checkpoint_format": "float32 WAV"}
    bind_run(work_dir, configuration)
    prepared = work_dir / "prepared.wav"
    if not prepared.exists():
        temporary = work_dir / "prepared.partial.wav"
        run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source,
             "-af", "highpass=f=60", "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-c:a", "pcm_f32le", temporary])
        os.replace(temporary, prepared)
    info = sf.info(prepared)
    if info.channels != 1 or info.samplerate != SAMPLE_RATE or abs(info.duration - duration) > 1:
        raise ValueError("The prepared audio is inconsistent with the source.")
    chunks = plan_chunks(info.frames, SAMPLE_RATE, core_seconds, context_seconds, overlap_seconds)
    checkpoint_dir = work_dir / "chunks"
    pending = any(valid_checkpoint(checkpoint_dir, chunk, SAMPLE_RATE) is None for chunk in chunks)
    started = time.monotonic()
    previous_manifest = work_dir / "manifest.json"
    previous = json.loads(previous_manifest.read_text()) if previous_manifest.exists() else {}
    enhancer, model_hash = None, previous.get("model_weights_sha256")
    environment = previous.get("environment", {})
    if pending:
        model_dir, model_hash = download_model(work_dir / "model", revision)
        enhancer = ReuseEnhancer(model_dir, SAMPLE_RATE)
        environment = {"gpu": enhancer.device_name, "torch": enhancer.torch_version}
    rows = enhance_chunks(prepared, chunks, checkpoint_dir, SAMPLE_RATE, enhancer,
                          progress=lambda row: print(json.dumps(row), flush=True))
    del enhancer
    manifest = {**configuration, "total_samples": info.frames, "duration_seconds": info.duration,
                "model_weights_sha256": model_hash, "environment": environment, "chunks": rows}
    write_json(work_dir / "manifest.json", manifest)
    return finalize_audio(work_dir, output_dir, overwrite=overwrite,
                          elapsed_seconds=time.monotonic() - started)


def finalize_audio(work_dir, output_dir, overwrite=False, elapsed_seconds=None):
    """Joining and export can run on a CPU after all GPU windows have finished."""
    work_dir, output_dir = Path(work_dir), Path(output_dir)
    manifest = json.loads((work_dir / "manifest.json").read_text())
    output = output_dir / "NVIDIA_REUSE_full_recording.flac"
    check_output(output, overwrite)
    check_output(output_dir / "NVIDIA_REUSE_full_recording.m4a", overwrite)
    rows = []
    for original in manifest["chunks"]:
        chunk = Chunk(**{key: original[key] for key in Chunk.__dataclass_fields__})
        row = valid_checkpoint(work_dir / "chunks", chunk, manifest["sample_rate"])
        if row is None:
            raise ValueError(f"Incomplete or corrupt checkpoint {chunk.index}; run restore again.")
        rows.append(row)
    master = work_dir / "joined.partial.wav"
    join_chunks(rows, work_dir / "chunks", master, manifest["total_samples"],
                manifest["sample_rate"], manifest["overlap_seconds"])
    values = normalize_master(master, output, overwrite=overwrite)
    from .verify import verify_audio

    checks = verify_audio(output)
    if checks["samples"] != manifest["total_samples"]:
        raise ValueError("Export changed the recording's sample count.")
    from .export import export_m4a

    playback = export_m4a(output, output_dir / "NVIDIA_REUSE_full_recording.m4a", overwrite=overwrite)
    report = {key: value for key, value in manifest.items() if key != "chunks"}
    report.update({"normalization_measurements": values, "verification": checks,
                   "m4a_verification": playback,
                   "elapsed_seconds_this_run": elapsed_seconds,
                   "method": "float32 CUDA RE-USE; 60 Hz high-pass; aligned raised-cosine crossfades; two-pass loudness normalization",
                   "notice": SOURCE_NOTICE})
    write_json(output_dir / "Restoration_metadata.json", report)
    print(f"Restored master: {output}", flush=True)
    return report
