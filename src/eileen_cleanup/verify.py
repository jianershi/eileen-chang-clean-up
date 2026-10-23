# SPDX-License-Identifier: MIT
"""Stream-decode and check complete lossless and compressed audio outputs."""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import probe, sha256


def verify_audio(path):
    path = Path(path)
    data = probe(path)
    streams = data["streams"]
    if len(streams) != 1 or streams[0]["codec_type"] != "audio":
        raise ValueError("Expected exactly one audio stream and no other streams.")
    stream = streams[0]
    rate, channels = int(stream["sample_rate"]), int(stream["channels"])
    samples, peak = 0, 0.0
    if path.suffix.lower() == ".flac":
        with sf.SoundFile(path) as audio:
            for block in audio.blocks(blocksize=262_144, dtype="float32", always_2d=True):
                if not np.isfinite(block).all():
                    raise ValueError("Audio contains non-finite samples.")
                samples += len(block)
                peak = max(peak, float(np.max(np.abs(block))))
    else:
        # Stream decoding keeps memory bounded for a long M4A. Frame counting
        # catches a truncated AAC stream hidden by a longer container time.
        process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
             "-map", "0:a:0", "-f", "f32le", "-"], stdout=subprocess.PIPE,
        )
        sample_values = 0
        try:
            while block := process.stdout.read(1_048_576):
                values = np.frombuffer(block, dtype="<f4")
                if not np.isfinite(values).all():
                    raise ValueError("Audio contains non-finite samples.")
                sample_values += len(values)
                peak = max(peak, float(np.max(np.abs(values))))
            if process.wait() != 0:
                raise ValueError("Audio failed a complete decode.")
            samples = sample_values // channels
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    duration = samples / rate
    if samples == 0 or peak >= 1.0:
        raise ValueError("The audio is empty or reaches digital full scale.")
    if abs(duration - float(data["format"]["duration"])) > 0.15:
        raise ValueError("Decoded audio does not cover the declared duration.")
    return {"file": path.name, "samples": samples, "sample_rate": rate,
            "channels": channels, "duration_seconds": float(data["format"]["duration"]),
            "decoded_duration_seconds": duration, "codec": stream["codec_name"],
            "peak": peak, "bytes": path.stat().st_size, "sha256": sha256(path),
            "finite_samples": True, "full_decode": "passed"}
