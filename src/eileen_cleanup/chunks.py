# SPDX-License-Identifier: MIT
"""Bounded-memory, sample-aligned joining of overlapping enhancement windows."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    read_start: int
    read_end: int

    def to_dict(self):
        return asdict(self)


def plan_chunks(total, rate, core_seconds=20.0, context_seconds=2.0, overlap_seconds=1.0):
    if total <= 0 or rate <= 0:
        raise ValueError("The recording must contain audio samples at a positive rate.")
    core = round(core_seconds * rate)
    context = round(context_seconds * rate)
    overlap = round(overlap_seconds * rate)
    if overlap < 1 or context < overlap or core <= 2 * overlap:
        raise ValueError("Require core > 2 × overlap > 0 and context >= overlap.")
    return [Chunk(i, start, min(total, start + core), max(0, start - context),
                  min(total, start + core + context))
            for i, start in enumerate(range(0, total, core))]


def checkpoint_paths(directory, index):
    directory = Path(directory)
    return directory / f"{index:05d}.wav", directory / f"{index:05d}.json"


def join_chunks(rows, directory, output, total, rate, overlap_seconds=1.0):
    """Crossfade each aligned pair; also handle a final core shorter than a fade."""
    if not rows or rows[0]["start"] != 0 or rows[-1]["end"] != total:
        raise ValueError("Chunk coverage must begin at zero and end at the source length.")
    for left, right in zip(rows, rows[1:]):
        if left["end"] != right["start"]:
            raise ValueError("Chunk cores contain a gap or overlap.")
    overlap = round(overlap_seconds * rate)
    if overlap < 1:
        raise ValueError("The crossfade must contain at least one sample.")
    pending = np.zeros(0, np.float32)
    weights = np.zeros(0, np.float32)
    written = 0
    with sf.SoundFile(output, "w", samplerate=rate, channels=1, subtype="FLOAT", format="WAV") as writer:
        for row in rows:
            path, _ = checkpoint_paths(directory, row["index"])
            data, saved_rate = sf.read(path, dtype="float32", always_2d=False)
            if saved_rate != rate or data.ndim != 1 or len(data) != row["read_end"] - row["read_start"]:
                raise ValueError(f"Incorrect chunk shape or sample rate: {path}")
            gain = float(row["gain"])
            if not np.isfinite(gain) or gain <= 0 or not np.isfinite(data).all():
                raise ValueError(f"Non-finite audio or invalid gain: {path}")
            lo = max(0, row["start"] - overlap)
            hi = min(total, row["end"] + overlap)
            if lo != written:
                raise ValueError("Unexpected crossfade position.")
            data = data[lo - row["read_start"]:hi - row["read_start"]] / gain
            t = np.arange(lo, hi, dtype=np.float64)
            weight = np.ones(len(data), np.float32)
            if row["start"] > 0:
                phase = np.clip((t - (row["start"] - overlap)) / (2 * overlap), 0, 1)
                weight *= (0.5 - 0.5 * np.cos(np.pi * phase)).astype(np.float32)
            if row["end"] < total:
                phase = np.clip((t - (row["end"] - overlap)) / (2 * overlap), 0, 1)
                weight *= (0.5 + 0.5 * np.cos(np.pi * phase)).astype(np.float32)
            if len(pending) < len(data):
                padding = len(data) - len(pending)
                pending = np.pad(pending, (0, padding))
                weights = np.pad(weights, (0, padding))
            pending[:len(data)] += data * weight
            weights[:len(data)] += weight
            flush_end = total if row["end"] == total else row["end"] - overlap
            count = flush_end - written
            if count <= 0 or np.any(weights[:count] <= 0):
                raise ValueError("An uncovered sample was found while joining.")
            block = pending[:count] / weights[:count]
            if not np.isfinite(block).all():
                raise ValueError("Joining produced non-finite samples.")
            writer.write(block)
            pending, weights = pending[count:].copy(), weights[count:].copy()
            written = flush_end
    if written != total or len(pending):
        raise ValueError("Joined output length does not match the source.")
    return written
