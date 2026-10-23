# SPDX-License-Identifier: MIT
"""Small process, metadata and atomic-file helpers."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def require_program(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} is required; install FFmpeg and ffprobe first.")


def run(command, **kwargs):
    """Use argument lists, never a shell, for paths and metadata supplied by a user."""
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def probe(path):
    require_program("ffprobe")
    result = run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check_output(path, overwrite=False, source=None):
    path = Path(path)
    if source is not None and path.resolve() == Path(source).resolve():
        raise ValueError("An output must not replace its source recording.")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def decode_check(path):
    require_program("ffmpeg")
    run(["ffmpeg", "-nostdin", "-v", "error", "-i", path,
         "-map", "0:a:0", "-f", "null", "-"], capture_output=True)
