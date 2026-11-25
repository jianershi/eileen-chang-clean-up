# SPDX-License-Identifier: MIT
"""Audio-only command-line entry points; GPU/MLX imports are deferred."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .profile import MODEL_REVISION, OUTPUT_STEM


def parser():
    root = argparse.ArgumentParser(description="Clean archival audio with RE-USE; export FLAC and M4A.")
    commands = root.add_subparsers(dest="command", required=True)

    restore = commands.add_parser("restore", help="Run/resume single-pass RE-USE enhancement and export FLAC and M4A.")
    restore.add_argument("source", type=Path)
    restore.add_argument("--output-dir", type=Path, default=Path("outputs"))
    restore.add_argument("--work-dir", type=Path, default=Path("work/nemotron-restoration"))
    restore.add_argument("--core-seconds", type=float, default=20.)
    restore.add_argument("--context-seconds", type=float, default=2.)
    restore.add_argument("--overlap-seconds", type=float, default=1.)
    restore.add_argument("--model-revision", default=MODEL_REVISION, dest="revision")
    restore.add_argument("--backend", choices=("auto", "cuda", "mlx"), default="auto")
    restore.add_argument("--expected-duration", type=float,
                         help="Optional expected duration, in seconds, for your source.")
    restore.add_argument("--overwrite", action="store_true")

    ensemble = commands.add_parser(
        "ensemble",
        help="Run/resume the 3-pass multi-resolution RE-USE ensemble + gated archival residual and export FLAC and M4A.",
    )
    ensemble.add_argument("source", type=Path)
    ensemble.add_argument("--output-dir", type=Path, default=Path("outputs"))
    ensemble.add_argument("--work-dir", type=Path, default=Path("work/reuse-ensemble"))
    ensemble.add_argument("--backend", choices=("auto", "cuda", "mlx"), default="auto")
    ensemble.add_argument("--core-seconds", type=float, default=20.)
    ensemble.add_argument("--context-seconds", type=float, default=2.)
    ensemble.add_argument("--overlap-seconds", type=float, default=1.)
    ensemble.add_argument("--shift-seconds", type=float, default=10.)
    ensemble.add_argument("--residual-alpha", type=float, default=0.20)
    ensemble.add_argument("--pass1-joined", type=Path,
                          help="Optional pre-computed float32 44.1 kHz laughter joined.wav to reuse for Pass 1.")
    ensemble.add_argument("--expected-duration", type=float,
                          help="Optional expected duration, in seconds, for your source.")
    ensemble.add_argument("--overwrite", action="store_true")

    finalize = commands.add_parser("finalize", help="Join completed checkpoints and export both audio formats on a CPU.")
    finalize.add_argument("--work-dir", type=Path, default=Path("work/nemotron-restoration"))
    finalize.add_argument("--output-dir", type=Path, default=Path("outputs"))
    finalize.add_argument("--overwrite", action="store_true")

    playback = commands.add_parser("m4a", help="Encode a smaller AAC/M4A copy from an existing restored FLAC.")
    playback.add_argument("source", type=Path)
    playback.add_argument("--output", type=Path, default=Path("outputs") / f"{OUTPUT_STEM}.m4a")
    playback.add_argument("--bitrate", default="192k")
    playback.add_argument("--overwrite", action="store_true")

    verify = commands.add_parser("verify", help="Decode and check a complete FLAC or M4A.")
    verify.add_argument("source", type=Path)
    return root


def main(argv=None):
    arguments = vars(parser().parse_args(argv))
    command = arguments.pop("command")
    try:
        if command == "restore":
            from .restore import restore_audio
            restore_audio(**arguments)
        elif command == "ensemble":
            from .ensemble import restore_ensemble
            restore_ensemble(**arguments)
        elif command == "finalize":
            from .restore import finalize_audio
            finalize_audio(**arguments)
        elif command == "m4a":
            from .export import export_m4a
            export_m4a(**arguments)
        else:
            from .verify import verify_audio
            print(json.dumps(verify_audio(arguments["source"]), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
