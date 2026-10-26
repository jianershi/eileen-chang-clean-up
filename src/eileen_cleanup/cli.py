# SPDX-License-Identifier: MIT
"""Audio-only command-line entry points; GPU imports are deferred."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .profile import MODEL_REVISION, OUTPUT_STEM


def parser():
    root = argparse.ArgumentParser(description="Clean archival audio with RE-USE; export FLAC and M4A.")
    commands = root.add_subparsers(dest="command", required=True)
    restore = commands.add_parser("restore", help="Run/resume GPU enhancement and export FLAC and M4A.")
    restore.add_argument("source", type=Path)
    restore.add_argument("--output-dir", type=Path, default=Path("outputs"))
    restore.add_argument("--work-dir", type=Path, default=Path("work/nemotron-restoration"))
    restore.add_argument("--core-seconds", type=float, default=20.)
    restore.add_argument("--context-seconds", type=float, default=2.)
    restore.add_argument("--overlap-seconds", type=float, default=1.)
    restore.add_argument("--model-revision", default=MODEL_REVISION, dest="revision")
    restore.add_argument("--expected-duration", type=float,
                         help="Optional expected duration, in seconds, for your source.")
    restore.add_argument("--overwrite", action="store_true")
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
