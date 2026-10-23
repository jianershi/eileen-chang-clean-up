# SPDX-License-Identifier: MIT
"""An AAC/M4A listening copy derived from the lossless processed master."""
import os
from pathlib import Path

from .common import check_output, probe, run, write_json
from .profile import SOURCE_NOTICE, TITLE
from .verify import verify_audio


def export_m4a(source, output, bitrate="192k", overwrite=False):
    source, output = Path(source), Path(output)
    check_output(output, overwrite, source)
    duration = float(probe(source)["format"]["duration"])
    temporary = output.with_name(output.stem + ".partial.m4a")
    run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source,
         "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
         "-c:a", "aac", "-b:a", bitrate, "-movflags", "+faststart",
         "-metadata", f"title={TITLE}", "-metadata", f"comment={SOURCE_NOTICE}", temporary])
    report = verify_audio(temporary)
    if abs(report["duration_seconds"] - duration) > 0.15:
        raise ValueError("M4A duration does not match the lossless master.")
    os.replace(temporary, output)
    report["file"] = output.name
    write_json(output.with_suffix(".json"), {"source": source.name, "notice": SOURCE_NOTICE,
                                           "bitrate": bitrate, "verification": report})
    print(f"Listening copy: {output}", flush=True)
    return report
