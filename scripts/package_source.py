# SPDX-License-Identifier: MIT
"""Create an allowlisted source ZIP for Colab; never package research/media caches."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT_FILES = {
    "README.md", "LICENSE", "pyproject.toml",
    ".gitignore", ".gitattributes", "MANIFEST.in", "REUSE_full_recording_GPU.ipynb",
}
TREES = {"src", "scripts", "tests", "docs", ".github"}
SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json"}


def source_files(root):
    root = Path(root)
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for tree in TREES:
        files.extend(path for path in (root / tree).rglob("*")
                     if path.is_file() and not path.is_symlink()
                     and path.suffix in SUFFIXES
                     and not any(part.startswith(".") or part == "__pycache__"
                                 for part in path.relative_to(root / tree).parts))
    return sorted(files)


def package(root, output):
    root, output = Path(root).resolve(), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in source_files(root):
            if path.is_symlink():
                raise ValueError(f"Refusing symlink: {path.name}")
            archive.write(path, "eileen-chang-clean-up/" + path.relative_to(root).as_posix())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/eileen-chang-clean-up-source.zip"))
    args = parser.parse_args()
    print(package(Path(__file__).resolve().parents[1], args.output))
