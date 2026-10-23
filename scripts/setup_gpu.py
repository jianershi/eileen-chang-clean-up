# SPDX-License-Identifier: MIT
"""Install the historical Colab CUDA stack and matching official Mamba wheels.

Run this in a fresh Python process, before importing torch in the notebook.
No upstream source is copied into this project and no source build is attempted.
"""
import argparse
import json
import platform
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

PACKAGES = (
    ("Dao-AILab/causal-conv1d", "v1.7.0", "causal_conv1d"),
    ("state-spaces/mamba", "v2.3.2.post1", "mamba_ssm"),
)


def matching_wheel(assets, package, python_tag, torch_version, cuda_major, abi, architecture):
    marker = f"cu{cuda_major}torch{torch_version}cxx11abi{str(abi).upper()}"
    suffix = f"-{python_tag}-{python_tag}-linux_{architecture}.whl"
    matches = [a for a in assets if a["name"].startswith(package + "-")
               and marker in a["name"] and a["name"].endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one official {package} wheel for {marker} {suffix}; found {len(matches)}. "
                           "Use Linux x86_64 and a supported Python version; no source build was started.")
    url = matches[0]["browser_download_url"]
    if not url.startswith("https://github.com/"):
        raise RuntimeError("The release did not return an official GitHub download URL.")
    return matches[0]


def pip(*arguments):
    subprocess.run([sys.executable, "-m", "pip", *arguments], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-torch", action="store_true",
                        help="Install Torch 2.10.0 / CUDA 13.0, as used by the completed T4 run.")
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("The CUDA setup supports Linux x86_64. CPU export/verification also works on macOS.")
    if args.install_torch:
        pip("install", "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0",
            "--index-url", "https://download.pytorch.org/whl/cu130")
    root = Path(__file__).resolve().parents[1]
    pip("install", "-e", str(root) + "[gpu]")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A working NVIDIA CUDA GPU is required; choose a GPU runtime in Colab.")
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_version = re.match(r"\d+\.\d+", torch.__version__).group()
    cuda_major = int(torch.version.cuda.split(".")[0])
    abi = torch._C._GLIBCXX_USE_CXX11_ABI
    wheels = []
    for repository, tag, package in PACKAGES:
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repository}/releases/tags/{tag}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "eileen-chang-clean-up"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            release = json.load(response)
        wheel = matching_wheel(release["assets"], package, python_tag, torch_version,
                               cuda_major, abi, platform.machine())
        pip("install", "--no-deps", "--force-reinstall", wheel["browser_download_url"])
        wheels.append({"package": package, "tag": tag, "wheel": wheel["name"],
                       "url": wheel["browser_download_url"]})
    # Exercise the native kernels, rather than accepting import success alone.
    from mamba_ssm import Mamba

    with torch.inference_mode():
        module = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda().eval()
        result = module(torch.randn(1, 32, 16, device="cuda"))
        if not torch.isfinite(result).all():
            raise RuntimeError("Native Mamba CUDA smoke check returned non-finite values.")
    report = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "wheels": wheels}
    report_path = root / "work" / "gpu-environment.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("GPU setup passed. If torch was already imported in your notebook, restart the runtime before using it.")


if __name__ == "__main__":
    main()
