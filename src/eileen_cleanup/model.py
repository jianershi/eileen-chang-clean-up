# SPDX-License-Identifier: MIT
"""Bind to the separately downloaded NVIDIA model; no model code is vendored."""
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from .common import sha256
from .profile import MODEL_REPOSITORY, MODEL_REVISION, MODEL_WEIGHTS_SHA256


def download_model(directory, revision=MODEL_REVISION):
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Use a full model commit hash, not a moving branch name.")
    from huggingface_hub import snapshot_download

    directory = Path(snapshot_download(
        repo_id=MODEL_REPOSITORY, revision=revision, local_dir=str(directory),
        allow_patterns=["models/*.py", "utils/*.py", "recipes/*.yaml", "config.json",
                        "model.safetensors", "README*", "LICENSE*"],
    ))
    digest = sha256(directory / "model.safetensors")
    if revision == MODEL_REVISION and digest != MODEL_WEIGHTS_SHA256:
        raise ValueError("The downloaded model weights do not match the pinned checkpoint.")
    return directory, digest


class ReuseEnhancer:
    """Full-precision CUDA inference using the upstream magnitude/phase interfaces."""

    def __init__(self, directory, sample_rate):
        import torch
        from safetensors.torch import load_file

        if not torch.cuda.is_available():
            raise RuntimeError("RE-USE needs a CUDA GPU. In Colab, select a T4 GPU runtime.")
        directory = Path(directory).resolve()
        # These names belong to NVIDIA's downloaded package. Avoid accidentally
        # binding to an unrelated top-level 'models' package in a notebook.
        for name in ("models", "models.generator_SEMamba_time_d4", "models.stfts"):
            existing = sys.modules.get(name)
            location = getattr(existing, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(directory):
                raise RuntimeError("Another 'models' package is loaded; use a fresh Python process.")
        sys.path.insert(0, str(directory))
        generator = importlib.import_module("models.generator_SEMamba_time_d4")
        transforms = importlib.import_module("models.stfts")
        cfg = json.loads((directory / "config.json").read_text())
        self.torch = torch
        self.network = generator.SEMamba(cfg)
        self.network.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        self.network = self.network.to(device="cuda", dtype=torch.float32).eval()
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        settings = cfg["stft_cfg"]
        self.parameters = {
            key: int(round((settings[key] * sample_rate // settings["sampling_rate"]) / 2)) * 2
            for key in ("n_fft", "hop_size", "win_size")
        }
        self.parameters["compress_factor"] = cfg["model_cfg"]["compress_factor"]
        self.stft, self.istft = transforms.mag_phase_stft, transforms.mag_phase_istft
        self.device_name = torch.cuda.get_device_name(0)
        self.torch_version = torch.__version__

    def __call__(self, audio):
        torch = self.torch
        wave = torch.from_numpy(np.asarray(audio, dtype=np.float32))[None].to("cuda")
        try:
            with torch.inference_mode():
                magnitude, phase, _ = self.stft(wave, **self.parameters, center=True, addeps=False)
                enhanced_magnitude, enhanced_phase, _ = self.network(magnitude, phase)
                # Retain the upstream demo's silence suppression convention.
                silent_bins = torch.expm1(torch.relu(enhanced_magnitude)).eq(0).float().mean(dim=1)
                enhanced_magnitude.masked_fill_((silent_bins > 0.5).unsqueeze(1), 0)
                result = self.istft(enhanced_magnitude, enhanced_phase, **self.parameters)
                result = result.squeeze(0).cpu().numpy()
            if not np.isfinite(result).all():
                raise RuntimeError("RE-USE produced non-finite samples; no checkpoint was saved.")
            return result
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "GPU memory exhausted. Retry with --core-seconds 10 and a new --work-dir."
            ) from error
        finally:
            del wave
            torch.cuda.empty_cache()
