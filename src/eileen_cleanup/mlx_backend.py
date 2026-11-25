# SPDX-License-Identifier: MIT
"""Float32 Apple Silicon MLX + Metal kernel inference backend for NVIDIA RE-USE checkpoints."""
import math
import platform
from pathlib import Path

import numpy as np

from .model import unwrap_model_config

# Custom Metal kernel for Mamba selective_scan (d_state == 16):
# One GPU thread computes one (b, d) sequence with its 16-element state in registers.
_SELECTIVE_SCAN_SRC = """
    uint tid = thread_position_in_grid.x;
    uint B_sz = u_shape[0];
    uint L = u_shape[1];
    uint D = u_shape[2];
    if (tid >= B_sz * D) return;

    uint b = tid / D;
    uint d = tid % D;

    float a[16];
    float st[16];
    for (uint n = 0; n < 16; ++n) {
        a[n] = A[d * 16 + n];
        st[n] = 0.0f;
    }
    float d_val = D_vec[d];

    uint u_base = b * L * D + d;
    uint bc_base = b * L * 16;

    for (uint t = 0; t < L; ++t) {
        uint idx = u_base + t * D;
        float inp = u[idx];
        float step = delta[idx];
        uint bc_idx = bc_base + t * 16;

        float val = 0.0f;
        for (uint n = 0; n < 16; ++n) {
            float b_val = B_in[bc_idx + n];
            float c_val = C_in[bc_idx + n];
            st[n] = exp(step * a[n]) * st[n] + step * b_val * inp;
            val += st[n] * c_val;
        }
        out[idx] = val + inp * d_val;
    }
"""

_KERNEL = None


def _get_scan_kernel(mx):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mamba_selective_scan_f32",
            input_names=["u", "delta", "A", "B_in", "C_in", "D_vec"],
            output_names=["out"],
            source=_SELECTIVE_SCAN_SRC,
        )
    return _KERNEL


class MlxReuseEnhancer:
    """Float32 Apple Silicon MLX inference matching the CUDA SEMamba forward pass."""

    def __init__(self, directory, sample_rate):
        import mlx.core as mx
        import torch
        from safetensors.torch import load_file

        directory = Path(directory).resolve()
        cfg = unwrap_model_config(directory)
        if cfg["model_cfg"]["d_state"] != 16:
            raise ValueError("MlxReuseEnhancer requires d_state == 16.")

        self.mx = mx
        self.torch = torch
        self.cfg = cfg
        self.sample_rate = int(sample_rate)
        self.num_tfmamba = cfg["model_cfg"]["num_tfmamba"]
        self.d_state = cfg["model_cfg"]["d_state"]
        self.d_conv = cfg["model_cfg"]["d_conv"]
        self.hid = cfg["model_cfg"]["hid_feature"]
        self.dt_rank = math.ceil(self.hid / 16)

        settings = cfg["stft_cfg"]
        self.n_fft = int(round((settings["n_fft"] * sample_rate // settings["sampling_rate"]) / 2)) * 2
        self.hop_size = int(round((settings["hop_size"] * sample_rate // settings["sampling_rate"]) / 2)) * 2
        self.win_size = int(round((settings["win_size"] * sample_rate // settings["sampling_rate"]) / 2)) * 2

        state = load_file(str(directory / "model.safetensors"))
        self.weights = {}
        for key, tensor in state.items():
            array = tensor.float().numpy()
            if array.ndim == 4:
                array = np.transpose(array, (0, 2, 3, 1))
            elif array.ndim == 3:
                array = np.transpose(array, (0, 2, 1))
            self.weights[key] = mx.array(array)
        for key in list(self.weights):
            if key.endswith(".A_log"):
                self.weights[key[:-6] + ".A_neg"] = -mx.exp(self.weights[key])
        mx.eval(self.weights)

        self.kernel = _get_scan_kernel(mx)
        self.device_name = f"Apple Silicon MLX ({platform.machine()})"
        self.torch_version = f"mlx-{mx.__version__} / torch-{torch.__version__}"

    def _selective_scan(self, u, delta, a_neg, b_in, c_in, d_vec):
        mx = self.mx
        batch, length, channels = u.shape
        total_threads = batch * channels
        tg = min(256, total_threads)
        grid = ((total_threads + tg - 1) // tg * tg, 1, 1)
        return self.kernel(
            inputs=[u, delta, a_neg, b_in, c_in, d_vec],
            output_shapes=[(batch, length, channels)],
            output_dtypes=[mx.float32],
            grid=grid,
            threadgroup=(tg, 1, 1),
        )[0]

    def _silu(self, x):
        return x * self.mx.sigmoid(x)

    def _softplus(self, x):
        mx = self.mx
        return mx.where(x > 20.0, x, mx.log1p(mx.exp(mx.minimum(x, 20.0))))

    def _instance_norm_2d(self, x, weight, bias, eps=1e-5):
        mx = self.mx
        mean = mx.mean(x, axis=(1, 2), keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
        return (x - mean) * mx.rsqrt(var + eps) * weight + bias

    def _layer_norm(self, x, weight, bias, eps=1e-5):
        mx = self.mx
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + eps) * weight + bias

    def _prelu(self, x, weight):
        mx = self.mx
        return mx.maximum(x, 0.0) + weight * mx.minimum(x, 0.0)

    def _linear(self, x, weight, bias=None):
        y = x @ weight.T
        return y + bias if bias is not None else y

    def _conv2d(self, x, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        y = self.mx.conv2d(x, weight, stride=stride, padding=padding, dilation=dilation)
        return y + bias if bias is not None else y

    def _conv_norm_prelu(self, x, prefix, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        w = self.weights
        x = self._conv2d(x, w[f"{prefix}.0.weight"], w[f"{prefix}.0.bias"],
                         stride=stride, padding=padding, dilation=dilation)
        x = self._instance_norm_2d(x, w[f"{prefix}.1.weight"], w[f"{prefix}.1.bias"])
        return self._prelu(x, w[f"{prefix}.2.weight"])

    def _dense_block(self, x, prefix):
        mx = self.mx
        skip = x
        for i in range(4):
            dil = 2 ** i
            pad_h = (3 * dil - dil) // 2
            x = self._conv_norm_prelu(skip, f"{prefix}.dense_block.{i}",
                                      padding=(pad_h, 1), dilation=(dil, 1))
            skip = mx.concatenate([x, skip], axis=-1)
        return x

    def _sp_conv_transpose(self, x, prefix, factor):
        mx = self.mx
        w = self.weights
        x = mx.pad(x, [(0, 0), (0, 0), (1, 1), (0, 0)])
        out = self._conv2d(x, w[f"{prefix}.0.conv.weight"], w[f"{prefix}.0.conv.bias"])
        n, height, width, channels = out.shape
        out = mx.reshape(out, (n, height, width * factor, channels // factor))
        out = self._instance_norm_2d(out, w[f"{prefix}.1.weight"], w[f"{prefix}.1.bias"])
        return self._prelu(out, w[f"{prefix}.2.weight"])

    def _mamba_single(self, hidden, prefix):
        mx = self.mx
        w = self.weights
        xz = self._linear(hidden, w[f"{prefix}.in_proj.weight"])
        d_inner = xz.shape[-1] // 2
        x, z = xz[..., :d_inner], xz[..., d_inner:]
        x_pad = mx.pad(x, [(0, 0), (self.d_conv - 1, 0), (0, 0)])
        x = mx.conv1d(x_pad, w[f"{prefix}.conv1d.weight"], groups=d_inner) + w[f"{prefix}.conv1d.bias"]
        x = self._silu(x)
        projected = self._linear(x, w[f"{prefix}.x_proj.weight"])
        dt = projected[..., :self.dt_rank]
        b_in = projected[..., self.dt_rank:self.dt_rank + self.d_state]
        c_in = projected[..., self.dt_rank + self.d_state:]
        delta = self._softplus(self._linear(dt, w[f"{prefix}.dt_proj.weight"], w[f"{prefix}.dt_proj.bias"]))
        y = self._selective_scan(x, delta, w[f"{prefix}.A_neg"], b_in, c_in, w[f"{prefix}.D"])
        return self._linear(y * self._silu(z), w[f"{prefix}.out_proj.weight"])

    def _mamba_bidir(self, x, prefix):
        mx = self.mx
        w = self.weights
        out_fw = self._mamba_single(x, f"{prefix}.forward_blocks") + x
        x_rev = x[:, ::-1, :]
        out_bw = (self._mamba_single(x_rev, f"{prefix}.backward_blocks") + x_rev)[:, ::-1, :]
        out = mx.concatenate([out_fw, out_bw], axis=-1)
        out = self._linear(out, w[f"{prefix}.output_proj.weight"], w[f"{prefix}.output_proj.bias"])
        return self._layer_norm(out, w[f"{prefix}.norm.weight"], w[f"{prefix}.norm.bias"])

    def _forward_stft(self, noisy_mag, noisy_pha):
        mx = self.mx
        w = self.weights
        mag = mx.array(np.transpose(noisy_mag, (0, 2, 1))[..., None])
        pha = mx.array(np.transpose(noisy_pha, (0, 2, 1))[..., None])
        x = mx.concatenate([mag, pha], axis=-1)
        _, time_steps, freq_bins, _ = x.shape
        x = mx.pad(x, [(0, 0), (0, 2), (0, 2), (0, 0)])

        x = self._conv_norm_prelu(x, "dense_encoder.dense_conv_1")
        x = self._dense_block(x, "dense_encoder.dense_block")
        x = self._conv_norm_prelu(x, "dense_encoder.dense_conv_2", stride=(4, 2))

        batch, t_enc, f_enc, channels = x.shape
        for i in range(self.num_tfmamba):
            xt = mx.reshape(mx.transpose(x, (0, 2, 1, 3)), (batch * f_enc, t_enc, channels))
            xt = self._mamba_bidir(xt, f"TSMamba.{i}.time_mamba") + xt
            xf = mx.reshape(mx.transpose(mx.reshape(xt, (batch, f_enc, t_enc, channels)), (0, 2, 1, 3)),
                            (batch * t_enc, f_enc, channels))
            xf = self._mamba_bidir(xf, f"TSMamba.{i}.freq_mamba") + xf
            x = mx.reshape(xf, (batch, t_enc, f_enc, channels))
            if (i + 1) % 6 == 0:
                mx.eval(x)

        xm = self._dense_block(x, "mask_decoder.dense_block")
        xm = self._sp_conv_transpose(xm, "mask_decoder.up_conv1", factor=2)
        xm = mx.transpose(xm, (0, 2, 1, 3))
        xm = self._sp_conv_transpose(xm, "mask_decoder.up_conv2", factor=4)
        xm = mx.transpose(xm, (0, 2, 1, 3))
        xm = self._conv2d(xm, w["mask_decoder.final_conv.weight"], w["mask_decoder.final_conv.bias"])

        xp = self._dense_block(x, "phase_decoder.dense_block")
        xp = self._sp_conv_transpose(xp, "phase_decoder.up_conv1", factor=2)
        xp = mx.transpose(xp, (0, 2, 1, 3))
        xp = self._sp_conv_transpose(xp, "phase_decoder.up_conv2", factor=4)
        xp = mx.transpose(xp, (0, 2, 1, 3))
        xr = self._conv2d(xp, w["phase_decoder.phase_conv_r.weight"], w["phase_decoder.phase_conv_r.bias"])
        xi = self._conv2d(xp, w["phase_decoder.phase_conv_i.weight"], w["phase_decoder.phase_conv_i.bias"])
        xp = mx.arctan2(xi, xr)

        dm = mx.transpose(xm[:, :time_steps, :freq_bins, 0], (0, 2, 1))
        dp = mx.transpose(xp[:, :time_steps, :freq_bins, 0], (0, 2, 1))
        mx.eval(dm, dp)
        return np.array(dm, dtype=np.float32), np.array(dp, dtype=np.float32)

    def __call__(self, audio):
        torch = self.torch
        audio = np.asarray(audio, dtype=np.float32)
        wave = torch.from_numpy(audio)[None]
        window = torch.hann_window(self.win_size)
        spec = torch.stft(wave, self.n_fft, hop_length=self.hop_size, win_length=self.win_size,
                          window=window, center=True, pad_mode="reflect",
                          normalized=False, return_complex=True)
        noisy_mag = torch.log1p(torch.abs(spec)).numpy()
        noisy_pha = torch.angle(spec).numpy()

        dm, dp = self._forward_stft(noisy_mag, noisy_pha)
        dm_t = torch.from_numpy(dm)
        dp_t = torch.from_numpy(dp)
        # Retain the upstream silence suppression convention.
        silent_bins = torch.expm1(torch.relu(dm_t)).eq(0).float().mean(dim=1)
        dm_t.masked_fill_((silent_bins > 0.5).unsqueeze(1), 0)
        lin_mag = torch.expm1(torch.relu(dm_t))
        complex_spec = torch.complex(lin_mag * torch.cos(dp_t), lin_mag * torch.sin(dp_t))
        result = torch.istft(complex_spec, self.n_fft, hop_length=self.hop_size,
                             win_length=self.win_size, window=window, center=True)
        output = result.squeeze(0).numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("RE-USE MLX backend produced non-finite samples; no checkpoint was saved.")
        return output
