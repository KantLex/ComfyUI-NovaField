"""Small torch helpers shared by the NovaField nodes (no ComfyUI imports at module level)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    """ComfyUI's torch device when running inside ComfyUI, else CUDA if available, else CPU."""
    try:  # lazy: tests run outside ComfyUI
        import comfy.model_management as mm  # type: ignore

        return mm.get_torch_device()
    except Exception:
        pass
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def hex_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luma(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Rec.709 luma of an RGB tensor along ``dim`` (kept)."""
    w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=x.dtype, device=x.device)
    shape = [1] * x.dim()
    shape[dim] = 3
    return (x * w.view(shape)).sum(dim=dim, keepdim=True)


def gaussian_kernel1d(sigma: float, device, dtype=torch.float32, shift: float = 0.0) -> torch.Tensor:
    """Normalised 1D gaussian. ``shift`` > 0 moves the image content towards +x / +y."""
    r = max(1, int(math.ceil(3.0 * sigma + abs(shift))))
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)
    k = torch.exp(-((x + shift) ** 2) / (2.0 * max(sigma, 1e-3) ** 2))
    return k / k.sum()


def _conv_axis(x: torch.Tensor, k: torch.Tensor, axis: str) -> torch.Tensor:
    """Per-channel 1D convolution of NCHW ``x`` along 'x' (width) or 'y' (height), replicate pad."""
    N, C, H, W = x.shape
    K = k.numel()
    r = K // 2
    n_axis = W if axis == "x" else H
    if r > n_axis - 1:  # kernel wider than the image: fall back to a mean along the axis
        return x.mean(dim=3 if axis == "x" else 2, keepdim=True).expand_as(x).contiguous()
    if K <= 11:  # short kernels: shifted multiply-adds beat conv2d by far on CPU
        if axis == "x":
            xp = F.pad(x, (r, r, 0, 0), mode="replicate")
            out = xp[..., 0:W] * k[0]
            for i in range(1, K):
                out = out + xp[..., i : i + W] * k[i]
        else:
            xp = F.pad(x, (0, 0, r, r), mode="replicate")
            out = xp[..., 0:H, :] * k[0]
            for i in range(1, K):
                out = out + xp[..., i : i + H, :] * k[i]
        return out
    # channels folded into the batch: a single-channel conv is much faster than groups=C on CPU
    y = x.reshape(N * C, 1, H, W)
    if axis == "x":
        y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, K))
    else:
        y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, K, 1))
    return y.view(N, C, H, W)


def blur(x: torch.Tensor, sigma_x: float, sigma_y: float) -> torch.Tensor:
    """Separable gaussian blur of NCHW ``x``. Large sigmas run on a downsampled copy."""
    H, W = x.shape[2], x.shape[3]
    dx = max(1, int(sigma_x // 2)) if sigma_x > 4 else 1
    dy = max(1, int(sigma_y // 2)) if sigma_y > 4 else 1
    dx = min(dx, max(1, W // 4))
    dy = min(dy, max(1, H // 4))
    y = x
    if dx > 1 or dy > 1:
        y = F.avg_pool2d(y, kernel_size=(dy, dx), stride=(dy, dx), ceil_mode=True)
    if sigma_x > 0.05:
        y = _conv_axis(y, gaussian_kernel1d(sigma_x / dx, x.device, x.dtype), "x")
    if sigma_y > 0.05:
        y = _conv_axis(y, gaussian_kernel1d(sigma_y / dy, x.device, x.dtype), "y")
    if dx > 1 or dy > 1:
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
    return y


def resize_nhwc(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize an IMAGE-like [B,H,W,C] tensor (antialiased when shrinking)."""
    if img.shape[1] == height and img.shape[2] == width:
        return img
    x = img.permute(0, 3, 1, 2)
    shrink = height < img.shape[1] or width < img.shape[2]
    try:
        y = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False, antialias=shrink)
    except TypeError:  # very old torch
        y = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
    return y.permute(0, 2, 3, 1).contiguous()
