"""NovaMatchColor: colour continuity between shots.

Moves a clip's colour statistics (mean and spread of L*, a*, b* in CIELAB) onto a reference
picture's. The clip's statistics are measured over the whole clip, so every frame gets the same
correction and nothing flickers. ``colour_only`` keeps the clip's own brightness and contrast (L*)
and matches only the colour. Spreads are scaled by at most 2x either way, so a flat or odd reference
can't blow a clip apart.
"""

from __future__ import annotations

from typing import Tuple

import torch

_M = torch.tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
_WHITE = torch.tensor([0.95047, 1.0, 1.08883])
_D = 6.0 / 29.0


def srgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """[..., 3] sRGB in 0..1 -> CIELAB (D65): L* 0..100, a* and b* roughly -128..127."""
    x = x.clamp(0.0, 1.0)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.to(x).T / _WHITE.to(x)
    f = torch.where(xyz > _D**3, xyz.clamp_min(1e-12) ** (1.0 / 3.0), xyz / (3 * _D * _D) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return torch.stack([L, a, b], dim=-1)


def lab_to_srgb(lab: torch.Tensor) -> torch.Tensor:
    """CIELAB (D65) -> sRGB, clamped to 0..1."""
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = torch.stack([fx, fy, fz], dim=-1)
    xyz = torch.where(f > _D, f**3, 3 * _D * _D * (f - 4.0 / 29.0)) * _WHITE.to(lab)
    lin = xyz @ torch.linalg.inv(_M).to(lab).T
    lin = lin.clamp(0.0, 1.0)
    srgb = torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin.clamp_min(1e-12) ** (1.0 / 2.4) - 0.055)
    return srgb.clamp(0.0, 1.0)


def stats(images: torch.Tensor, max_samples: int = 400_000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of L*, a*, b* over a batch [N,H,W,3], from an even sample of pixels."""
    px = images[..., :3].reshape(-1, 3)
    step = max(1, px.shape[0] // max_samples)
    lab = srgb_to_lab(px[::step].float())
    return lab.mean(0), lab.std(0).clamp_min(1e-3)


def match(images: torch.Tensor, reference: torch.Tensor, amount: float = 1.0, colour_only: bool = False, chunk: int = 16, progress=None) -> torch.Tensor:
    """images [B,H,W,3] with the colour of ``reference`` [R,H',W',3], blended in by ``amount`` (0..1)."""
    if amount <= 0:
        return images
    mu_s, sd_s = stats(images)
    mu_r, sd_r = stats(reference)
    scale = (sd_r / sd_s).clamp(0.5, 2.0)
    if colour_only:
        scale[0] = 1.0
        mu_r = mu_r.clone()
        mu_r[0] = mu_s[0]
    B = images.shape[0]
    out = torch.empty((B, *images.shape[1:3], 3), dtype=torch.float32)
    for s0 in range(0, B, chunk):
        s1 = min(B, s0 + chunk)
        x = images[s0:s1, ..., :3].float()
        lab = srgb_to_lab(x)
        moved = lab_to_srgb((lab - mu_s) * scale + mu_r)
        out[s0:s1] = x + (moved - x) * float(amount)
        if progress:
            progress(s1, B)
    return out.clamp(0.0, 1.0)
