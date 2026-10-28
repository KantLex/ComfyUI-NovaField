"""NovaLensFilm: film emulation + lens character for a frame batch, run in chunks.

Pipeline per frame (each stage is skipped entirely when its control is 0):

1. look grade (exposure, channel mix, saturation, S tone curve, black/white levels,
   split toning, look-specific extras), blended with the input by ``intensity``;
   then the colour corrections (exposure, temperature/tint, saturation, contrast, fade)
2. geometric: gate weave (per-frame subpixel translation, smooth seeded random walk)
   + chromatic aberration (radial RGB scale), one bilinear resample per channel
3. halation: blurred highlights tinted red-orange, added back
4. anamorphic streak: very bright pixels blurred horizontally, tinted blue, added back
5. vignette: smooth elliptical radial falloff
6. grain: seeded, temporally varying, luminance-dependent (strongest in the midtones)

``look == "none"`` with every slider at 0 returns the input tensor unchanged.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .util import blur, gaussian_kernel1d, _conv_axis, luma, smoothstep

LOOKS = ["none", "kodak_2383", "fuji_eterna", "bleach_bypass", "teal_orange", "noir", "technicolor", "moonlight", "vhs_1989",
         "kodak_portra_400", "kodak_vision3_500t", "ektachrome_e100", "cinestill_800t", "fuji_velvia_50", "kodachrome_64",
         "ilford_hp5", "instant_film", "super8"]

_I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# exposure, mix (3x3, rows = output channel), sat, contrast/pivot (S-curve), black/white
# levels (rgb), split-tone offsets for shadows / highlights, extras.
LOOK_PARAMS: Dict[str, Dict] = {
    "kodak_2383": dict(
        exposure=1.0,
        mix=[[1.05, -0.03, -0.02], [-0.02, 1.02, 0.00], [-0.02, -0.05, 1.07]],
        sat=1.08, contrast=1.45, pivot=0.42,
        black=(0.010, 0.016, 0.024), white=(0.98, 0.95, 0.89),
        shadow=(-0.030, 0.006, 0.028), highlight=(0.035, 0.012, -0.030),
    ),
    "fuji_eterna": dict(
        exposure=1.0,
        mix=[[0.96, 0.06, -0.02], [0.00, 1.00, 0.00], [-0.02, 0.07, 0.95]],
        sat=0.74, contrast=1.10, pivot=0.45,
        black=(0.036, 0.046, 0.046), white=(0.94, 0.96, 0.94),
        shadow=(-0.018, 0.014, 0.010), highlight=(0.010, 0.010, -0.008),
    ),
    "bleach_bypass": dict(
        exposure=1.02,
        mix=_I,
        sat=0.5, contrast=1.4, pivot=0.5,
        black=(0.012, 0.014, 0.018), white=(0.96, 0.96, 0.97),
        shadow=(-0.006, 0.0, 0.010), highlight=(0.004, 0.004, 0.0),
        silver=0.22,
    ),
    "teal_orange": dict(
        exposure=1.0,
        mix=[[1.04, -0.02, -0.02], [-0.02, 1.02, 0.00], [-0.02, 0.02, 1.00]],
        sat=1.05, contrast=1.28, pivot=0.45,
        black=(0.0, 0.022, 0.034), white=(1.0, 0.96, 0.90),
        shadow=(-0.060, 0.022, 0.060), highlight=(0.050, 0.018, -0.045),
        teal_orange=1.0,
    ),
    "noir": dict(
        exposure=1.03,
        mono=(0.52, 0.40, 0.08),
        mix=_I, sat=0.0, contrast=2.0, pivot=0.42,
        black=(0.0, 0.0, 0.0), white=(1.0, 1.0, 1.0),
        shadow=(0.0, 0.0, 0.0), highlight=(0.0, 0.0, 0.0),
    ),
    "technicolor": dict(
        exposure=1.0,
        mix=[[1.28, -0.16, -0.12], [-0.12, 1.24, -0.12], [-0.05, -0.20, 1.25]],
        sat=1.22, contrast=1.30, pivot=0.45,
        black=(0.016, 0.010, 0.022), white=(1.0, 0.97, 0.90),
        shadow=(0.0, -0.010, 0.020), highlight=(0.030, 0.020, -0.020),
    ),
    "moonlight": dict(
        exposure=0.62,
        mix=[[0.72, 0.18, 0.10], [0.10, 0.80, 0.10], [0.08, 0.22, 0.86]],
        sat=0.55, contrast=1.25, pivot=0.40,
        black=(0.008, 0.018, 0.045), white=(0.70, 0.82, 1.0),
        shadow=(-0.020, 0.0, 0.050), highlight=(-0.020, 0.010, 0.040),
    ),
    "vhs_1989": dict(
        exposure=1.02,
        mix=[[1.02, 0.0, 0.0], [0.0, 0.97, 0.03], [0.02, 0.03, 0.98]],
        sat=1.18, contrast=0.86, pivot=0.5,
        black=(0.060, 0.050, 0.075), white=(0.94, 0.93, 0.91),
        shadow=(0.012, -0.010, 0.030), highlight=(0.025, 0.004, -0.020),
        vhs=True,
    ),
    "kodak_portra_400": dict(
        exposure=1.02,
        mix=[[1.03, -0.01, -0.02], [-0.01, 1.01, 0.00], [-0.02, -0.02, 1.04]],
        sat=0.9, contrast=1.12, pivot=0.45,
        black=(0.030, 0.028, 0.030), white=(0.98, 0.96, 0.92),
        shadow=(-0.010, 0.008, 0.012), highlight=(0.030, 0.015, -0.020),
    ),
    "kodak_vision3_500t": dict(
        exposure=1.0,
        mix=[[1.02, -0.01, -0.01], [-0.01, 1.00, 0.01], [0.00, -0.02, 1.02]],
        sat=0.95, contrast=1.22, pivot=0.43,
        black=(0.012, 0.020, 0.030), white=(0.97, 0.95, 0.92),
        shadow=(-0.025, 0.004, 0.030), highlight=(0.030, 0.012, -0.025),
    ),
    "ektachrome_e100": dict(
        exposure=0.98,
        mix=[[1.02, -0.02, 0.00], [-0.02, 1.03, -0.01], [-0.01, -0.02, 1.08]],
        sat=1.2, contrast=1.4, pivot=0.48,
        black=(0.005, 0.008, 0.014), white=(0.99, 0.99, 1.0),
        shadow=(-0.010, 0.0, 0.020), highlight=(0.0, 0.006, 0.012),
    ),
    "cinestill_800t": dict(
        exposure=1.0,
        mix=[[0.98, 0.02, 0.00], [0.00, 1.00, 0.00], [0.00, 0.04, 1.04]],
        sat=1.05, contrast=1.25, pivot=0.42,
        black=(0.0, 0.020, 0.035), white=(1.0, 0.95, 0.90),
        shadow=(-0.050, 0.012, 0.050), highlight=(0.045, 0.010, -0.030),
    ),
    "fuji_velvia_50": dict(
        exposure=0.97,
        mix=[[1.12, -0.08, -0.04], [-0.06, 1.14, -0.08], [-0.03, -0.08, 1.11]],
        sat=1.4, contrast=1.5, pivot=0.47,
        black=(0.004, 0.006, 0.010), white=(1.0, 0.99, 0.97),
        shadow=(0.0, 0.006, 0.012), highlight=(0.010, 0.010, -0.010),
    ),
    "kodachrome_64": dict(
        exposure=1.0,
        mix=[[1.10, -0.06, -0.04], [-0.04, 1.04, 0.00], [-0.02, -0.06, 1.02]],
        sat=1.15, contrast=1.38, pivot=0.44,
        black=(0.014, 0.010, 0.012), white=(0.99, 0.96, 0.86),
        shadow=(0.0, -0.004, 0.016), highlight=(0.030, 0.020, -0.040),
    ),
    "ilford_hp5": dict(
        exposure=1.0,
        mono=(0.30, 0.59, 0.11),
        mix=_I, sat=0.0, contrast=1.35, pivot=0.45,
        black=(0.025, 0.025, 0.025), white=(0.97, 0.97, 0.97),
        shadow=(0.0, 0.0, 0.0), highlight=(0.0, 0.0, 0.0),
    ),
    "instant_film": dict(
        exposure=1.04,
        mix=[[0.98, 0.04, -0.02], [0.02, 1.00, -0.02], [0.00, 0.06, 0.90]],
        sat=0.82, contrast=0.9, pivot=0.5,
        black=(0.090, 0.100, 0.080), white=(0.95, 0.94, 0.86),
        shadow=(-0.010, 0.020, 0.0), highlight=(0.030, 0.020, -0.030),
    ),
    "super8": dict(
        exposure=1.03,
        mix=[[1.04, 0.00, -0.04], [0.00, 1.00, -0.02], [-0.02, 0.02, 0.88]],
        sat=0.92, contrast=1.15, pivot=0.48,
        black=(0.060, 0.045, 0.030), white=(0.97, 0.92, 0.80),
        shadow=(0.010, 0.0, -0.010), highlight=(0.040, 0.020, -0.040),
    ),
}


# =========================================================================== grade
def _s_curve(x: torch.Tensor, c: float, p: float) -> torch.Tensor:
    """Power S-curve through (0,0), (p,p), (1,1); c > 1 adds contrast, c < 1 flattens."""
    x = x.clamp(0.0, 1.0)
    lo = p * (x / p).clamp_min(0) ** c
    hi = 1.0 - (1.0 - p) * ((1.0 - x) / (1.0 - p)).clamp_min(0) ** c
    return torch.where(x < p, lo, hi)


def _vec(v, ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(v, dtype=ref.dtype, device=ref.device).view(1, 1, 1, 3)


def _vhs(x: torch.Tensor) -> torch.Tensor:
    """Chroma bleed + scanline softness. x [N,H,W,3] -> same."""
    N, H, W, _ = x.shape
    sx = max(W / 640.0, 0.5)
    sy = max(H / 480.0, 0.5)
    xc = x.permute(0, 3, 1, 2)
    r, g, b = xc[:, 0:1], xc[:, 1:2], xc[:, 2:3]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = b - Y
    cr = r - Y
    # chroma bleed: wide horizontal smear shifted to the right
    k = gaussian_kernel1d(2.6 * sx, x.device, x.dtype, shift=1.8 * sx)
    chroma = _conv_axis(torch.cat([cb, cr], dim=1), k, "x")
    chroma = blur(chroma, 0.0, 0.6 * sy)
    cb, cr = chroma[:, 0:1], chroma[:, 1:2]
    # luma: slight horizontal softness, vertical scanline softness + faint line structure
    Y = blur(Y, 0.75 * sx, 0.55 * sy)
    rows = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
    lines = 0.5 + 0.5 * torch.cos(2 * math.pi * (rows + 0.5) * (240.0 / H))
    Y = Y * (1.0 - 0.07 * lines)
    r2 = Y + cr
    b2 = Y + cb
    g2 = (Y - 0.299 * r2 - 0.114 * b2) / 0.587
    return torch.cat([r2, g2, b2], dim=1).permute(0, 2, 3, 1)


def grade(x: torch.Tensor, look: str) -> torch.Tensor:
    """Apply a look at full strength. x [N,H,W,3] float in 0..1."""
    p = LOOK_PARAMS[look]
    if "mono" in p:
        m = (x * _vec(p["mono"], x)).sum(-1, keepdim=True)
        y = m.expand_as(x)
    else:
        M = torch.tensor(p["mix"], dtype=x.dtype, device=x.device)
        y = torch.einsum("nhwc,oc->nhwo", x, M)
    y = y * p["exposure"]
    if p["sat"] != 1.0:
        L = luma(y)
        y = L + (y - L) * p["sat"]
    if p.get("teal_orange"):
        # warm hues (skin, fire) are pushed to orange; everything else collapses towards teal
        k = p["teal_orange"]
        L = luma(y)
        chroma = (y.amax(-1, keepdim=True) - y.amin(-1, keepdim=True)).clamp(0, 1)
        warm = smoothstep(-0.10, 0.25, y[..., 0:1] - y[..., 2:3])
        cool_mix = torch.tensor([[0.80, 0.10, 0.10], [0.0, 0.70, 0.30], [0.0, 0.42, 0.58]], dtype=y.dtype, device=y.device)
        teal = torch.einsum("nhwc,oc->nhwo", y, cool_mix)
        y = y + (teal - y) * (1 - warm) * k
        push = warm * chroma * _vec((0.09, 0.02, -0.08), y) + (1 - warm) * (0.3 + chroma) * _vec((-0.05, 0.012, 0.035), y)
        y = y + k * push * (0.3 + 0.7 * L)
    y = _s_curve(y, p["contrast"], p["pivot"])
    if p.get("silver"):  # bleach bypass silver retention: overlay the luma
        L = luma(y)
        ov = torch.where(L < 0.5, 2 * y * L, 1 - 2 * (1 - y) * (1 - L))
        y = y + (ov - y) * p["silver"]
    blk, wht = _vec(p["black"], y), _vec(p["white"], y)
    y = blk + y * (wht - blk)
    L = luma(y).clamp(0, 1)
    y = y + (1 - L) ** 2 * _vec(p["shadow"], y) + L ** 2 * _vec(p["highlight"], y)
    if p.get("vhs"):
        y = _vhs(y.clamp(0, 1))
    return y.clamp(0.0, 1.0)


# =========================================================================== colour
def adjust(x: torch.Tensor, exposure: float = 0.0, contrast: float = 0.0, saturation: float = 0.0,
           temperature: float = 0.0, tint: float = 0.0, fade: float = 0.0) -> torch.Tensor:
    """Colour corrections after the look. Each control is neutral at 0 (and skipped).

    exposure in stops; temperature warms (+) or cools (-); tint goes magenta (+) or green (-);
    contrast is an S-curve around mid grey; fade lifts the blacks and dims the whites.
    """
    if exposure:
        x = x * (2.0 ** exposure)
    if temperature or tint:
        gain = (1 + 0.12 * temperature + 0.04 * tint, 1 - 0.12 * tint, 1 - 0.12 * temperature + 0.04 * tint)
        x = x * _vec(gain, x)
    if saturation:
        L = luma(x)
        x = L + (x - L) * (1 + saturation)
    if contrast:
        x = _s_curve(x, 2.0 ** contrast, 0.5)
    if fade:
        x = 0.1 * fade + x.clamp(0.0, 1.0) * (1 - 0.25 * fade)
    return x.clamp(0.0, 1.0)


# =========================================================================== gate weave
def weave_offsets(n_frames: int, seed: int) -> np.ndarray:
    """(F,2) smooth random walk (OU process + 5-tap smoothing), roughly unit std."""
    rs = np.random.RandomState((int(seed) * 2654435761 + 97) % (2 ** 32))
    a = 0.8
    v = np.zeros((n_frames + 4, 2))
    v[0] = rs.randn(2)
    for i in range(1, n_frames + 4):
        v[i] = a * v[i - 1] + math.sqrt(1 - a * a) * rs.randn(2)
    k = np.array([1, 2, 3, 2, 1], dtype=np.float64) / 9.0
    out = np.stack([np.convolve(v[:, c], k, mode="valid") for c in range(2)], axis=1)
    return out[:n_frames] * 1.35


def _warp(x: torch.Tensor, chromatic: float, shifts_px: Optional[torch.Tensor]) -> torch.Tensor:
    """x [N,H,W,3]. Radial RGB scale (R out, B in) + per-frame translation (N,2) in px."""
    N, H, W, _ = x.shape
    xc = x.permute(0, 3, 1, 2)
    ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H * 2 - 1
    xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # 1,H,W,2
    off = torch.zeros((N, 1, 1, 2), dtype=x.dtype, device=x.device)
    if shifts_px is not None:
        # sampling at (p - shift) moves the image content by +shift
        off = -torch.stack([shifts_px[:, 0] * 2 / W, shifts_px[:, 1] * 2 / H], dim=-1).view(N, 1, 1, 2)
    k = 0.006 * chromatic
    chans = []
    for c, sc in enumerate((1.0 - k, 1.0, 1.0 + k)):
        grid = base * sc + off
        chans.append(F.grid_sample(xc[:, c : c + 1], grid, mode="bilinear", padding_mode="reflection", align_corners=False))
    return torch.cat(chans, dim=1).permute(0, 2, 3, 1)


# =========================================================================== lens effects
def _halation(x: torch.Tensor, amount: float) -> torch.Tensor:
    N, H, W, _ = x.shape
    L = luma(x)
    hi = (smoothstep(0.75, 1.0, L) * L).permute(0, 3, 1, 2)
    s = 0.012 * H
    b = 0.65 * blur(hi, s, s) + 0.35 * blur(hi, 3 * s, 3 * s)
    b = b.permute(0, 2, 3, 1)
    return x + amount * 1.1 * b * _vec((1.0, 0.36, 0.12), x)


def _streak(x: torch.Tensor, amount: float) -> torch.Tensor:
    N, H, W, _ = x.shape
    L = luma(x)
    hi = (smoothstep(0.85, 1.0, L) * L).permute(0, 3, 1, 2)
    sy = max(0.6, 0.0015 * H)
    b = 0.55 * blur(hi, 0.02 * W, sy) + 0.45 * blur(hi, 0.09 * W, sy)
    b = b.permute(0, 2, 3, 1)
    return x + amount * 4.0 * b * _vec((0.30, 0.58, 1.0), x)


def _vignette(x: torch.Tensor, amount: float) -> torch.Tensor:
    N, H, W, _ = x.shape
    ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H * 2 - 1
    xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) / W * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(gx * gx + gy * gy) / math.sqrt(2.0)  # 0 centre, 1 corners (elliptical)
    v = 1.0 - amount * 0.8 * smoothstep(0.25, 1.0, r)
    return x * v.view(1, H, W, 1)


def _grain(x: torch.Tensor, amount: float, seed: int, frame0: int, size: float = 1.0) -> torch.Tensor:
    N, H, W, _ = x.shape
    dev = x.device
    sig = 0.55 * max(1.0, H / 720.0) * size
    k = gaussian_kernel1d(sig, torch.device("cpu"), torch.float64)
    norm = float((k * k).sum())  # std of separably blurred unit white noise
    noises = []
    for i in range(N):
        g = torch.Generator(device=dev)
        g.manual_seed((int(seed) * 1000003 + (frame0 + i) * 7919 + 12345) % (2 ** 63))
        noises.append(torch.randn((1, 4, H, W), generator=g, device=dev, dtype=x.dtype))
    n = torch.cat(noises, 0)
    n = blur(n, sig, sig) / norm
    mono = n[:, 0:1]
    col = n[:, 1:4] * 0.3
    noise = (mono + col).permute(0, 2, 3, 1)
    L = luma(x).clamp(0, 1)
    amp = amount * 0.065 * (0.3 + 0.7 * 4.0 * L * (1.0 - L))
    return x + noise * amp


# =========================================================================== entry
def is_identity(look: str, intensity: float, grain: float, halation: float, vignette: float,
                chromatic: float, gate_weave: float, anamorphic_streak: float, exposure: float = 0.0,
                contrast: float = 0.0, saturation: float = 0.0, temperature: float = 0.0, tint: float = 0.0,
                fade: float = 0.0) -> bool:
    graded = look != "none" and intensity > 0
    corrected = any((exposure, contrast, saturation, temperature, tint, fade))
    return not graded and not corrected and max(grain, halation, vignette, chromatic, gate_weave, anamorphic_streak) <= 0


def apply(images: torch.Tensor, look: str = "none", intensity: float = 0.8, grain: float = 0.0,
          halation: float = 0.0, vignette: float = 0.0, chromatic: float = 0.0, gate_weave: float = 0.0,
          anamorphic_streak: float = 0.0, seed: int = 0, device=None, chunk: int = 16,
          progress=None, *, exposure: float = 0.0, contrast: float = 0.0, saturation: float = 0.0,
          temperature: float = 0.0, tint: float = 0.0, fade: float = 0.0, grain_size: float = 1.0) -> torch.Tensor:
    """images [B,H,W,3] float 0..1 -> same shape, float32 on CPU."""
    if look not in LOOKS:
        raise ValueError(f"unknown look {look!r} (expected one of {LOOKS})")
    corrections = dict(exposure=float(exposure), contrast=float(contrast), saturation=float(saturation),
                       temperature=float(temperature), tint=float(tint), fade=float(fade))
    if is_identity(look, intensity, grain, halation, vignette, chromatic, gate_weave, anamorphic_streak, **corrections):
        return images
    if device is None:
        device = torch.device("cpu")
    B, H, W, C = images.shape
    shifts_all = None
    if gate_weave > 0:
        w = weave_offsets(B, seed)
        amp = np.array([0.0018 * W, 0.0026 * H]) * gate_weave
        shifts_all = torch.tensor(w * amp, dtype=torch.float32)
    out = torch.empty((B, H, W, 3), dtype=torch.float32)
    for s0 in range(0, B, chunk):
        s1 = min(B, s0 + chunk)
        x = images[s0:s1, ..., :3].to(device=device, dtype=torch.float32)
        if look != "none" and intensity > 0:
            g = grade(x, look)
            x = x + (g - x) * float(intensity)
        if any(corrections.values()):
            x = adjust(x, **corrections)
        if chromatic > 0 or shifts_all is not None:
            sh = shifts_all[s0:s1].to(device) if shifts_all is not None else None
            x = _warp(x, float(chromatic), sh)
        if halation > 0:
            x = _halation(x, float(halation))
        if anamorphic_streak > 0:
            x = _streak(x, float(anamorphic_streak))
        if vignette > 0:
            x = _vignette(x, float(vignette))
        if grain > 0:
            x = _grain(x, float(grain), seed, s0, float(grain_size))
        out[s0:s1] = x.clamp(0.0, 1.0).cpu()
        if progress:
            progress(s1, B)
    return out
