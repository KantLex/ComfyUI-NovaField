"""NovaGhostRig renderer: a camera-motion carrier video from a camera path.

Scene geometry follows docs/contract.md section 2. Lines are drawn by sampling points
densely in screen space along each (near- and screen-clipped) 3D segment, with
perspective-correct depth, and splatting them bilinearly (anti-aliased). Coverage and
colour are accumulated per pixel and composited over the background:

    alpha = 1 - exp(-COVERAGE_GAIN * coverage),  out = bg * (1 - alpha) + mean_colour * alpha

Depth fade (shared with the app's 3D preview): ``fade(z) = 0.12 + 0.88 * exp(-z / 16)``
with z = camera-space depth in metres; it scales each sample's coverage.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import camera
from .util import hex_rgb, resize_nhwc

STYLES = ["stage", "subject", "starfield", "parallax"]

BG = hex_rgb("#0B0C10")
GRID = hex_rgb("#3A4152")
AXIS = hex_rgb("#6B7690")
PILLAR = hex_rgb("#8A93A6")
PROP = hex_rgb("#C8C0AA")
SUBJECT = hex_rgb("#FFB46B")
ACTOR_COLORS = [hex_rgb("#FFB46B"), hex_rgb("#7FC8F8"), hex_rgb("#B7E37D")]
STAR = (0.86, 0.90, 1.0)

PILLAR_CENTERS = [(-4, -6), (4, -6), (-7, 0), (7, 0), (-4, 6), (4, 6)]
PILLAR_SIZE = (0.8, 3.0, 0.8)
SUBJECT_RADIUS = 0.32
SUBJECT_HEIGHT = 1.75

NEAR = camera.NEAR
SAMPLE_SPACING_PX = 0.5
COVERAGE_GAIN = 1.7
FADE_MIN, FADE_RANGE = 0.12, 16.0


def depth_fade(z: torch.Tensor) -> torch.Tensor:
    return FADE_MIN + (1.0 - FADE_MIN) * torch.exp(-z.clamp_min(0.0) / FADE_RANGE)


# =========================================================================== geometry
def _grid_segments() -> Tuple[List, List, List]:
    A, B, C = [], [], []
    for v in range(-12, 13):
        col = AXIS if v == 0 else GRID
        A.append((v, 0.0, -12.0)); B.append((v, 0.0, 12.0)); C.append(col)  # line x = v
        A.append((-12.0, 0.0, v)); B.append((12.0, 0.0, v)); C.append(col)  # line z = v
    return A, B, C


def _pillar_segments() -> Tuple[List, List, List]:
    A, B, C = [], [], []
    hx, hy, hz = PILLAR_SIZE[0] / 2, PILLAR_SIZE[1], PILLAR_SIZE[2] / 2
    for cx, cz in PILLAR_CENTERS:
        corners = [(cx + sx * hx, y, cz + sz * hz) for y in (0.0, hy) for sx in (-1, 1) for sz in (-1, 1)]
        # index = yi*4 + sxi*2 + szi
        edges = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            A.append(corners[a]); B.append(corners[b]); C.append(PILLAR)
    return A, B, C


BOX_EDGES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _prop_segments(props) -> Tuple[List, List, List]:
    A, B, C = [], [], []
    for pr in props or []:
        corners = camera.prop_corners(pr)
        for a, b in BOX_EDGES:
            A.append(tuple(corners[a])); B.append(tuple(corners[b])); C.append(PROP)
    return A, B, C


def subject_profile() -> List[Tuple[float, float]]:
    """(radius, y) profile of the capsule from the bottom pole to the top pole."""
    R, top = SUBJECT_RADIUS, SUBJECT_HEIGHT
    prof = []
    for a in np.linspace(-math.pi / 2, 0.0, 9):
        prof.append((R * math.cos(a), R + R * math.sin(a)))
    for a in np.linspace(0.0, math.pi / 2, 9):
        prof.append((R * math.cos(a), (top - R) + R * math.sin(a)))
    return prof


def subject_ring_radius(y: float) -> float:
    R, top = SUBJECT_RADIUS, SUBJECT_HEIGHT
    if y < R:
        return math.sqrt(max(R * R - (R - y) ** 2, 0.0))
    if y > top - R:
        return math.sqrt(max(R * R - (y - (top - R)) ** 2, 0.0))
    return R


def _subject_segments() -> Tuple[List, List, List]:
    A, B, C = [], [], []
    ring_n = 64
    k = 1
    while 0.15 * k < SUBJECT_HEIGHT - 1e-9:  # rings at y = 0.15, 0.30, ... 1.65
        y = 0.15 * k
        rr = subject_ring_radius(y)
        pts = [(rr * math.sin(2 * math.pi * i / ring_n), y, rr * math.cos(2 * math.pi * i / ring_n)) for i in range(ring_n)]
        for i in range(ring_n):
            A.append(pts[i]); B.append(pts[(i + 1) % ring_n]); C.append(SUBJECT)
        k += 1
    prof = subject_profile()
    for m in range(8):
        ph = 2 * math.pi * m / 8
        pts = [(r * math.sin(ph), y, r * math.cos(ph)) for r, y in prof]
        for i in range(len(pts) - 1):
            A.append(pts[i]); B.append(pts[i + 1]); C.append(SUBJECT)
    return A, B, C


def scene_segments(style: str, origin_subject: bool = True, props=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static segments of a style. ``origin_subject=False`` leaves out the origin capsule
    (used when the path has an ``actors`` list, whose capsules move per frame). ``props`` are the
    path's set pieces (contract 1c), drawn in the stage and subject styles."""
    A, B, C = [], [], []
    if style in ("stage", "subject"):
        for a, b, c in (_grid_segments(), _pillar_segments(), _prop_segments(props)):
            A += a; B += b; C += c
    if style == "subject" and origin_subject:
        a, b, c = _subject_segments()
        A += a; B += b; C += c
    if not A:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(A, dtype=np.float64), np.array(B, dtype=np.float64), np.array(C, dtype=np.float64)


def starfield_points(n: int = 2400, seed: int = 1234) -> np.ndarray:
    """Deterministic LCG starfield (contract section 2).

    x = r sin(phi) cos(theta), y = r cos(phi), z = r sin(phi) sin(theta).
    """
    state = seed & 0xFFFFFFFF
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        draws = []
        for _ in range(3):
            state = (state * 1664525 + 1013904223) % 4294967296
            draws.append(state / 4294967296.0)
        a, b, c = draws
        theta = 2 * math.pi * a
        phi = math.acos(2 * b - 1)
        r = 8 + 22 * c
        out[i] = (r * math.sin(phi) * math.cos(theta), r * math.cos(phi), r * math.sin(phi) * math.sin(theta))
    return out


# =========================================================================== splatting
def _splat_bilinear(buf: torch.Tensor, fi: torch.Tensor, x: torch.Tensor, y: torch.Tensor, vals: torch.Tensor, H: int, W: int):
    """Accumulate ``vals`` (M,C) at pixel coords (x, y) (pixel centres at +0.5) into flat buf (F*H*W, C)."""
    gx = x - 0.5
    gy = y - 0.5
    x0 = torch.floor(gx)
    y0 = torch.floor(gy)
    fx = gx - x0
    fy = gy - y0
    x0 = x0.long()
    y0 = y0.long()
    base = fi * (H * W)
    for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        wx = fx if ox else (1.0 - fx)
        wy = fy if oy else (1.0 - fy)
        xi = x0 + ox
        yi = y0 + oy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if not bool(ok.any()):
            continue
        idx = (base + yi * W + xi)[ok]
        buf.index_add_(0, idx, vals[ok] * (wx * wy)[ok].unsqueeze(1))


def _to_camera(Pw: torch.Tensor, R: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Pw (N,3) static or (F,N,3) per-frame world points -> (F,N,3) camera coords (xc, yc, zc)."""
    d = (Pw.unsqueeze(0) if Pw.dim() == 2 else Pw) - pos.unsqueeze(1)  # F,N,3
    return torch.einsum("fij,fnj->fni", R, d)


def _render_lines(buf, A, B, col, R, pos, focal, H, W, strength):
    """A, B (N,3) world segments, or (F,N,3) per frame; col (N,3); R (F,3,3); pos (F,3); focal (F,)."""
    Fn = R.shape[0]
    N = A.shape[-2]
    if N == 0:
        return
    dev = A.device
    ca = _to_camera(A, R, pos)  # F,N,3
    cb = _to_camera(B, R, pos)
    za, zb = ca[..., 2], cb[..., 2]
    near = NEAR + 1e-4
    keep = (za > near) | (zb > near)
    # near-plane clipping in camera space
    dz = zb - za
    safe_dz = torch.where(dz.abs() < 1e-12, torch.full_like(dz, 1e-12), dz)
    t_cut = ((near - za) / safe_dz).clamp(0.0, 1.0).unsqueeze(-1)
    ca2 = torch.where((za < near).unsqueeze(-1), ca + (cb - ca) * t_cut, ca)
    cb2 = torch.where((zb < near).unsqueeze(-1), ca + (cb - ca) * t_cut, cb)
    za2 = ca2[..., 2].clamp_min(near)
    zb2 = cb2[..., 2].clamp_min(near)
    fo = focal.view(Fn, 1)
    xa = W / 2 + fo * ca2[..., 0] / za2
    ya = H / 2 - fo * ca2[..., 1] / za2
    xb = W / 2 + fo * cb2[..., 0] / zb2
    yb = H / 2 - fo * cb2[..., 1] / zb2

    # Liang-Barsky clip against the screen (+2 px margin)
    dx = xb - xa
    dy = yb - ya
    t0 = torch.zeros_like(xa)
    t1 = torch.ones_like(xa)
    m = 2.0
    for p, q in ((-dx, xa + m), (dx, (W + m) - xa), (-dy, ya + m), (dy, (H + m) - ya)):
        par = p.abs() < 1e-12
        keep = keep & ~(par & (q < 0))
        r = q / torch.where(par, torch.ones_like(p), p)
        t0 = torch.where((p < 0) & ~par, torch.maximum(t0, r), t0)
        t1 = torch.where((p > 0) & ~par, torch.minimum(t1, r), t1)
    keep = keep & (t1 > t0)
    if not bool(keep.any()):
        return

    fidx, sidx = torch.nonzero(keep, as_tuple=True)
    xa, ya, dx, dy = xa[keep], ya[keep], dx[keep], dy[keep]
    t0, t1 = t0[keep], t1[keep]
    iza, izb = 1.0 / za2[keep], 1.0 / zb2[keep]
    seg_len = torch.sqrt(dx * dx + dy * dy) * (t1 - t0)
    n = torch.clamp(torch.ceil(seg_len / SAMPLE_SPACING_PX), min=1).long()
    total = int(n.sum())
    if total == 0:
        return
    rep = torch.repeat_interleave(torch.arange(n.numel(), device=dev), n)
    starts = torch.cumsum(n, 0) - n
    k = torch.arange(total, device=dev) - starts[rep]
    nn_ = n[rep].to(A.dtype)
    s = t0[rep] + (t1[rep] - t0[rep]) * (k.to(A.dtype) + 0.5) / nn_
    x = xa[rep] + s * dx[rep]
    y = ya[rep] + s * dy[rep]
    z = 1.0 / ((1.0 - s) * iza[rep] + s * izb[rep])  # perspective-correct depth
    w = (seg_len[rep] / nn_) * depth_fade(z) * strength
    c = col[sidx[rep]]
    vals = torch.cat([c * w.unsqueeze(1), w.unsqueeze(1)], dim=1)
    _splat_bilinear(buf, fidx[rep], x, y, vals, H, W)


def _render_points(buf, P, col, bright, R, pos, focal, H, W, strength, size_m=0.03, gain=2.6):
    """Gaussian dots for points P (N,3) with per-point colour (N,3) and brightness (N,)."""
    Fn = R.shape[0]
    cp = _to_camera(P, R, pos)
    z = cp[..., 2]
    keep = z > NEAR
    if not bool(keep.any()):
        return
    fidx, pidx = torch.nonzero(keep, as_tuple=True)
    zc = z[keep]
    fo = focal[fidx]
    x = W / 2 + fo * cp[..., 0][keep] / zc
    y = H / 2 - fo * cp[..., 1][keep] / zc
    on = (x > -6) & (x < W + 6) & (y > -6) & (y < H + 6)
    fidx, pidx, zc, fo, x, y = fidx[on], pidx[on], zc[on], fo[on], x[on], y[on]
    if x.numel() == 0:
        return
    sigma = (fo * size_m / zc).clamp(0.6, 2.0)
    amp = bright[pidx] * depth_fade(zc) * strength * gain
    gx = torch.floor(x - 0.5)
    gy = torch.floor(y - 0.5)
    base = fidx * (H * W)
    rad = 4
    for oy in range(-rad + 1, rad + 1):
        for ox in range(-rad + 1, rad + 1):
            xi = gx + ox
            yi = gy + oy
            ddx = (xi + 0.5) - x
            ddy = (yi + 0.5) - y
            g = torch.exp(-(ddx * ddx + ddy * ddy) / (2 * sigma * sigma)) * amp
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (g > 1e-4)
            if not bool(ok.any()):
                continue
            idx = (base + yi.long() * W + xi.long())[ok]
            gg = g[ok].unsqueeze(1)
            buf.index_add_(0, idx, torch.cat([col[pidx[ok]] * gg, gg], dim=1))


def _composite(buf: torch.Tensor, Fn: int, H: int, W: int) -> torch.Tensor:
    buf = buf.view(Fn, H, W, 4)
    A = buf[..., 3:4]
    colr = buf[..., :3] / A.clamp_min(1e-8)
    alpha = 1.0 - torch.exp(-COVERAGE_GAIN * A)
    bg = torch.tensor(BG, dtype=buf.dtype, device=buf.device).view(1, 1, 1, 3)
    return (bg * (1.0 - alpha) + colr * alpha).clamp(0.0, 1.0)


# =========================================================================== parallax
def _fill_holes(img: torch.Tensor, valid: torch.Tensor, depth: torch.Tensor, dilate_iters: int = 32) -> torch.Tensor:
    """Fill z-buffer holes. img (F,3,H,W), valid (F,1,H,W) bool, depth (F,1,H,W).

    1. Iterative 3x3 dilation that copies the *farthest* valid neighbour, so
       disocclusions are filled with stretched background rather than foreground.
    2. Push-pull (pyramid of valid-weighted averages) for whatever is left.
    """
    Fn, C, H, W = img.shape
    c = img.clone()
    v = valid.clone()
    neg = torch.finfo(depth.dtype).min
    d = torch.where(v, depth, torch.full_like(depth, neg))
    for _ in range(dilate_iters):
        if bool(v.all()):
            return c
        dm, idx = F.max_pool2d(d, 3, stride=1, padding=1, return_indices=True)
        new = (~v) & (dm > neg)
        if not bool(new.any()):
            break
        src = torch.gather(c.view(Fn, C, H * W), 2, idx.view(Fn, 1, H * W).expand(Fn, C, H * W)).view_as(c)
        c = torch.where(new, src, c)
        d = torch.where(new, dm, d)
        v = v | new
    if bool(v.all()):
        return c
    vf = v.to(img.dtype)
    pyr = [(c * vf, vf)]
    cc, vv = c * vf, vf
    while min(cc.shape[2], cc.shape[3]) > 1 and not bool((vv > 0).all()):
        sc = F.avg_pool2d(cc * vv, 2, ceil_mode=True)
        sv = F.avg_pool2d(vv, 2, ceil_mode=True)
        cc = sc / sv.clamp_min(1e-8)
        vv = (sv > 0).to(img.dtype)
        pyr.append((cc, vv))
    up_c = pyr[-1][0]
    for lvl in range(len(pyr) - 2, -1, -1):
        cl, vl = pyr[lvl]
        up = F.interpolate(up_c, size=cl.shape[2:], mode="bilinear", align_corners=False)
        up_c = cl * vl + up * (1 - vl)
    return up_c


def _snap_depth_edges(d: torch.Tensor, thresh: float = 0.04) -> torch.Tensor:
    """Remove 'flying pixels': at depth discontinuities snap each value to the nearer of
    the local 3x3 min / max so no point floats between foreground and background."""
    x = d.view(1, 1, *d.shape)
    dmax = F.max_pool2d(x, 3, stride=1, padding=1)
    dmin = -F.max_pool2d(-x, 3, stride=1, padding=1)
    edge = (dmax - dmin) > thresh
    snapped = torch.where((dmax - x) < (x - dmin), dmax, dmin)
    return torch.where(edge, snapped, x).view_as(d)


def synthetic_photo(H: int, W: int) -> torch.Tensor:
    """Fallback 'photo' for parallax when no image is connected: a colour test pattern."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    r = 0.25 + 0.6 * xx
    g = 0.25 + 0.5 * (1 - yy)
    b = 0.45 + 0.4 * yy
    img = torch.stack([r, g, b], dim=-1)
    checker = (((xx * 16).floor() + (yy * 9).floor()) % 2).unsqueeze(-1)
    img = img * (0.75 + 0.25 * checker)
    return img.unsqueeze(0).float()


def render_parallax(path, poses, H, W, image, depth, device, ss: int = 2, progress=None) -> torch.Tensor:
    """Forward-splat the frame-0 'photo' lifted to 3D, z-buffered, at ss x supersampling.

    Each point is splatted as a k x k block of subpixels, k = ceil(its magnification
    relative to frame 0) (1..4), so surfaces that come closer stay closed instead of
    letting the background show through.
    """
    Fn = path["frames"]
    Hs, Ws = H * ss, W * ss
    if image is None:
        image = synthetic_photo(H, W)
    img = resize_nhwc(image[:1].float(), Hs, Ws)[0].to(device)  # Hs,Ws,3
    if depth is not None:
        d = resize_nhwc(depth[:1].float(), Hs, Ws)[0].to(device)
        d = d[..., :3].mean(dim=-1) if d.shape[-1] >= 3 else d[..., 0]
        d = _snap_depth_edges(d.clamp(0, 1))
    else:  # vertical ramp: bottom near (1), top far (0)
        d = ((torch.arange(Hs, device=device, dtype=torch.float32) + 0.5) / Hs).view(Hs, 1).expand(Hs, Ws)
    zdepth = 1.5 + (1.0 - d) * 10.5  # camera-space depth (planar z) at frame 0

    # lift every (supersampled) pixel to world space using the frame-0 camera
    R0 = torch.tensor(poses["R"][0], dtype=torch.float32, device=device)
    p0 = torch.tensor(poses["pos"][0], dtype=torch.float32, device=device)
    f0 = camera.focal_px(float(poses["fov"][0]), H)
    py, px = torch.meshgrid(
        (torch.arange(Hs, device=device, dtype=torch.float32) + 0.5) / ss,
        (torch.arange(Ws, device=device, dtype=torch.float32) + 0.5) / ss,
        indexing="ij",
    )
    xn = (px - W / 2) / f0
    yn = -(py - H / 2) / f0
    dirs = xn.unsqueeze(-1) * R0[0] + yn.unsqueeze(-1) * R0[1] + R0[2]
    P = (p0 + dirs * zdepth.unsqueeze(-1)).reshape(-1, 3)
    z0 = zdepth.reshape(-1)
    cols = img.reshape(-1, 3)
    Np = P.shape[0]
    bits = max(1, int(math.ceil(math.log2(Np + 1))))
    BIG = torch.iinfo(torch.int64).max
    pidx = torch.arange(Np, device=device, dtype=torch.int64)
    KMAX = 4

    out = torch.empty((Fn, H, W, 3), dtype=torch.float32)
    chunk = max(1, min(16, int(12_000_000 // Np)))
    for s0 in range(0, Fn, chunk):
        s1 = min(Fn, s0 + chunk)
        n = s1 - s0
        R = torch.tensor(poses["R"][s0:s1], dtype=torch.float32, device=device)
        pos = torch.tensor(poses["pos"][s0:s1], dtype=torch.float32, device=device)
        fk = [camera.focal_px(float(v), H) for v in poses["fov"][s0:s1]]
        zbuf = torch.full((n * Hs * Ws,), BIG, dtype=torch.int64, device=device)
        for k in range(n):
            cp = (P - pos[k]) @ R[k].T
            zc = cp[:, 2]
            ok = zc > NEAR
            zs = zc.clamp_min(NEAR)
            sx = Ws / 2 + fk[k] * ss * cp[:, 0] / zs
            sy = Hs / 2 - fk[k] * ss * cp[:, 1] / zs
            ok &= (sx > -KMAX) & (sx < Ws + KMAX) & (sy > -KMAX) & (sy < Hs + KMAX)
            sx, sy, zc = sx[ok], sy[ok], zc[ok]
            mag = (z0[ok] / zc) * (fk[k] / f0)
            kk = torch.ceil(mag - 0.05).clamp(1, KMAX).long()
            dq = (zc * 10000.0).long().clamp(0, (1 << (62 - bits)) - 1)
            key = (dq << bits) | pidx[ok]
            bx = torch.floor(sx - (kk - 1) * 0.5).long()
            by = torch.floor(sy - (kk - 1) * 0.5).long()
            for oy in range(KMAX):
                for ox in range(KMAX):
                    sel = kk > max(ox, oy)
                    if not bool(sel.any()):
                        continue
                    xi = bx[sel] + ox
                    yi = by[sel] + oy
                    inb = (xi >= 0) & (xi < Ws) & (yi >= 0) & (yi < Hs)
                    pix = k * Hs * Ws + yi[inb] * Ws + xi[inb]
                    zbuf.scatter_reduce_(0, pix, key[sel][inb], reduce="amin", include_self=True)
        valid = zbuf != BIG
        frame = torch.zeros((n * Hs * Ws, 3), dtype=torch.float32, device=device)
        zimg = torch.zeros((n * Hs * Ws,), dtype=torch.float32, device=device)
        win = zbuf[valid] & ((1 << bits) - 1)
        frame[valid] = cols[win]
        zimg[valid] = (zbuf[valid] >> bits).float() / 10000.0
        frame = frame.view(n, Hs, Ws, 3).permute(0, 3, 1, 2)
        vs = valid.view(n, 1, Hs, Ws)
        zimg = zimg.view(n, 1, Hs, Ws)
        # drop a thin rim around holes: source edge pixels are colour-blended (anti-aliased)
        # and would otherwise be smeared across the disocclusion by the fill
        rim = 2 if ss > 1 else 1
        vs = vs & ~(F.max_pool2d((~vs).float(), 2 * rim + 1, stride=1, padding=rim) > 0)
        vf = vs.float()
        if ss > 1:  # resolve the supersampling (anti-aliasing) before filling holes
            cov = F.avg_pool2d(vf, ss)
            frame = F.avg_pool2d(frame * vf, ss) / cov.clamp_min(1e-8)
            zimg = F.avg_pool2d(zimg * vf, ss) / cov.clamp_min(1e-8)
            vs = cov > 0
        frame = _fill_holes(frame, vs, zimg)
        out[s0:s1] = frame.permute(0, 2, 3, 1).clamp(0, 1).cpu()
        if progress:
            progress(s1, Fn)
    return out


# =========================================================================== main entry
def render(path_json, width: int, height: int, style: str = "subject", line_strength: float = 1.0,
           image: Optional[torch.Tensor] = None, depth: Optional[torch.Tensor] = None,
           device=None, progress=None) -> torch.Tensor:
    """Render the ghost rig. Returns float32 [F, H, W, 3] on CPU."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r} (expected one of {STYLES})")
    path = camera.parse_camera_path(path_json)
    poses = camera.all_poses(path)
    H, W = int(height), int(width)
    if device is None:
        device = torch.device("cpu")
    if style == "parallax":
        return render_parallax(path, poses, H, W, image, depth, device, progress=progress)

    Fn = path["frames"]
    dt = torch.float32
    use_actors = style == "subject" and path.get("actors") is not None
    A_np, B_np, C_np = scene_segments(style, origin_subject=not use_actors, props=path.get("props"))
    actor_geo = None
    if use_actors and path["actors"]:
        # unit capsule template (1.75 m), scaled per actor by height / 1.75 and moved per frame
        ta, tb, _ = _subject_segments()
        ta = torch.tensor(ta, dtype=dt, device=device)
        tb = torch.tensor(tb, dtype=dt, device=device)
        scales = torch.tensor([a["height"] / SUBJECT_HEIGHT for a in path["actors"]], dtype=dt, device=device)
        Na, Ns = len(path["actors"]), ta.shape[0]
        cols = torch.tensor([ACTOR_COLORS[i % len(ACTOR_COLORS)] for i in range(Na)], dtype=dt, device=device)
        cols = cols.repeat_interleave(Ns, dim=0)  # (Na*Ns, 3)
        apos = torch.tensor(np.array([np.stack(camera.actor_positions(path, i)) for i in range(Fn)]),
                            dtype=dt, device=device)  # (F, Na, 3)
        sa = ta.unsqueeze(0) * scales.view(Na, 1, 1)  # (Na, Ns, 3)
        sb = tb.unsqueeze(0) * scales.view(Na, 1, 1)
        actor_geo = (sa, sb, cols, apos)
    A = torch.tensor(A_np, dtype=dt, device=device)
    B = torch.tensor(B_np, dtype=dt, device=device)
    C = torch.tensor(C_np, dtype=dt, device=device)
    stars = None
    if style == "starfield":
        P = starfield_points()
        idx = np.arange(P.shape[0])
        bright = 0.55 + 0.45 * np.abs(np.sin(idx * 12.9898) * 43758.5453 % 1.0)
        stars = (
            torch.tensor(P, dtype=dt, device=device),
            torch.tensor(STAR, dtype=dt, device=device).view(1, 3).expand(P.shape[0], 3).contiguous(),
            torch.tensor(bright, dtype=dt, device=device),
        )

    out = torch.empty((Fn, H, W, 3), dtype=torch.float32)
    chunk = 8
    for s0 in range(0, Fn, chunk):
        s1 = min(Fn, s0 + chunk)
        n = s1 - s0
        R = torch.tensor(poses["R"][s0:s1], dtype=dt, device=device)
        pos = torch.tensor(poses["pos"][s0:s1], dtype=dt, device=device)
        foc = torch.tensor([camera.focal_px(float(v), H) for v in poses["fov"][s0:s1]], dtype=dt, device=device)
        buf = torch.zeros((n * H * W, 4), dtype=dt, device=device)
        _render_lines(buf, A, B, C, R, pos, foc, H, W, float(line_strength))
        if actor_geo is not None:
            sa, sb, cols, apos = actor_geo
            off = apos[s0:s1].unsqueeze(2)  # (n, Na, 1, 3)
            Ad = (sa.unsqueeze(0) + off).reshape(n, -1, 3)
            Bd = (sb.unsqueeze(0) + off).reshape(n, -1, 3)
            _render_lines(buf, Ad, Bd, cols, R, pos, foc, H, W, float(line_strength))
        if stars is not None:
            _render_points(buf, stars[0], stars[1], stars[2], R, pos, foc, H, W, float(line_strength))
        out[s0:s1] = _composite(buf, n, H, W).cpu()
        if progress:
            progress(s1, Fn)
    return out
