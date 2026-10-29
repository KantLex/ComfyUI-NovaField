"""NovaShotQC: a quick health check for a generated clip, without any model.

Looks for the faults AI video tends to have and returns a JSON report the NovaField app shows on the
take: frozen or repeated frames, sudden jumps (a frame that lurches or morphs), brightness flicker,
soft frames (focus drop-outs), blown highlights and crushed shadows. Each issue has the frames it
spans; ``score`` is 100 for a clean take and drops with every fault found.
"""

from __future__ import annotations

import json
from typing import Dict, List

import torch
import torch.nn.functional as F

from .util import luma

LAP = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _small(images: torch.Tensor, width: int) -> torch.Tensor:
    """Luma of each frame, shrunk to ``width`` pixels wide: [N,1,h,w]."""
    y = luma(images[..., :3].float()).permute(0, 3, 1, 2)
    H, W = y.shape[-2:]
    if W <= width:
        return y
    h = max(8, round(H * width / W))
    return F.interpolate(y, size=(h, width), mode="area")


def _runs(flags: List[bool]) -> List[List[int]]:
    """[[start, end], ...] of consecutive True entries (inclusive)."""
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append([start, i - 1])
            start = None
    return out


def _median(x: torch.Tensor) -> float:
    return float(x.median()) if x.numel() else 0.0


def _curve(x: torch.Tensor, points: int = 120) -> List[float]:
    if x.numel() <= points:
        return [round(float(v), 5) for v in x]
    idx = torch.linspace(0, x.numel() - 1, points).round().long()
    return [round(float(v), 5) for v in x[idx]]


def analyse(images: torch.Tensor, fps: float = 24.0, chunk: int = 32) -> Dict:
    N = int(images.shape[0])
    small, sharp_imgs, means, clipped, crushed = [], [], [], [], []
    for s0 in range(0, N, chunk):
        x = images[s0 : s0 + chunk, ..., :3].float()
        small.append(_small(x, 160))
        sharp_imgs.append(_small(x, 480))
        y = luma(x[:, ::2, ::2])
        means.append(y.mean(dim=(1, 2, 3)))
        clipped.append((y > 0.985).float().mean(dim=(1, 2, 3)))
        crushed.append((y < 0.015).float().mean(dim=(1, 2, 3)))
    small_t = torch.cat(small)
    mean = torch.cat(means)
    clip = torch.cat(clipped)
    crush = torch.cat(crushed)
    sharp = torch.cat([F.conv2d(s, LAP.to(s).view(1, 1, 3, 3)).flatten(1).var(dim=1) for s in sharp_imgs])

    diff = (small_t[1:] - small_t[:-1]).abs().mean(dim=(1, 2, 3)) if N > 1 else torch.zeros(0)
    issues: List[Dict] = []
    score = 100.0

    if N > 2:
        med = _median(diff)
        # Frozen: the picture stops while the rest of the clip moves.
        if med > 0.004:
            frozen = [bool(d < 0.0012) for d in diff]
            for a, b in _runs(frozen):
                issues.append({"kind": "frozen", "start": a, "end": b + 1, "severity": "high" if b - a >= 2 else "medium"})
                score -= min(30, 4 * (b - a + 1))
        # Jumps: one step much bigger than the clip's usual motion (a lurch, a morph, a hidden cut).
        jump_at = max(0.06, 5.0 * med)
        for i, d in enumerate(diff):
            if float(d) > jump_at:
                issues.append({"kind": "jump", "start": i, "end": i + 1, "severity": "high" if float(d) > 2 * jump_at else "medium"})
                score -= 10
        # Flicker: a frame brighter (or darker) than both neighbours. A steady change, like something
        # dark sliding into frame, isn't flicker.
        up, down = mean[1:-1] - mean[:-2], mean[1:-1] - mean[2:]
        d2 = (mean[1:-1] - (mean[:-2] + mean[2:]) / 2).abs()
        pop = max(0.015, 4.0 * _median(d2))
        popped = (up * down > 0) & (up.abs() > 0.6 * pop) & (down.abs() > 0.6 * pop) & (d2 > pop)
        for a, b in _runs([bool(v) for v in popped]):
            issues.append({"kind": "flicker", "start": a + 1, "end": b + 1, "severity": "medium"})
            score -= 4
        # Soft frames: focus drops well below the clip's own sharpness for two frames or more.
        ms = _median(sharp)
        if ms > 1e-6:
            for a, b in _runs([bool(s < 0.45 * ms) for s in sharp]):
                if b == a:  # one soft frame is noise (a fast pan, a flash); a drop-out lasts
                    continue
                issues.append({"kind": "soft", "start": a, "end": b, "severity": "low" if b - a < 3 else "medium"})
                score -= min(20, 1.5 * (b - a + 1))

    if float(clip.mean()) > 0.05:
        issues.append({"kind": "blown", "start": 0, "end": N - 1, "severity": "medium", "amount": round(float(clip.mean()), 3)})
        score -= 10
    # Night scenes are meant to be dark: only a frame that's largely pure black counts.
    if float(crush.mean()) > 0.35:
        issues.append({"kind": "crushed", "start": 0, "end": N - 1, "severity": "low", "amount": round(float(crush.mean()), 3)})
        score -= 3

    issues.sort(key=lambda i: (i["start"], i["kind"]))
    return {
        "version": 1,
        "frames": N,
        "fps": float(fps),
        "score": int(round(max(0.0, min(100.0, score)))),
        "issues": issues[:60],
        "curves": {"brightness": _curve(mean), "motion": _curve(diff), "sharpness": _curve(sharp / max(_median(sharp), 1e-6))},
    }


def report_json(images: torch.Tensor, fps: float = 24.0) -> str:
    return json.dumps(analyse(images, fps), separators=(",", ":"))
