"""NovaBoxTrackMask: keyframed box / ellipse track -> feathered mask video.

Track JSON: ``{"shape": "ellipse" | "box", "keys": [{"f": 0, "x": .4, "y": .3, "w": .2, "h": .4}, ...]}``
x, y = top-left corner, all normalised 0..1; f = frame index.

* Interpolation between keys a, b: ``u = (i - a.f) / (b.f - a.f)``, ``s = u*u*(3 - 2u)``,
  x/y/w/h lerped with s. Before the first key / after the last key the key is held.
* ``expand`` grows the box about its centre by ``expand * w`` on the left and right and
  ``expand * h`` on the top and bottom (so w' = w (1 + 2 expand)).
* ``feather`` is a soft edge *outside* the expanded shape: the mask is 1 inside it and
  falls to 0 with a smoothstep over ``feather * w'`` horizontally / ``feather * h'``
  vertically (for the ellipse: over normalised radius 1 -> 1 + 2 feather).
* Pixel (j, i) is sampled at its centre ((j + 0.5) / W, (i + 0.5) / H).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch

from .util import resize_nhwc


def parse_track(value: Any) -> Dict[str, Any]:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        raise ValueError("track must be a JSON object")
    shape = str(data.get("shape", "box")).lower()
    if shape in ("rect", "rectangle", "box"):
        shape = "box"
    elif shape not in ("ellipse",):
        raise ValueError(f"track: unknown shape {shape!r} (expected 'box' or 'ellipse')")
    keys = []
    for k in data.get("keys") or []:
        keys.append({c: float(k[c]) for c in ("f", "x", "y", "w", "h")})
    if not keys:
        raise ValueError("track: at least one key is required")
    keys.sort(key=lambda k: k["f"])
    return {"shape": shape, "keys": keys}


def box_at(track: Dict[str, Any], i: float) -> Dict[str, float]:
    keys = track["keys"]
    if i <= keys[0]["f"]:
        return dict(keys[0])
    if i >= keys[-1]["f"]:
        return dict(keys[-1])
    for a, b in zip(keys[:-1], keys[1:]):
        if a["f"] <= i < b["f"]:
            u = (i - a["f"]) / (b["f"] - a["f"])
            s = u * u * (3.0 - 2.0 * u)
            return {c: a[c] + (b[c] - a[c]) * s for c in ("f", "x", "y", "w", "h")}
    return dict(keys[-1])


def _smoothstep01(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mask_frame(shape: str, box: Dict[str, float], width: int, height: int, feather: float, expand: float,
               device=None) -> torch.Tensor:
    xs = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width
    ys = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height
    w = max(box["w"], 0.0) * (1.0 + 2.0 * expand)
    h = max(box["h"], 0.0) * (1.0 + 2.0 * expand)
    cx = box["x"] + box["w"] / 2.0
    cy = box["y"] + box["h"] / 2.0
    if w <= 0 or h <= 0:
        return torch.zeros((height, width), dtype=torch.float32, device=device)
    dx = (xs - cx).abs().view(1, width)
    dy = (ys - cy).abs().view(height, 1)
    if shape == "ellipse":
        rho = torch.sqrt((dx / (w / 2)) ** 2 + (dy / (h / 2)) ** 2)
        if feather <= 0:
            return (rho <= 1.0).float()
        return 1.0 - _smoothstep01((rho - 1.0) / (2.0 * feather))
    ox = (dx - w / 2).clamp_min(0.0)
    oy = (dy - h / 2).clamp_min(0.0)
    if feather <= 0:
        return ((ox <= 0) & (oy <= 0)).float()
    d = torch.sqrt((ox / (feather * w)) ** 2 + (oy / (feather * h)) ** 2)
    return 1.0 - _smoothstep01(d)


def render(track_json: Any, width: int, height: int, frames: int, feather: float = 0.08, expand: float = 0.05,
           images: Optional[torch.Tensor] = None):
    """Returns (mask [F,H,W], masked_images [F,H,W,3]) float32 on CPU.

    When ``images`` is given it is resized to width x height if needed; if it has fewer
    frames than ``frames`` the last frame is held.
    """
    track = parse_track(track_json)
    Fn, H, W = int(frames), int(height), int(width)
    mask = torch.empty((Fn, H, W), dtype=torch.float32)
    for i in range(Fn):
        mask[i] = mask_frame(track["shape"], box_at(track, i), W, H, float(feather), float(expand))
    if images is None:
        rgb = mask.unsqueeze(-1).expand(Fn, H, W, 3).contiguous()
        return mask, rgb
    out = torch.empty((Fn, H, W, 3), dtype=torch.float32)
    B = images.shape[0]
    chunk = 16
    for s0 in range(0, Fn, chunk):
        s1 = min(Fn, s0 + chunk)
        idx = torch.arange(s0, s1).clamp(max=B - 1)
        img = resize_nhwc(images[idx, ..., :3].float(), H, W)
        m = mask[s0:s1].unsqueeze(-1)
        out[s0:s1] = img * (1.0 - m) + 0.5 * m
    return mask, out
