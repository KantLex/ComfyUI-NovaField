"""NovaContactSheet: evenly sampled frames tiled into a grid (no gutters, black empty cells)."""

from __future__ import annotations

import math
from typing import List

import torch

from .util import resize_nhwc


def sample_indices(n_frames: int, count: int) -> List[int]:
    """``count`` indices spread evenly over [0, n_frames - 1], first and last included."""
    if n_frames <= 0:
        return []
    count = max(1, min(int(count), n_frames))
    if count == 1:
        return [0]
    return [int(round(i * (n_frames - 1) / (count - 1))) for i in range(count)]


def render(images: torch.Tensor, columns: int = 4, count: int = 8, cell_width: int = 320) -> torch.Tensor:
    """images [B,H,W,3] -> sheet [1, rows*cell_h, columns*cell_w, 3]."""
    B, H, W = images.shape[0], images.shape[1], images.shape[2]
    idx = sample_indices(B, count)
    cols = max(1, int(columns))
    cw = max(1, int(cell_width))
    ch = max(1, int(round(cw * H / W)))
    rows = max(1, math.ceil(len(idx) / cols))
    sheet = torch.zeros((1, rows * ch, cols * cw, 3), dtype=torch.float32)
    cells = resize_nhwc(images[idx, ..., :3].float(), ch, cw)
    for n in range(len(idx)):
        r, c = divmod(n, cols)
        sheet[0, r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = cells[n]
    return sheet.clamp(0.0, 1.0)
