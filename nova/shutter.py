"""NovaShutter: real shutter motion blur and speed changes for a clip.

Takes frames that were interpolated ``multiplier`` times (FrameInterpolate before it), so between
two original frames there are ``multiplier`` sub-frames. Each output frame averages the sub-frames
inside its exposure window: ``shutter_angle`` / 360 of a frame, centred on the frame's time. 180 is
the film standard; 0 keeps every frame sharp (no blur); 360 is the dreamiest.

``speed`` retimes the clip: 0.5 is half speed (the interpolated frames make it smooth), 2 is double.
The exposure window scales with it, as a real camera's would, so slow motion stays crisp.
"""

from __future__ import annotations

import math

import torch


def source_frames(n_sub: int, multiplier: int) -> int:
    """Original frame count of an interpolated batch ((N - 1) * m + 1 sub-frames)."""
    m = max(1, int(multiplier))
    return (max(1, int(n_sub)) - 1) // m + 1


def output_frames(n_src: int, speed: float) -> int:
    return max(1, int(math.floor((n_src - 1) / max(speed, 1e-3) + 1e-6)) + 1)


def retime(images: torch.Tensor, multiplier: int = 1, shutter_angle: float = 180.0, speed: float = 1.0, progress=None) -> torch.Tensor:
    """images [S,H,W,3] (S sub-frames) -> [F,H,W,3] at the clip's own frame rate."""
    m = max(1, int(multiplier))
    S = images.shape[0]
    n_src = source_frames(S, m)
    speed = float(min(max(speed, 0.1), 4.0))
    n_out = output_frames(n_src, speed)
    half = (max(0.0, min(float(shutter_angle), 360.0)) / 360.0) * speed / 2.0  # in source frames
    out = torch.empty((n_out, *images.shape[1:3], 3), dtype=torch.float32)
    for i in range(n_out):
        t = i * speed  # source-frame time of output frame i
        lo = max(0, int(math.ceil((t - half) * m - 1e-6)))
        hi = min(S - 1, int(math.floor((t + half) * m + 1e-6)))
        if hi < lo:  # a closed shutter, or a window between sub-frames: the nearest sub-frame
            lo = hi = min(S - 1, max(0, int(math.floor(t * m + 0.5))))  # half-up, so strides stay regular
        out[i] = images[lo : hi + 1, ..., :3].float().mean(0)
        if progress:
            progress(i + 1, n_out)
    return out
