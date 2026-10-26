"""Render visual previews into tests/_out/ (PNG stills + mp4 via ffmpeg).

Run:  uv run --python 3.12 --with torch --with numpy python tests/render_previews.py
"""

import json
import math
import pathlib
import shutil
import struct
import subprocess
import sys
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _pkg  # noqa: E402

_pkg.load()
from comfyui_novafield.nova import box_track, contact_sheet, ghost_rig, lens_film  # noqa: E402
import make_fixtures  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "_out"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
W, H = 864, 480


# --------------------------------------------------------------------------- io
def write_png(path: pathlib.Path, img) -> None:
    """img: [H,W,3] float 0..1 (torch or numpy) -> 8-bit RGB PNG (zlib, no PIL)."""
    a = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def write_mp4(path: pathlib.Path, frames: torch.Tensor, fps: int = 24) -> None:
    F_, h, w, _ = frames.shape
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(path)]
    data = (frames.clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy().tobytes()
    subprocess.run(cmd, input=data, check=True)


def save_sequence(name: str, frames: torch.Tensor, stills=(0, 31, 62, 93, 123)) -> None:
    write_mp4(OUT / f"{name}.mp4", frames)
    for i in stills:
        if i < frames.shape[0]:
            write_png(OUT / f"{name}_f{i:03d}.png", frames[i])
    (sheet,) = (contact_sheet.render(frames, 4, 8, 216),)
    write_png(OUT / f"{name}_sheet.png", sheet[0])


# --------------------------------------------------------------------------- synthetic inputs
def synthetic_scene(h=H, w=W):
    """A little layered 'photo' + matching depth map (white = near)."""
    yy, xx = torch.meshgrid((torch.arange(h) + 0.5) / h, (torch.arange(w) + 0.5) / w, indexing="ij")
    horizon = 0.55
    sky = torch.stack([0.35 + 0.3 * yy, 0.55 + 0.25 * yy, 0.85 - 0.1 * yy], -1)
    ground_c = torch.stack([0.30 + 0.15 * yy, 0.42 + 0.1 * yy, 0.22 + 0.0 * yy], -1)
    checker = ((((xx - 0.5) / (yy - horizon + 0.02).clamp_min(0.02) * 2).floor()
                + (1.0 / (yy - horizon + 0.02).clamp_min(0.02)).floor()) % 2).unsqueeze(-1)
    ground = ground_c * (0.8 + 0.2 * checker)
    is_sky = (yy < horizon).unsqueeze(-1)
    img = torch.where(is_sky, sky, ground)
    depth = torch.where(yy < horizon, torch.zeros_like(yy), ((yy - horizon) / (1 - horizon)) ** 0.7)
    # distant mountains
    m = horizon - 0.12 * (0.5 + 0.5 * torch.sin(xx * 9.0) * torch.cos(xx * 3.3))
    mount = (yy > m) & (yy < horizon)
    img[mount] = torch.tensor([0.35, 0.38, 0.5])
    depth[mount] = 0.08
    # three posts at different depths
    for cx, base, hgt, wid, d, col in ((0.22, 0.95, 0.55, 0.06, 0.95, (0.85, 0.3, 0.25)),
                                       (0.62, 0.78, 0.35, 0.04, 0.6, (0.95, 0.75, 0.3)),
                                       (0.80, 0.66, 0.18, 0.025, 0.3, (0.3, 0.6, 0.9))):
        sel = ((xx - cx).abs() < wid / 2) & (yy > base - hgt) & (yy < base)
        shade = 0.75 + 0.25 * torch.cos((xx - cx) / wid * math.pi)
        img[sel] = torch.tensor(col) * shade[sel].unsqueeze(-1)
        depth[sel] = d
    # sun highlight
    sun = ((xx - 0.7) ** 2 * (w / h) ** 2 + (yy - 0.18) ** 2) < 0.002
    img[sun] = torch.tensor([1.0, 0.98, 0.9])
    return img.unsqueeze(0), depth.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0)


def gradient_test_image(h=360, w=640):
    yy, xx = torch.meshgrid((torch.arange(h) + 0.5) / h, (torch.arange(w) + 0.5) / w, indexing="ij")
    hue = xx * 6.0
    r = (torch.abs((hue % 6) - 3) - 1).clamp(0, 1)
    g = (2 - torch.abs((hue % 6) - 2)).clamp(0, 1)
    b = (2 - torch.abs((hue % 6) - 4)).clamp(0, 1)
    rgb = torch.stack([r, g, b], -1)
    v = (1 - yy) * 0.95 + 0.03
    s = (0.35 + 0.65 * (1 - (yy - 0.5).abs() * 1.6)).clamp(0, 1)
    img = v.unsqueeze(-1) * (1 - s.unsqueeze(-1) + s.unsqueeze(-1) * rgb)
    # skin-ish patch, grey ramp and a few hot highlights (halation / streak sources)
    img[int(h * 0.62):int(h * 0.8), int(w * 0.06):int(w * 0.2)] = torch.tensor([0.86, 0.62, 0.5])
    ramp = torch.linspace(0, 1, w).view(1, w, 1).expand(int(h * 0.08), w, 3)
    img[int(h * 0.9):int(h * 0.9) + ramp.shape[0]] = ramp
    for cx, cy, rad in ((0.3, 0.42, 0.018), (0.55, 0.3, 0.012), (0.8, 0.55, 0.02)):
        sel = ((xx - cx) * w / h) ** 2 + (yy - cy) ** 2 < rad ** 2
        img[sel] = 1.0
    return img.clamp(0, 1).unsqueeze(0)


def tile(images, cols, pad=4):
    n, h, w, _ = images.shape
    rows = math.ceil(n / cols)
    out = torch.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 0.08)
    for i in range(n):
        r, c = divmod(i, cols)
        out[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = images[i]
    return out


# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timings = {}

    stage_path = make_fixtures.CASES[0]["path"]
    t = time.perf_counter()
    stage = ghost_rig.render(stage_path, W, H, "stage", 1.0)
    timings["stage_push_in_124f_cpu"] = time.perf_counter() - t
    save_sequence("stage_push_in", stage)

    orbit_path = make_fixtures.CASES[1]["path"]
    t = time.perf_counter()
    orbit = ghost_rig.render(orbit_path, W, H, "subject", 1.0)
    timings["subject_orbit_124f_cpu"] = time.perf_counter() - t
    save_sequence("subject_orbit", orbit)

    crane_path = make_fixtures.CASES[2]["path"]
    crane = ghost_rig.render(crane_path, W, H, "subject", 1.6)
    save_sequence("subject_crane_shake", crane)

    walk_path = make_fixtures.CASES[3]["path"]
    t = time.perf_counter()
    walk = ghost_rig.render(walk_path, W, H, "subject", 1.0)
    timings["follow_walk_2actors_124f_cpu"] = time.perf_counter() - t
    save_sequence("follow_walk", walk)

    star_path = {"frames": 124, "fps": 24, "ease": "inOutSine", "keys": [
        {"t": 0, "pos": [0, 0.5, 5.0], "target": [0, 0.5, -10], "fov": 50, "roll": 0},
        {"t": 1, "pos": [0, 0.5, -5.0], "target": [0, 0.5, -20], "fov": 50, "roll": 8}]}
    t = time.perf_counter()
    stars = ghost_rig.render(star_path, W, H, "starfield", 1.0)
    timings["starfield_dolly_124f_cpu"] = time.perf_counter() - t
    save_sequence("starfield_dolly", stars)

    img, depth = synthetic_scene()
    write_png(OUT / "parallax_input.png", img[0])
    write_png(OUT / "parallax_depth.png", depth[0])
    par_path = {"frames": 124, "fps": 24, "ease": "inOutCubic", "keys": [
        {"t": 0, "pos": [0, 1.5, 4.0], "target": [0, 1.4, 0], "fov": 40, "roll": 0},
        {"t": 1, "pos": [0.35, 1.45, 3.0], "target": [0, 1.4, 0], "fov": 40, "roll": 0}]}
    t = time.perf_counter()
    par = ghost_rig.render(par_path, W, H, "parallax", 1.0, image=img, depth=depth)
    timings["parallax_push_in_124f_cpu"] = time.perf_counter() - t
    save_sequence("parallax_push_in", par)
    par_nod = ghost_rig.render(par_path, W, H, "parallax", 1.0, image=img)
    save_sequence("parallax_push_in_ramp", par_nod, stills=(123,))

    src = gradient_test_image()
    tiles = []
    for look in lens_film.LOOKS:
        tiles.append(lens_film.apply(src, look, 1.0)[0])
    write_png(OUT / "lensfilm_looks.png", tile(torch.stack(tiles), 3))
    fx = []
    fx.append(src[0])
    for kw in (dict(grain=1.0), dict(halation=1.0), dict(vignette=1.0), dict(chromatic=1.0),
               dict(anamorphic_streak=1.0), dict(look="kodak_2383", intensity=0.8, grain=0.4, halation=0.5,
                                                 vignette=0.4, chromatic=0.3, gate_weave=0.5, anamorphic_streak=0.4)):
        fx.append(lens_film.apply(src, **kw)[0])
    write_png(OUT / "lensfilm_effects.png", tile(torch.stack(fx), 4))
    t = time.perf_counter()
    film = lens_film.apply(orbit, "kodak_2383", 0.8, grain=0.5, halation=0.4, vignette=0.5, chromatic=0.3,
                           gate_weave=0.6, anamorphic_streak=0.3, seed=3)
    timings["lensfilm_all_124f_cpu"] = time.perf_counter() - t
    save_sequence("subject_orbit_lensfilm", film, stills=(62,))

    track = {"shape": "ellipse", "keys": [{"f": 0, "x": 0.38, "y": 0.2, "w": 0.18, "h": 0.6},
                                          {"f": 123, "x": 0.5, "y": 0.25, "w": 0.14, "h": 0.55}]}
    mask, masked = box_track.render(json.dumps(track), W, H, 124, 0.08, 0.05, images=orbit)
    save_sequence("box_track_masked", masked, stills=(62,))

    print(json.dumps({k: round(v, 2) for k, v in timings.items()}, indent=2))
    print(f"previews in {OUT}")


if __name__ == "__main__":
    main()
