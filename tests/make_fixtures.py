"""Write tests/fixtures/camera_cases.json from the Python camera implementation.

Run:  python tests/make_fixtures.py
The TypeScript app reads the same file to check its camera math.
"""

import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _pkg  # noqa: E402

pkg = _pkg.load()
from comfyui_novafield.nova import camera  # noqa: E402

WIDTH, HEIGHT = 864, 480
POINT = [0.3, 1.2, -0.4]
SAMPLE_FRAMES = [0, 31, 62, 93, 123]
OUT = pathlib.Path(__file__).resolve().parent / "fixtures" / "camera_cases.json"


def orbit_keys():
    keys = []
    for i in range(9):
        a = 2 * math.pi * i / 8
        keys.append({
            "t": i / 8,
            "pos": [round(4.0 * math.sin(a), 6), 1.6, round(4.0 * math.cos(a), 6)],
            "target": [0, 1.0, 0],
            "fov": 40,
            "roll": 0,
        })
    return keys


CASES = [
    {
        "name": "push_in_inOutCubic",
        "path": {
            "version": 1, "frames": 124, "fps": 24, "ease": "inOutCubic",
            "keys": [
                {"t": 0.0, "pos": [0, 1.5, 4.0], "target": [0, 1.4, 0], "fov": 40, "roll": 0},
                {"t": 1.0, "pos": [0, 1.5, 2.2], "target": [0, 1.4, 0], "fov": 40, "roll": 0},
            ],
            "shake": {"amount": 0.0, "freq": 1.2, "seed": 7},
        },
    },
    {
        "name": "orbit_9key_linear",
        "path": {"version": 1, "frames": 124, "fps": 24, "ease": "linear", "keys": orbit_keys()},
    },
    {
        "name": "crane_3key_shake",
        "path": {
            "version": 1, "frames": 124, "fps": 24, "ease": "inOutSine",
            "keys": [
                {"t": 0.0, "pos": [0.0, 0.6, 4.5], "target": [0, 1.0, 0], "fov": 45, "roll": 0},
                {"t": 0.5, "pos": [0.6, 2.2, 3.8], "target": [0, 1.2, 0], "fov": 38, "roll": 3},
                {"t": 1.0, "pos": [1.2, 3.6, 3.0], "target": [0, 1.1, 0], "fov": 34, "roll": -2},
            ],
            "shake": {"amount": 0.6, "freq": 1.2, "seed": 7},
        },
    },
    {
        # contract 1b: two stand-ins, camera trucks alongside and follows actor 0
        "name": "follow_walk",
        "path": {
            "version": 1, "frames": 124, "fps": 24, "ease": "inOutSine",
            "keys": [
                {"t": 0.0, "pos": [-2.4, 1.5, 3.6], "target": [-2.0, 1.3, 0.0], "fov": 40, "roll": 0},
                {"t": 0.5, "pos": [0.3, 1.55, 3.2], "target": [0.25, 1.3, -0.5], "fov": 38, "roll": 0},
                {"t": 1.0, "pos": [2.9, 1.5, 2.6], "target": [2.5, 1.3, -1.0], "fov": 36, "roll": 0},
            ],
            "actors": [
                {"from": [-2, 0], "to": [2.5, -1], "height": 1.75},
                {"from": [1.5, 1], "to": [1.5, 1], "height": 1.2},
            ],
            "follow": {"actor": 0, "height": 1.4, "amount": 0.85},
            "shake": {"amount": 0.3, "freq": 1.1, "seed": 7},
        },
    },
    {
        # contract 1b timing + 1c set pieces: one stand-in waits, then walks in the middle of the shot
        "name": "timed_walk_props",
        "path": {
            "version": 1, "frames": 124, "fps": 24, "ease": "linear",
            "keys": [
                {"t": 0.15, "pos": [0.0, 1.6, 5.0], "target": [0, 1.2, 0], "fov": 40, "roll": 0},
                {"t": 1.0, "pos": [1.5, 1.6, 3.0], "target": [0.5, 1.2, 0], "fov": 40, "roll": 0},
            ],
            "actors": [
                {"from": [-1.5, 0], "to": [1.5, -0.5], "height": 1.75, "start": 0.25, "end": 0.75},
                {"from": [2, 1], "to": [2, 1], "height": 1.6, "start": 0.4, "end": 0.4},
            ],
            "props": [
                {"at": [0.0, -1.5], "size": [1.6, 0.75, 0.9], "rot": 30},
                {"at": [-3.0, 0.5], "size": [4.0, 2.6, 0.2]},
            ],
        },
    },
]


def build():
    cases = []
    for case in CASES:
        path = camera.parse_camera_path(case["path"])
        samples = []
        for i in SAMPLE_FRAMES:
            pose = camera.pose_at(path, i)
            px, py, zc, vis = camera.project(POINT, pose, WIDTH, HEIGHT)
            sample = {
                "frame": i,
                "pos": [float(v) for v in pose["pos"]],
                "target": [float(v) for v in pose["target"]],
                "fov": float(pose["fov"]),
                "roll": float(pose["roll"]),
                "pixel": [float(px), float(py)] if vis else None,
            }
            if path["actors"] is not None:
                sample["actors"] = [[float(v) for v in p] for p in camera.actor_positions(path, i)]
            samples.append(sample)
        out = {"name": case["name"], "path": case["path"], "samples": samples}
        if path["props"]:
            out["prop_corners"] = [camera.prop_corners(p) for p in path["props"]]
        cases.append(out)
    return {"width": WIDTH, "height": HEIGHT, "point": POINT, "cases": cases}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT}")
