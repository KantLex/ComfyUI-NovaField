"""Camera path math shared with the NovaField app (see docs/contract.md, section 1).

Everything here is plain Python / numpy in float64 so the results can be compared
bit-for-bit-ish (1e-9) against the TypeScript implementation through the fixtures in
``tests/fixtures/camera_cases.json``.

Conventions that the contract leaves implicit (the TypeScript side must do the same):

* Rotations (shake yaw / pitch) use Rodrigues' formula with the right-hand rule:
  ``rot(v, k, a) = v cos a + (k x v) sin a + k (k . v)(1 - cos a)``.
  Positive yaw (about +Y) turns a camera looking down -Z towards -X (left);
  positive pitch (about the camera right axis) tilts it up.
* The pitch axis is the camera right axis computed (step 7) from the *yawed* view vector.
* A degenerate cross product (``|f x Y| < 1e-9``) uses right = [1, 0, 0]. A degenerate
  view vector (target == pos) uses forward = [0, 0, -1].
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EASES = ("linear", "inOutSine", "inOutCubic", "outExpo", "inExpo")
WORLD_UP = np.array([0.0, 1.0, 0.0])
NEAR = 0.05
MAX_ACTORS = 3
MAX_PROPS = 8

DEFAULT_PATH: Dict[str, Any] = {
    "version": 1,
    "frames": 124,
    "fps": 24,
    "ease": "inOutCubic",
    "keys": [
        {"t": 0.0, "pos": [0, 1.5, 4.0], "target": [0, 1.4, 0], "fov": 40, "roll": 0},
        {"t": 1.0, "pos": [0, 1.5, 2.2], "target": [0, 1.4, 0], "fov": 40, "roll": 0},
    ],
    "shake": {"amount": 0.0, "freq": 1.2, "seed": 7},
}


# --------------------------------------------------------------------------- parsing
def _vec3(v: Any, name: str) -> List[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"camera_path: '{name}' must be a list of 3 numbers, got {v!r}")
    return [float(x) for x in v]


def _vec2(v: Any, name: str) -> List[float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"camera_path: '{name}' must be a list of 2 numbers [x, z], got {v!r}")
    return [float(x) for x in v]


def parse_camera_path(value: Any) -> Dict[str, Any]:
    """Parse and normalise a camera path (JSON string or dict).

    Returns a dict with ``frames`` (int >= 1), ``fps`` (float > 0), ``ease`` (str),
    ``keys`` (sorted by t; each with t, pos, target, fov, roll) and ``shake``
    (dict or None).
    """
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("camera_path is empty")
        data = json.loads(value)
    elif isinstance(value, dict):
        data = value
    else:
        raise ValueError(f"camera_path must be a JSON string or dict, got {type(value)!r}")

    frames = int(data.get("frames", 124))
    if frames < 1:
        raise ValueError("camera_path: 'frames' must be >= 1")
    fps = float(data.get("fps", 24))
    if fps <= 0:
        raise ValueError("camera_path: 'fps' must be > 0")
    ease_name = data.get("ease", "linear") or "linear"
    if ease_name not in EASES:
        raise ValueError(f"camera_path: unknown ease {ease_name!r} (expected one of {EASES})")

    raw_keys = data.get("keys") or []
    if not raw_keys:
        raise ValueError("camera_path: at least one key is required")
    keys = []
    for k in raw_keys:
        keys.append(
            {
                "t": float(k.get("t", 0.0)),
                "pos": _vec3(k["pos"], "pos"),
                "target": _vec3(k["target"], "target"),
                "fov": float(k.get("fov", 40.0)),
                "roll": float(k.get("roll", 0.0)),
            }
        )
    keys.sort(key=lambda k: k["t"])  # stable: equal t keep their order

    shake = data.get("shake")
    if shake:
        shake = {
            "amount": float(shake.get("amount", 0.0)),
            "freq": float(shake.get("freq", 1.2)),
            "seed": float(shake.get("seed", 0)),
        }
        if shake["amount"] <= 0:
            shake = None
    else:
        shake = None

    actors = None
    if data.get("actors") is not None:
        raw = data["actors"]
        if not isinstance(raw, list):
            raise ValueError("camera_path: 'actors' must be a list")
        actors = []
        for a in raw[:MAX_ACTORS]:  # contract: up to 3 stand-ins; extra entries are ignored
            frm = _vec2(a.get("from", [0, 0]), "actors.from")
            to = _vec2(a.get("to", frm), "actors.to")
            start = min(1.0, max(0.0, float(a.get("start", 0.0))))
            end = min(1.0, max(0.0, float(a.get("end", 1.0))))
            actors.append({"from": frm, "to": to, "height": float(a.get("height", 1.75)), "start": start, "end": end})

    props = None
    if data.get("props") is not None:
        raw = data["props"]
        if not isinstance(raw, list):
            raise ValueError("camera_path: 'props' must be a list")
        props = []
        for pr in raw[:MAX_PROPS]:  # contract 1c: up to 8 set pieces
            at = _vec2(pr.get("at", [0, 0]), "props.at")
            size = _vec3(pr.get("size", [1, 1, 1]), "props.size")
            props.append({"at": at, "size": [max(0.05, float(v)) for v in size], "rot": float(pr.get("rot", 0.0))})

    follow = None
    if data.get("follow"):
        fo = data["follow"]
        follow = {
            "actor": int(fo.get("actor", 0)),
            "height": float(fo.get("height", 1.4)),
            "amount": min(1.0, max(0.0, float(fo.get("amount", 1.0)))),
        }

    return {"frames": frames, "fps": fps, "ease": ease_name, "keys": keys, "shake": shake,
            "actors": actors, "follow": follow, "props": props}


# --------------------------------------------------------------------------- easing
def ease(name: str, t: float) -> float:
    t = float(t)
    if name == "linear":
        return t
    if name == "inOutSine":
        return -(math.cos(math.pi * t) - 1.0) / 2.0
    if name == "inOutCubic":
        return 4.0 * t * t * t if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0
    if name == "outExpo":
        return 1.0 if t == 1.0 else 1.0 - 2.0 ** (-10.0 * t)
    if name == "inExpo":
        return 0.0 if t == 0.0 else 2.0 ** (10.0 * t - 10.0)
    raise ValueError(f"unknown ease {name!r}")


# --------------------------------------------------------------------------- spline
def catmull_rom(p0, p1, p2, p3, u: float) -> np.ndarray:
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


# --------------------------------------------------------------------------- vectors
def _normalize(v: np.ndarray) -> Tuple[np.ndarray, float]:
    n = float(math.sqrt(float(np.dot(v, v))))
    return (v / n if n > 0 else v), n


def rotate_axis(v, axis, degrees: float) -> np.ndarray:
    """Rodrigues rotation of ``v`` about unit ``axis`` by ``degrees`` (right-hand rule)."""
    v = np.asarray(v, dtype=np.float64)
    k = np.asarray(axis, dtype=np.float64)
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def _forward_right(pos, target) -> Tuple[np.ndarray, np.ndarray]:
    f, n = _normalize(np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64))
    if n < 1e-12:
        f = np.array([0.0, 0.0, -1.0])
    r, rn = _normalize(np.cross(f, WORLD_UP))
    if rn < 1e-9:
        r = np.array([1.0, 0.0, 0.0])
    return f, r


def camera_basis(pos, target, roll: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contract steps 7-8. Returns (right', up', forward)."""
    f, r = _forward_right(pos, target)
    u = np.cross(r, f)
    th = math.radians(roll)
    c, s = math.cos(th), math.sin(th)
    r2 = r * c + u * s
    u2 = u * c - r * s
    return r2, u2, f


# --------------------------------------------------------------------------- shake
def shake_angles(i: int, fps: float, shake: Optional[Dict[str, float]]) -> Tuple[float, float]:
    """(yaw, pitch) in degrees for frame ``i``; zero when shake is off."""
    if not shake or shake.get("amount", 0.0) <= 0:
        return 0.0, 0.0
    amount = float(shake["amount"])
    f = float(shake.get("freq", 1.2))
    ph = float(shake.get("seed", 0)) * 0.7311
    s = i / fps
    tau = 2.0 * math.pi
    yaw = amount * 1.6 * (
        0.6 * math.sin(tau * f * s + 1.3 + ph)
        + 0.3 * math.sin(tau * 2.3 * f * s + 0.7 + ph)
        + 0.1 * math.sin(tau * 5.1 * f * s + 2.1 + ph)
    )
    pitch = amount * 1.1 * (
        0.6 * math.sin(tau * 0.9 * f * s + 0.4 + ph)
        + 0.3 * math.sin(tau * 2.7 * f * s + 1.9 + ph)
        + 0.1 * math.sin(tau * 4.3 * f * s + 0.2 + ph)
    )
    return yaw, pitch


def apply_shake(pos, target, yaw: float, pitch: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    v = np.asarray(target, dtype=np.float64) - pos
    v1 = rotate_axis(v, WORLD_UP, yaw)
    _, r = _forward_right(pos, pos + v1)
    v2 = rotate_axis(v1, r, pitch)
    return pos + v2


# --------------------------------------------------------------------------- actors (1b)
def t_linear(path: Dict[str, Any], i: int) -> float:
    frames = path["frames"]
    return (i / (frames - 1)) if frames > 1 else 0.0


def walk_progress(t: float, start: float, end: float) -> float:
    """Contract 1b: how far through its walk an actor is at linear time ``t`` (0 before start, 1 after end)."""
    if end <= start:
        return 1.0 if t >= start else 0.0
    return min(1.0, max(0.0, (t - start) / (end - start)))


def prop_corners(prop: Dict[str, Any]) -> List[List[float]]:
    """Contract 1c: the 8 corners of a set piece, index = yi*4 + sxi*2 + szi (bottom first)."""
    (cx, cz), (w, h, d) = prop["at"], prop["size"]
    th = math.radians(prop.get("rot", 0.0))
    c, s = math.cos(th), math.sin(th)
    out = []
    for y in (0.0, h):
        for sx in (-1, 1):
            for sz in (-1, 1):
                lx, lz = sx * w / 2, sz * d / 2
                out.append([cx + lx * c + lz * s, y, cz - lx * s + lz * c])
    return out


def actor_positions(path: Dict[str, Any], i: int) -> List[np.ndarray]:
    """World positions [x, 0, z] of every actor at frame ``i`` (inOutSine on t_lin, always)."""
    actors = path.get("actors") or []
    t = t_linear(path, i)
    out = []
    for a in actors:
        w = ease("inOutSine", walk_progress(t, a.get("start", 0.0), a.get("end", 1.0)))
        x = a["from"][0] + (a["to"][0] - a["from"][0]) * w
        z = a["from"][1] + (a["to"][1] - a["from"][1]) * w
        out.append(np.array([x, 0.0, z], dtype=np.float64))
    return out


def apply_follow(path: Dict[str, Any], i: int, target: np.ndarray) -> np.ndarray:
    """Contract 1b: pull the target towards the followed actor (after step 5, before shake)."""
    fo = path.get("follow")
    actors = path.get("actors") or []
    if not fo or not (0 <= fo["actor"] < len(actors)):
        return target
    p = actor_positions(path, i)[fo["actor"]]
    aim = np.array([p[0], fo["height"], p[2]], dtype=np.float64)
    return target + (aim - target) * fo["amount"]


# --------------------------------------------------------------------------- pose
def pose_at(path: Dict[str, Any], i: int) -> Dict[str, Any]:
    """Camera pose at frame ``i`` (contract steps 1-6). ``path`` must be parsed."""
    frames = path["frames"]
    t_lin = (i / (frames - 1)) if frames > 1 else 0.0
    t = ease(path["ease"], t_lin)
    keys = path["keys"]
    n = len(keys)

    if n == 1 or t <= keys[0]["t"]:
        k = keys[0]
        pos, target, fov, roll = np.array(k["pos"]), np.array(k["target"]), k["fov"], k["roll"]
    elif t >= keys[-1]["t"]:
        k = keys[-1]
        pos, target, fov, roll = np.array(k["pos"]), np.array(k["target"]), k["fov"], k["roll"]
    else:
        j = 0
        for jj in range(n - 1):
            if keys[jj]["t"] <= t < keys[jj + 1]["t"]:
                j = jj
                break
        k0 = keys[j - 1] if j > 0 else keys[j]
        k1 = keys[j]
        k2 = keys[j + 1]
        k3 = keys[j + 2] if j + 2 < n else keys[j + 1]
        u = (t - k1["t"]) / (k2["t"] - k1["t"])
        pos = catmull_rom(k0["pos"], k1["pos"], k2["pos"], k3["pos"], u)
        target = catmull_rom(k0["target"], k1["target"], k2["target"], k3["target"], u)
        fov = k1["fov"] + (k2["fov"] - k1["fov"]) * u
        roll = k1["roll"] + (k2["roll"] - k1["roll"]) * u

    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target = apply_follow(path, i, target)
    yaw, pitch = shake_angles(i, path["fps"], path["shake"])
    if yaw != 0.0 or pitch != 0.0:
        target = apply_shake(pos, target, yaw, pitch)
    return {"pos": pos, "target": target, "fov": float(fov), "roll": float(roll)}


def all_poses(path: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Arrays for every frame: pos (F,3), target (F,3), fov (F,), roll (F,),
    R (F,3,3) with rows right', up', forward."""
    F = path["frames"]
    pos = np.zeros((F, 3))
    target = np.zeros((F, 3))
    fov = np.zeros(F)
    roll = np.zeros(F)
    R = np.zeros((F, 3, 3))
    for i in range(F):
        p = pose_at(path, i)
        pos[i], target[i], fov[i], roll[i] = p["pos"], p["target"], p["fov"], p["roll"]
        r, u, f = camera_basis(p["pos"], p["target"], p["roll"])
        R[i] = np.stack([r, u, f])
    return {"pos": pos, "target": target, "fov": fov, "roll": roll, "R": R}


# --------------------------------------------------------------------------- projection
def focal_px(fov_deg: float, height: int) -> float:
    return (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def project(P, pose: Dict[str, Any], width: int, height: int):
    """Project world point P with a pose dict. Returns (px, py, zc, visible)."""
    r, u, f = camera_basis(pose["pos"], pose["target"], pose["roll"])
    d = np.asarray(P, dtype=np.float64) - np.asarray(pose["pos"], dtype=np.float64)
    xc, yc, zc = float(np.dot(d, r)), float(np.dot(d, u)), float(np.dot(d, f))
    if zc <= NEAR:
        return None, None, zc, False
    fo = focal_px(pose["fov"], height)
    px = width / 2.0 + fo * xc / zc
    py = height / 2.0 - fo * yc / zc
    return px, py, zc, True


def project_point(P: Sequence[float], path: Dict[str, Any], i: int, width: int, height: int):
    return project(P, pose_at(path, i), width, height)
