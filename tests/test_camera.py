import json
import math
import pathlib

import numpy as np
import pytest

from comfyui_novafield.nova import camera

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "camera_cases.json"


@pytest.mark.parametrize("name", camera.EASES)
def test_ease_endpoints(name):
    assert camera.ease(name, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert camera.ease(name, 1.0) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", ["linear", "inOutSine", "inOutCubic"])
def test_symmetric_eases_midpoint(name):
    assert camera.ease(name, 0.5) == pytest.approx(0.5, abs=1e-12)


def test_ease_monotonic():
    ts = np.linspace(0, 1, 101)
    for name in camera.EASES:
        v = [camera.ease(name, t) for t in ts]
        assert all(b >= a - 1e-12 for a, b in zip(v, v[1:])), name


def test_catmull_rom_endpoints():
    p0, p1, p2, p3 = [0, 0, 0], [1, 2, 3], [4, 0, -1], [5, 5, 5]
    assert np.allclose(camera.catmull_rom(p0, p1, p2, p3, 0.0), p1)
    assert np.allclose(camera.catmull_rom(p0, p1, p2, p3, 1.0), p2)


def test_pose_passes_through_keys():
    keys = [
        {"t": 0.0, "pos": [0, 1, 4], "target": [0, 1, 0], "fov": 40, "roll": 0},
        {"t": 0.25, "pos": [2, 1.5, 3], "target": [0, 1.2, 0], "fov": 35, "roll": 5},
        {"t": 0.5, "pos": [3, 2, 0], "target": [0, 1.1, 0], "fov": 30, "roll": -5},
        {"t": 0.75, "pos": [1, 2.5, -3], "target": [0.2, 1, 0], "fov": 45, "roll": 0},
        {"t": 1.0, "pos": [-2, 1, -2], "target": [0, 1, 0.3], "fov": 50, "roll": 10},
    ]
    path = camera.parse_camera_path({"frames": 5, "fps": 24, "ease": "linear", "keys": keys})
    for i, k in enumerate(keys):
        p = camera.pose_at(path, i)
        assert np.allclose(p["pos"], k["pos"]), i
        assert np.allclose(p["target"], k["target"]), i
        assert p["fov"] == pytest.approx(k["fov"])
        assert p["roll"] == pytest.approx(k["roll"])


def test_single_key_and_single_frame():
    path = camera.parse_camera_path({"frames": 1, "keys": [{"t": 0.3, "pos": [1, 2, 3], "target": [0, 0, 0]}]})
    p = camera.pose_at(path, 0)
    assert np.allclose(p["pos"], [1, 2, 3]) and p["fov"] == 40.0


def test_projection_known_point():
    pose = {"pos": np.array([0.0, 0.0, 5.0]), "target": np.array([0.0, 0.0, 0.0]), "fov": 90.0, "roll": 0.0}
    # focal = 50 for H=100, fov 90
    px, py, zc, vis = camera.project([1.0, 0.0, 0.0], pose, 100, 100)
    assert vis and zc == pytest.approx(5.0)
    assert px == pytest.approx(60.0) and py == pytest.approx(50.0)  # +x world -> right of centre
    px, py, _, _ = camera.project([0.0, 1.0, 0.0], pose, 100, 100)
    assert px == pytest.approx(50.0) and py == pytest.approx(40.0)  # +y world -> up (smaller py)
    # roll +90: r' = u, so world +y lands right of centre
    pose["roll"] = 90.0
    px, py, _, _ = camera.project([0.0, 1.0, 0.0], pose, 100, 100)
    assert px == pytest.approx(60.0) and py == pytest.approx(50.0)
    # behind the camera
    _, _, _, vis = camera.project([0.0, 0.0, 6.0], pose, 100, 100)
    assert not vis


def test_degenerate_basis_looking_down():
    r, u, f = camera.camera_basis([0, 5, 0], [0, 0, 0], 0.0)
    assert np.allclose(r, [1, 0, 0]) and np.allclose(f, [0, -1, 0])
    assert np.allclose(u, np.cross(r, f))


def test_shake_rotation_conventions():
    # yaw +90 about +Y turns a -Z view towards -X; pitch +90 about right (+X) tilts it to +Y
    t = camera.apply_shake([0, 0, 0], [0, 0, -1], 90.0, 0.0)
    assert np.allclose(t, [-1, 0, 0], atol=1e-12)
    t = camera.apply_shake([0, 0, 0], [0, 0, -1], 0.0, 45.0)
    assert np.allclose(t, [0, math.sqrt(0.5), -math.sqrt(0.5)], atol=1e-12)


def test_shake_keeps_distance():
    path = camera.parse_camera_path({
        "frames": 50, "fps": 24, "ease": "linear",
        "keys": [{"t": 0, "pos": [0, 1.5, 4], "target": [0, 1.4, 0]}],
        "shake": {"amount": 1.0, "freq": 1.2, "seed": 3},
    })
    for i in range(50):
        p = camera.pose_at(path, i)
        assert np.linalg.norm(p["target"] - p["pos"]) == pytest.approx(np.linalg.norm([0, -0.1, -4]))


def test_fixtures_match_implementation():
    data = json.loads(FIXTURES.read_text())
    assert data["width"] == 864 and data["height"] == 480
    assert [c["name"] for c in data["cases"]] == ["push_in_inOutCubic", "orbit_9key_linear", "crane_3key_shake", "follow_walk", "timed_walk_props"]
    for case in data["cases"]:
        path = camera.parse_camera_path(case["path"])
        assert [s["frame"] for s in case["samples"]] == [0, 31, 62, 93, 123]
        for s in case["samples"]:
            pose = camera.pose_at(path, s["frame"])
            assert np.allclose(pose["pos"], s["pos"], atol=1e-9, rtol=0)
            assert np.allclose(pose["target"], s["target"], atol=1e-9, rtol=0)
            assert pose["fov"] == pytest.approx(s["fov"], abs=1e-9)
            assert pose["roll"] == pytest.approx(s["roll"], abs=1e-9)
            px, py, _, vis = camera.project(data["point"], pose, data["width"], data["height"])
            assert vis and s["pixel"] is not None
            assert px == pytest.approx(s["pixel"][0], abs=1e-6)
            assert py == pytest.approx(s["pixel"][1], abs=1e-6)
            if "actors" in s:
                got = camera.actor_positions(path, s["frame"])
                assert np.allclose(np.array(got), np.array(s["actors"]), atol=1e-12, rtol=0)
        if "prop_corners" in case:
            got = [camera.prop_corners(p) for p in path["props"]]
            assert np.allclose(np.array(got), np.array(case["prop_corners"]), atol=1e-12, rtol=0)


def _same(a, b):
    """Equal, with floats allowed to differ in their last digits (libm differs between platforms)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def test_fixture_file_is_up_to_date():
    import make_fixtures

    assert _same(json.loads(FIXTURES.read_text()), json.loads(json.dumps(make_fixtures.build())))


def _walk_path(**extra):
    p = {
        "frames": 11, "fps": 24, "ease": "inOutCubic",
        "keys": [{"t": 0, "pos": [0, 1.5, 5], "target": [0, 1.2, 0], "fov": 40},
                 {"t": 1, "pos": [1, 1.5, 4], "target": [0.5, 1.2, 0], "fov": 40}],
        "actors": [{"from": [-2, 0], "to": [2, -1]}, {"from": [1, 1], "to": [1, 1], "height": 1.2}],
    }
    p.update(extra)
    return camera.parse_camera_path(p)


def test_actor_interpolation_midpoint_and_ends():
    path = _walk_path()
    assert path["actors"][0]["height"] == 1.75 and path["actors"][1]["height"] == 1.2
    a0, a1 = camera.actor_positions(path, 0)
    assert np.allclose(a0, [-2, 0, 0]) and np.allclose(a1, [1, 0, 1])
    m0, _ = camera.actor_positions(path, 5)  # t_lin 0.5 -> inOutSine 0.5 (camera ease ignored)
    assert np.allclose(m0, [0.0, 0.0, -0.5])
    q0, _ = camera.actor_positions(path, 2)  # t_lin 0.2
    w = -(math.cos(math.pi * 0.2) - 1) / 2
    assert np.allclose(q0, [-2 + 4 * w, 0, -w])
    e0, _ = camera.actor_positions(path, 10)
    assert np.allclose(e0, [2, 0, -1])


def test_actors_capped_at_three():
    acts = [{"from": [i, 0], "to": [i, 0]} for i in range(5)]
    path = camera.parse_camera_path({"frames": 2, "keys": [{"t": 0, "pos": [0, 1, 4], "target": [0, 1, 0]}], "actors": acts})
    assert len(path["actors"]) == 3


def test_follow_amount_zero_is_no_change():
    base = _walk_path()
    fol = _walk_path(follow={"actor": 0, "height": 1.4, "amount": 0.0})
    for i in range(11):
        a, b = camera.pose_at(base, i), camera.pose_at(fol, i)
        assert np.array_equal(a["target"], b["target"]) and np.array_equal(a["pos"], b["pos"])


def test_follow_full_aims_at_actor():
    path = _walk_path(follow={"actor": 1, "height": 1.0})  # amount default 1
    for i in (0, 5, 10):
        p = camera.pose_at(path, i)
        assert np.allclose(p["target"], [1, 1.0, 1])
    half = _walk_path(follow={"actor": 0, "amount": 0.5})
    base = _walk_path()
    p, b = camera.pose_at(half, 5), camera.pose_at(base, 5)
    aim = np.array([0.0, 1.4, -0.5])
    assert np.allclose(p["target"], b["target"] + (aim - b["target"]) * 0.5)
    clamp = _walk_path(follow={"actor": 0, "amount": 7})
    assert clamp["follow"]["amount"] == 1.0


def test_follow_missing_actor_is_ignored():
    base = _walk_path()
    for fo in ({"actor": 2}, {"actor": -1}):
        path = _walk_path(follow=fo)
        for i in range(11):
            assert np.array_equal(camera.pose_at(path, i)["target"], camera.pose_at(base, i)["target"])
    no_actors = camera.parse_camera_path({"frames": 3, "keys": [{"t": 0, "pos": [0, 1, 4], "target": [0, 1, 0]}],
                                          "follow": {"actor": 0}})
    assert np.allclose(camera.pose_at(no_actors, 1)["target"], [0, 1, 0])


def test_paths_without_actors_unchanged():
    path = camera.parse_camera_path({"frames": 3, "keys": [{"t": 0, "pos": [0, 1, 4], "target": [0, 1, 0]}]})
    assert path["actors"] is None and path["follow"] is None
    assert camera.actor_positions(path, 1) == []


def test_walk_timing_holds_then_walks():
    """Contract 1b: a stand-in stays on its first mark until `start` and arrives at `end`."""
    path = camera.parse_camera_path({
        "frames": 11, "keys": [{"t": 0, "pos": [0, 1, 4], "target": [0, 1, 0]}],
        "actors": [{"from": [0, 0], "to": [4, 0], "start": 0.2, "end": 0.6}, {"from": [1, 1], "to": [3, 1], "start": 0.5, "end": 0.5}],
    })
    xs = [camera.actor_positions(path, i)[0][0] for i in range(11)]
    assert xs[0] == xs[2] == 0.0  # waiting
    assert xs[4] == pytest.approx(2.0)  # halfway through its walk at t = 0.4
    assert xs[6] == xs[10] == 4.0  # arrived and stays
    jump = [camera.actor_positions(path, i)[1][0] for i in (4, 5)]
    assert jump == [1.0, 3.0]  # end <= start: steps across at `start`


def test_default_walk_timing_unchanged():
    old = camera.parse_camera_path({"frames": 9, "keys": [{"t": 0, "pos": [0, 1, 4], "target": [0, 1, 0]}], "actors": [{"from": [0, 0], "to": [2, 2]}]})
    for i in range(9):
        w = camera.ease("inOutSine", i / 8)
        assert camera.actor_positions(old, i)[0][0] == pytest.approx(2 * w, abs=1e-12)


def test_props_parse_and_corners():
    path = camera.parse_camera_path({
        "frames": 2, "keys": [{"t": 0, "pos": [0, 1, 4], "target": [0, 1, 0]}],
        "props": [{"at": [1, 2], "size": [2, 1, 4], "rot": 90}] + [{"at": [0, 0]}] * 10,
    })
    assert len(path["props"]) == camera.MAX_PROPS
    assert path["props"][1] == {"at": [0.0, 0.0], "size": [1.0, 1.0, 1.0], "rot": 0.0}
    c = np.array(camera.prop_corners(path["props"][0]))
    # Turned 90°, the 2 m width runs along z and the 4 m depth along x.
    assert c[:, 0].min() == pytest.approx(-1) and c[:, 0].max() == pytest.approx(3)
    assert c[:, 2].min() == pytest.approx(1) and c[:, 2].max() == pytest.approx(3)
    assert c[:4, 1].max() == 0 and c[4:, 1].min() == 1
