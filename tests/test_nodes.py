import json
import math

import pytest
import torch

import comfyui_novafield as pkg
from comfyui_novafield import nodes
from comfyui_novafield.nova import box_track, contact_sheet, ghost_rig, lens_film


def small_path(frames=6):
    return json.dumps({
        "version": 1, "frames": frames, "fps": 24, "ease": "inOutCubic",
        "keys": [
            {"t": 0.0, "pos": [0, 1.5, 4.0], "target": [0, 1.0, 0], "fov": 40, "roll": 0},
            {"t": 1.0, "pos": [1.0, 1.6, 2.5], "target": [0, 1.0, 0], "fov": 40, "roll": 0},
        ],
        "shake": {"amount": 0.3, "freq": 1.2, "seed": 7},
    })


def check_image(x, shape):
    assert isinstance(x, torch.Tensor)
    assert tuple(x.shape) == tuple(shape)
    assert x.dtype == torch.float32
    assert x.device.type == "cpu"
    assert torch.isfinite(x).all()
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_mappings():
    names = {"NovaGhostRig", "NovaLensFilm", "NovaBoxTrackMask", "NovaContactSheet"}
    assert set(pkg.NODE_CLASS_MAPPINGS) == names
    assert set(pkg.NODE_DISPLAY_NAME_MAPPINGS) == names
    assert pkg.WEB_DIRECTORY is None
    for cls in pkg.NODE_CLASS_MAPPINGS.values():
        assert cls.CATEGORY == "NovaField"
        it = cls.INPUT_TYPES()
        assert "required" in it
        assert hasattr(cls, cls.FUNCTION)
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)


def test_input_contract():
    g = nodes.NovaGhostRig.INPUT_TYPES()
    assert list(g["required"]) == ["camera_path", "width", "height", "style", "line_strength"]
    assert g["required"]["style"][0] == ["stage", "subject", "starfield", "parallax"]
    assert g["required"]["width"][1]["step"] == 8 and g["required"]["height"][1]["step"] == 8
    assert g["required"]["line_strength"][1]["min"] == 0.2 and g["required"]["line_strength"][1]["max"] == 3.0
    assert set(g["optional"]) == {"image", "depth"}
    assert nodes.NovaGhostRig.RETURN_TYPES == ("IMAGE", "INT")
    assert nodes.NovaGhostRig.RETURN_NAMES == ("frames", "frame_count")

    lf = nodes.NovaLensFilm.INPUT_TYPES()["required"]
    assert list(lf) == ["images", "look", "intensity", "grain", "halation", "vignette", "chromatic",
                        "gate_weave", "anamorphic_streak", "seed"]
    assert lf["look"][0] == ["none", "kodak_2383", "fuji_eterna", "bleach_bypass", "teal_orange", "noir",
                             "technicolor", "moonlight", "vhs_1989"]
    assert lf["intensity"][1]["default"] == 0.8

    bt = nodes.NovaBoxTrackMask.INPUT_TYPES()
    assert list(bt["required"]) == ["track", "width", "height", "frames", "feather", "expand"]
    assert bt["required"]["feather"][1]["default"] == 0.08 and bt["required"]["expand"][1]["default"] == 0.05
    assert nodes.NovaBoxTrackMask.RETURN_TYPES == ("MASK", "IMAGE")
    assert nodes.NovaBoxTrackMask.RETURN_NAMES == ("mask", "masked_images")

    cs = nodes.NovaContactSheet.INPUT_TYPES()["required"]
    assert [cs[k][1]["default"] for k in ("columns", "count", "cell_width")] == [4, 8, 320]


@pytest.mark.parametrize("style", ["stage", "subject", "starfield", "parallax"])
def test_ghost_rig_shapes(style):
    frames, count = nodes.NovaGhostRig().render(small_path(6), 96, 64, style, 1.0)
    check_image(frames, (6, 64, 96, 3))
    assert count == 6
    assert float(frames.std()) > 0.005  # something was drawn


def test_ghost_rig_parallax_with_inputs():
    img = torch.rand(2, 50, 70, 3)
    depth = torch.rand(1, 40, 40, 3)
    frames, count = nodes.NovaGhostRig().render(small_path(4), 80, 48, "parallax", 1.0, image=img, depth=depth)
    check_image(frames, (4, 48, 80, 3))


def test_ghost_rig_parallax_frame0_reproduces_image():
    """With a static camera the parallax render must give the input photo back."""
    path = json.dumps({"frames": 2, "keys": [{"t": 0, "pos": [0, 1.5, 4], "target": [0, 1.4, 0], "fov": 40}]})
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 48), torch.linspace(0, 1, 80), indexing="ij")
    img = torch.stack([xx, yy, 0.5 * torch.ones_like(xx)], -1).unsqueeze(0)
    frames, _ = nodes.NovaGhostRig().render(path, 80, 48, "parallax", 1.0, image=img)
    assert torch.allclose(frames[0], img[0], atol=0.02)


def test_ghost_rig_subject_centred():
    """Camera targeting the subject's middle puts the warm subject colour at the frame centre."""
    path = json.dumps({"frames": 1, "keys": [{"t": 0, "pos": [0, 0.875, 4], "target": [0, 0.875, 0], "fov": 40}]})
    frames, _ = nodes.NovaGhostRig().render(path, 160, 96, "subject", 1.0)
    f = frames[0]
    warm = (f[..., 0] - f[..., 2]) > 0.1  # orange-ish pixels
    ys, xs = torch.nonzero(warm, as_tuple=True)
    assert xs.numel() > 50
    assert abs(float(xs.float().mean()) + 0.5 - 80) < 3
    assert abs(float(ys.float().mean()) + 0.5 - 48) < 4


def test_starfield_lcg_first_point():
    P = ghost_rig.starfield_points(3)
    state = 1234
    draws = []
    for _ in range(3):
        state = (state * 1664525 + 1013904223) % 2 ** 32
        draws.append(state / 2 ** 32)
    a, b, c = draws
    r = 8 + 22 * c
    phi = math.acos(2 * b - 1)
    assert P[0].tolist() == pytest.approx([r * math.sin(phi) * math.cos(2 * math.pi * a), r * math.cos(phi),
                                           r * math.sin(phi) * math.sin(2 * math.pi * a)])
    assert all(8 <= (p ** 2).sum() ** 0.5 <= 30 for p in ghost_rig.starfield_points(200))


def _test_clip(B=3, H=40, W=56, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    base = torch.stack([xx, yy, 1 - xx * yy], -1)
    return (base.unsqueeze(0).repeat(B, 1, 1, 1) * 0.9 + 0.1 * torch.rand(B, H, W, 3, generator=g)).clamp(0, 1)


def test_lens_film_identity():
    x = _test_clip()
    (y,) = nodes.NovaLensFilm().apply(x, "none", 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 123)
    assert y.shape == x.shape
    assert torch.max((y - x).abs()) <= 1e-6
    (y,) = nodes.NovaLensFilm().apply(x, "kodak_2383", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    assert torch.max((y - x).abs()) <= 1e-6


@pytest.mark.parametrize("look", lens_film.LOOKS)
def test_lens_film_looks(look):
    x = _test_clip()
    (y,) = nodes.NovaLensFilm().apply(x, look, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 7)
    check_image(y, x.shape)
    (g,) = nodes.NovaLensFilm().apply(x, look, 1.0, 0, 0, 0, 0, 0, 0, 7)
    if look != "none":
        assert float((g - x).abs().mean()) > 0.01  # the look does something
    if look == "noir":
        assert torch.allclose(g[..., 0], g[..., 1], atol=1e-5) and torch.allclose(g[..., 1], g[..., 2], atol=1e-5)


def test_lens_film_effects_individually_and_seeded():
    x = _test_clip(B=4)
    for i in range(6):
        sl = [0.0] * 6
        sl[i] = 1.0
        (y,) = nodes.NovaLensFilm().apply(x, "none", 0.8, *sl, 5)
        check_image(y, x.shape)
        if i != 1:  # halation needs highlights; the test clip barely has any
            assert float((y - x).abs().max()) > 1e-3, i
    (a,) = nodes.NovaLensFilm().apply(x, "none", 0.8, 1.0, 0, 0, 0, 0, 0, 5)
    (b,) = nodes.NovaLensFilm().apply(x, "none", 0.8, 1.0, 0, 0, 0, 0, 0, 5)
    (c,) = nodes.NovaLensFilm().apply(x, "none", 0.8, 1.0, 0, 0, 0, 0, 0, 6)
    assert torch.equal(a, b) and not torch.equal(a, c)
    # grain varies over time
    assert not torch.allclose(a[0] - x[0], a[1] - x[1])


def test_lens_film_chunking_consistent():
    x = _test_clip(B=5)
    a = lens_film.apply(x, "vhs_1989", 0.9, 0.4, 0.3, 0.3, 0.3, 0.6, 0.3, seed=9, chunk=16)
    b = lens_film.apply(x, "vhs_1989", 0.9, 0.4, 0.3, 0.3, 0.3, 0.6, 0.3, seed=9, chunk=2)
    assert torch.allclose(a, b, atol=1e-5)


def test_box_track_midpoint_interpolation():
    track = json.dumps({"shape": "box", "keys": [
        {"f": 0, "x": 0.1, "y": 0.2, "w": 0.2, "h": 0.4},
        {"f": 10, "x": 0.5, "y": 0.2, "w": 0.2, "h": 0.4},
    ]})
    b = box_track.box_at(box_track.parse_track(track), 5)
    assert b["x"] == pytest.approx(0.3)  # smoothstep(0.5) = 0.5
    b = box_track.box_at(box_track.parse_track(track), 2.5)
    assert b["x"] == pytest.approx(0.1 + 0.4 * 0.15625)  # smoothstep(0.25)
    mask, rgb = nodes.NovaBoxTrackMask().render(track, 100, 50, 14, 0.0, 0.0)
    assert tuple(mask.shape) == (14, 50, 100) and mask.dtype == torch.float32
    check_image(rgb, (14, 50, 100, 3))
    cols = torch.nonzero(mask[5, 25] > 0.5).flatten()
    assert int(cols.min()) == 30 and int(cols.max()) == 49
    rows = torch.nonzero(mask[5, :, 40] > 0.5).flatten()
    assert int(rows.min()) == 10 and int(rows.max()) == 29
    # hold before the first / after the last key
    assert torch.equal(mask[10], mask[13])
    assert torch.allclose(rgb[..., 0], mask)


def test_box_track_feather_expand_ellipse_and_images():
    track = json.dumps({"shape": "ellipse", "keys": [{"f": 0, "x": 0.4, "y": 0.3, "w": 0.2, "h": 0.4}]})
    imgs = torch.rand(3, 30, 40, 3)
    mask, masked = nodes.NovaBoxTrackMask().render(track, 80, 60, 4, 0.08, 0.05, images=imgs)
    assert tuple(mask.shape) == (4, 60, 80)
    check_image(masked, (4, 60, 80, 3))
    assert float(mask.min()) >= 0 and float(mask.max()) <= 1
    assert float(mask[0, 30, 40]) == pytest.approx(1.0)
    assert float(mask[0, 2, 2]) == 0.0
    soft = ((mask[0] > 0.01) & (mask[0] < 0.99)).sum()
    assert int(soft) > 0  # feathered edge exists
    # grey fill where the mask is 1
    assert torch.allclose(masked[0, 30, 40], torch.tensor([0.5, 0.5, 0.5]))
    # expansion grows the area
    m0, _ = box_track.render(track, 80, 60, 1, 0.0, 0.0)
    m1, _ = box_track.render(track, 80, 60, 1, 0.0, 0.2)
    assert m1.sum() > m0.sum() * 1.5


def test_contact_sheet():
    x = torch.rand(20, 36, 64, 3)
    (sheet,) = nodes.NovaContactSheet().render(x, 4, 8, 64)
    check_image(sheet, (1, 2 * 36, 4 * 64, 3))
    assert contact_sheet.sample_indices(20, 8) == [0, 3, 5, 8, 11, 14, 16, 19]
    assert contact_sheet.sample_indices(3, 8) == [0, 1, 2]
    (sheet,) = nodes.NovaContactSheet().render(x[:3], 2, 8, 32)
    check_image(sheet, (1, 2 * 18, 2 * 32, 3))
    # first cell equals the (resized) first frame
    (s1,) = nodes.NovaContactSheet().render(x[:1], 1, 1, 64)
    assert torch.allclose(s1[0], x[0], atol=1e-6)


def _actor_rig(actors):
    path = {"frames": 1, "keys": [{"t": 0, "pos": [0, 1.0, 6], "target": [0, 0.9, 0], "fov": 40}]}
    if actors is not None:
        path["actors"] = actors
    frames, _ = nodes.NovaGhostRig().render(json.dumps(path), 192, 112, "subject", 1.0)
    f = frames[0]
    warm = (f[..., 0] - f[..., 2]) > 0.12          # #FFB46B
    blue = (f[..., 2] - f[..., 0]) > 0.12          # #7FC8F8
    return f, warm, blue


def test_ghost_rig_two_actors():
    f, warm, blue = _actor_rig([{"from": [-1.5, 0], "to": [-1.5, 0]}, {"from": [1.5, 0], "to": [1.5, 0], "height": 1.2}])
    W = f.shape[1]
    wy, wx = torch.nonzero(warm, as_tuple=True)
    by, bx = torch.nonzero(blue, as_tuple=True)
    assert wx.numel() > 30 and bx.numel() > 30
    assert float(wx.float().mean()) < W / 2 - 15     # actor 0 on the left (x = -1.5)
    assert float(bx.float().mean()) > W / 2 + 15     # actor 1 on the right (x = +1.5)
    assert int(warm[:, W // 2 - 4 : W // 2 + 4].sum()) == 0  # no origin figure any more
    # the shorter actor (1.2 m) reaches less high on screen than the 1.75 m one
    assert int(by.min()) > int(wy.min()) + 5


def test_ghost_rig_empty_actor_list_has_no_figures():
    _, warm, blue = _actor_rig([])
    assert int(warm.sum()) == 0 and int(blue.sum()) == 0
    _, warm, _ = _actor_rig(None)  # no key: the classic origin figure
    assert int(warm.sum()) > 30


def test_ghost_rig_actor_moves_over_time():
    path = {"frames": 3, "keys": [{"t": 0, "pos": [0, 1.0, 6], "target": [0, 0.9, 0], "fov": 40}],
            "actors": [{"from": [-2, 0], "to": [2, 0]}]}
    frames, _ = nodes.NovaGhostRig().render(json.dumps(path), 192, 112, "subject", 1.0)
    xs = []
    for i in range(3):
        warm = (frames[i, ..., 0] - frames[i, ..., 2]) > 0.12
        xs.append(float(torch.nonzero(warm, as_tuple=True)[1].float().mean()))
    assert xs[0] < xs[1] < xs[2]  # walks towards +x = screen right
    assert abs(xs[1] + 0.5 - 96) < 4
    # other styles ignore actors
    st, _ = nodes.NovaGhostRig().render(json.dumps(path), 64, 48, "stage", 1.0)
    path.pop("actors")
    st2, _ = nodes.NovaGhostRig().render(json.dumps(path), 64, 48, "stage", 1.0)
    assert torch.equal(st, st2)


def test_ghost_rig_draws_props():
    """A set piece in front of the lens adds its colour to the frame; the styles without a set ignore it."""
    base = {"frames": 1, "keys": [{"t": 0, "pos": [0, 1.2, 5], "target": [0, 0.6, 0], "fov": 40}], "actors": []}
    with_prop = dict(base, props=[{"at": [0, 1.5], "size": [1.6, 0.75, 0.9], "rot": 20}])
    plain, _ = nodes.NovaGhostRig().render(json.dumps(base), 160, 96, "stage", 1.0)
    boxed, _ = nodes.NovaGhostRig().render(json.dumps(with_prop), 160, 96, "stage", 1.0)
    assert float((boxed - plain).abs().sum()) > 5
    stars_a, _ = nodes.NovaGhostRig().render(json.dumps(base), 64, 48, "starfield", 1.0)
    stars_b, _ = nodes.NovaGhostRig().render(json.dumps(with_prop), 64, 48, "starfield", 1.0)
    assert torch.equal(stars_a, stars_b)
