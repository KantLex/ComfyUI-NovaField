# ComfyUI-NovaField

Custom nodes for the NovaField app: camera-motion carrier videos, film and lens looks, keyframed
track masks and contact sheets. The shared spec is `docs/contract.md` in the NovaField repo; the
app compiles ComfyUI graphs against these class names and reimplements the same camera math.

Pure **torch + numpy + standard library**. There is nothing to pip install.

## Install

Any of these works; nothing extra needs to be pip-installed. The nodes appear under the
**NovaField** category after ComfyUI restarts.

- **From the NovaField app:** any locked feature (Ghost rig, Film lab, Replace) shows an
  **Install NovaField nodes** button. It installs through ComfyUI Manager and restarts ComfyUI for you.
- **ComfyUI Manager:** open Manager → Custom Nodes Manager, search for **NovaField**, click
  Install, then restart ComfyUI.
- **comfy-cli:**

  ```bash
  comfy node install comfyui-novafield
  ```

- **From GitHub:** in Manager choose **Install via Git URL** and paste
  `https://github.com/KantLex/ComfyUI-NovaField`. Manager treats Git-URL installs as high-risk,
  so you may first need to lower its security level (Manager → settings).
- **By hand:** clone into ComfyUI's `custom_nodes` folder and restart ComfyUI:

  ```bash
  git clone https://github.com/KantLex/ComfyUI-NovaField
  ```

  For the ComfyUI desktop app on Windows, `custom_nodes` is inside the install folder, for
  example `C:\Users\<you>\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes`.

## Nodes

### NovaGhostRig: camera carrier
Renders a previs-style "ghost rig" video that follows a `camera_path` JSON (contract §1).

| input | type | notes |
|---|---|---|
| `camera_path` | STRING (multiline) | `{"frames","fps","ease","keys":[{t,pos,target,fov,roll}],"shake":{amount,freq,seed}}` |
| `width`, `height` | INT (step 8) | output size |
| `style` | `stage` · `subject` · `starfield` · `parallax` | scene, see below |
| `line_strength` | FLOAT 0.2–3 | scales line and star coverage |
| `image`, `depth` | IMAGE (optional) | used by `parallax` |

Outputs: `frames` IMAGE `[F,H,W,3]` and `frame_count` INT.

* `stage`: a floor grid (x and z in [-12, 12], every 1 m, with brighter axes) plus six wireframe pillars.
* `subject`: the stage plus a warm capsule "stand-in" at the origin (r 0.32, 1.75 m tall). When the
  path has an `actors` list, it draws one capsule per actor instead, at that actor's position for the frame and scaled
  uniformly by `height / 1.75` (radius, ring spacing and meridians). The colours by index are #FFB46B, #7FC8F8 and
  #B7E37D. An empty list draws no figures. Other styles ignore `actors`.
* `starfield`: 2400 deterministic LCG stars in a shell of 8–30 m.
* `parallax`: the `image` is treated as the frame-0 photo, lifted to 3D using `depth`
  (white = near, `z = 1.5 + (1 - d) * 10.5`), or a vertical ramp when no depth is given. It is
  re-projected every frame with a z-buffer and the holes are filled. Without an `image`, a
  built-in colour test pattern is used.

Rendering details (the app's 3D preview can match these):
* Lines are sampled about every 0.5 px in screen space after near-plane (z = 0.05) and screen
  clipping, with perspective-correct depth, and splatted bilinearly, so they come out anti-aliased.
* Depth fade: `fade(z) = 0.12 + 0.88 * exp(-z / 16)`, where z is the camera-space depth in metres.
  It scales each sample's coverage.
* Composite: `alpha = 1 - exp(-1.7 * coverage)`, `out = bg * (1 - alpha) + colour * alpha` over `#0B0C10`.
* Star position from LCG draws a, b, c: `theta = 2πa`, `phi = acos(2b - 1)`, `r = 8 + 22c`,
  `x = r sinφ cosθ`, `y = r cosφ`, `z = r sinφ sinθ`.
* Parallax: depth is the planar camera-space z along each pixel's ray, with pixel centres at +0.5.
  Rendering is 2× supersampled. Each point is splatted as a k×k block (k = its magnification
  since frame 0, 1–4) with an `amin` z-buffer. Depth edges are snapped to avoid flying pixels.
  Holes are filled by dilation that prefers the farthest neighbour, then push-pull.

Speed (CPU, Apple M-series, 124 frames at 864×480): stage ≈ 0.45 s, subject ≈ 0.5 s, two actors ≈ 0.55 s,
starfield ≈ 0.3 s, parallax ≈ 10 s. On CUDA, frames are processed in chunks and returned on the CPU.

### NovaLensFilm: film looks and lens character
Inputs: `images`, `look` (`none`, `kodak_2383`, `fuji_eterna`, `bleach_bypass`, `teal_orange`, `noir`,
`technicolor`, `moonlight`, `vhs_1989`, `kodak_portra_400`, `kodak_vision3_500t`, `ektachrome_e100`,
`cinestill_800t`, `fuji_velvia_50`, `kodachrome_64`, `ilford_hp5`, `instant_film`, `super8`),
`intensity` 0–1 (blends the look's grade), `grain`, `halation`, `vignette`, `chromatic`, `gate_weave`,
`anamorphic_streak` (each 0–1), `seed`.
Optional (0.3.0, neutral by default): `exposure` (stops, −2–2), `contrast`, `saturation`, `temperature`,
`tint` (each −1–1), `fade` 0–1 and `grain_size` 0.5–3 (default 1).
Output: `images` with the same shape. Frames run on the GPU in chunks of 16.

* Each look has its own tone curve, channel mix and split toning. `noir` and `ilford_hp5` are monochrome
  (`noir` with deeper contrast), and `vhs_1989` adds chroma bleed and scanline softness.
* The colour controls run after the look: `exposure` in stops, `temperature` warms (+) or cools (−),
  `tint` goes magenta (+) or green (−), `contrast` is an S-curve around mid grey, and `fade` lifts the
  blacks and dims the whites. `grain_size` scales the grain's blur.
* `grain` is seeded, changes every frame and depends on luminance (strongest in the midtones).
* `halation` adds blurred highlights back, tinted red-orange. `anamorphic_streak` adds very bright
  pixels blurred horizontally and tinted blue.
* `chromatic` scales the R and B channels radially. `gate_weave` is a smooth, seeded subpixel
  random walk applied per frame. `vignette` is a smooth elliptical falloff.
* With look `none`, every slider at 0 and the colour controls neutral, the input is returned unchanged.

### NovaBoxTrackMask: keyframed track to mask
`track` JSON: `{"shape":"ellipse"|"box","keys":[{"f":0,"x":0.4,"y":0.3,"w":0.2,"h":0.4}, ...]}`
(x, y = top-left corner, normalised 0–1). Other inputs: `width`, `height`, `frames`, `feather` 0–0.5,
`expand` 0–0.5, and optional `images`.
Outputs: `mask` MASK `[F,H,W]`, and `masked_images`: the source with the mask area filled with 0.5
grey (the Wan VACE `control_video` format), or the mask as RGB when no `images` are connected.

* Between keys, interpolation uses smoothstep of the local `u`. Before the first key and after the last, that key is held.
* `expand` grows the box about its centre by `expand × w` on each side horizontally and `expand × h` vertically.
* `feather` is a smoothstep falloff outside the expanded shape, `feather × size` wide, so the
  expanded shape is fully covered.
* `images` are resized to `width × height` if needed. When they are shorter than `frames`, the last frame is held.

### NovaContactSheet
Inputs: `images`, `columns` (4), `count` (8), `cell_width` (320). Output: `sheet` `[1, rows·cell_h, columns·cell_w, 3]`.
It samples `count` frames evenly (first and last included) and tiles them row by row, with no gutters.

## Stand-ins and follow (contract §1b)

Optional `camera_path` fields:

```json
"actors": [ { "from": [0, 0], "to": [2.5, -1.0], "height": 1.75 } ],
"follow": { "actor": 0, "height": 1.4, "amount": 1.0 }
```

* `actors` (up to 3, and any extras are ignored): `from` and `to` are floor positions `[x, z]`, and `height` defaults to 1.75.
  The position at frame i is `from + (to - from) * inOutSine(t_lin)`. This is always inOutSine, whatever the camera
  `ease` is. The world point is `[x, 0, z]`. `nova.camera.actor_positions(path, i)` returns them.
* `follow`: after key interpolation (step 5) and before shake (step 6),
  `target += ([p.x, height, p.z] - target) * amount`. `height` defaults to 1.4. `amount` defaults to 1 and is clamped to 0–1.
  `follow` is ignored when the actor index does not exist (including negative indices).
* A path without these fields gives exactly the same poses as before.

## Camera conventions not spelled out in the contract

* Shake rotations use Rodrigues' formula with the right-hand rule. Positive yaw (about +Y) turns a
  −Z view towards −X, and positive pitch tilts it up. The pitch axis is the camera right axis of the
  *yawed* view vector.
* If `|f × Y| < 1e-9`, right is `[1,0,0]`. If target == pos, forward is `[0,0,-1]`.
* Defaults for missing fields: `ease` linear, `fov` 40, `roll` 0, `fps` 24. No `shake`, or an
  amount ≤ 0, means no shake.

## Development

```
cd comfyui-novafield
uv run --python 3.12 --with torch --with numpy --with pytest pytest -q
uv run --python 3.12 --with torch --with numpy python tests/make_fixtures.py     # regenerate fixtures
uv run --python 3.12 --with torch --with numpy python tests/render_previews.py   # previews -> tests/_out/
```

`tests/fixtures/camera_cases.json` is the shared truth for the TypeScript camera implementation.
