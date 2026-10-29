"""NovaField ComfyUI nodes (classic V1 node API). See docs/contract.md section 3."""

from __future__ import annotations

import json

import torch

from .nova import box_track, camera, contact_sheet, ghost_rig, lens_film, match_color, shot_qc, shutter
from .nova.util import get_device

CATEGORY = "NovaField"

_DEFAULT_TRACK = json.dumps(
    {"shape": "ellipse", "keys": [{"f": 0, "x": 0.4, "y": 0.3, "w": 0.2, "h": 0.4}, {"f": 123, "x": 0.45, "y": 0.28, "w": 0.2, "h": 0.42}]},
    indent=2,
)


def _progress(total: int):
    """ComfyUI progress bar callback when available (lazy import), else None."""
    try:
        import comfy.utils  # type: ignore

        bar = comfy.utils.ProgressBar(total)
        return lambda done, _total: bar.update_absolute(done, _total)
    except Exception:
        return None


class NovaGhostRig:
    DESCRIPTION = "Renders a camera-motion carrier video (previs ghost rig) from a NovaField camera path."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "camera_path": ("STRING", {"multiline": True, "default": json.dumps(camera.DEFAULT_PATH, indent=2)}),
                "width": ("INT", {"default": 864, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 8}),
                "style": (list(ghost_rig.STYLES),),
                "line_strength": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05}),
            },
            "optional": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "frame_count")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, camera_path, width, height, style, line_strength, image=None, depth=None):
        path = camera.parse_camera_path(camera_path)
        frames = ghost_rig.render(
            path, int(width), int(height), style, float(line_strength),
            image=image, depth=depth, device=get_device(), progress=_progress(path["frames"]),
        )
        return (frames, int(frames.shape[0]))


class NovaLensFilm:
    DESCRIPTION = "Film emulation (looks, grain, halation) and lens character (vignette, chromatic, gate weave, anamorphic streak)."

    @classmethod
    def INPUT_TYPES(cls):
        f01 = lambda d=0.0: ("FLOAT", {"default": d, "min": 0.0, "max": 1.0, "step": 0.01})
        fpm = lambda: ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01})
        return {
            "required": {
                "images": ("IMAGE",),
                "look": (list(lens_film.LOOKS),),
                "intensity": f01(0.8),
                "grain": f01(),
                "halation": f01(),
                "vignette": f01(),
                "chromatic": f01(),
                "gate_weave": f01(),
                "anamorphic_streak": f01(),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            # Colour corrections and grain size (0.3.0). Optional and neutral by default, so graphs
            # made for earlier versions run unchanged.
            "optional": {
                "exposure": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "contrast": fpm(),
                "saturation": fpm(),
                "temperature": fpm(),
                "tint": fpm(),
                "fade": f01(),
                "grain_size": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, images, look, intensity, grain, halation, vignette, chromatic, gate_weave, anamorphic_streak, seed,
              exposure=0.0, contrast=0.0, saturation=0.0, temperature=0.0, tint=0.0, fade=0.0, grain_size=1.0):
        out = lens_film.apply(
            images, look=look, intensity=float(intensity), grain=float(grain), halation=float(halation),
            vignette=float(vignette), chromatic=float(chromatic), gate_weave=float(gate_weave),
            anamorphic_streak=float(anamorphic_streak), seed=int(seed), device=get_device(),
            progress=_progress(int(images.shape[0])),
            exposure=float(exposure), contrast=float(contrast), saturation=float(saturation),
            temperature=float(temperature), tint=float(tint), fade=float(fade), grain_size=float(grain_size),
        )
        return (out,)


class NovaBoxTrackMask:
    DESCRIPTION = "Keyframed box/ellipse track to a feathered mask video (+ grey-filled control video for Wan VACE)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track": ("STRING", {"multiline": True, "default": _DEFAULT_TRACK}),
                "width": ("INT", {"default": 864, "min": 8, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 480, "min": 8, "max": 8192, "step": 8}),
                "frames": ("INT", {"default": 124, "min": 1, "max": 100000}),
                "feather": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "expand": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "masked_images")
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, track, width, height, frames, feather, expand, images=None):
        mask, masked = box_track.render(track, int(width), int(height), int(frames), float(feather), float(expand), images)
        return (mask, masked)


class NovaContactSheet:
    DESCRIPTION = "Evenly samples frames from a batch and tiles them into a single contact-sheet image."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "columns": ("INT", {"default": 4, "min": 1, "max": 64}),
                "count": ("INT", {"default": 8, "min": 1, "max": 1024}),
                "cell_width": ("INT", {"default": 320, "min": 16, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, images, columns, count, cell_width):
        return (contact_sheet.render(images, int(columns), int(count), int(cell_width)),)


class NovaMatchColor:
    DESCRIPTION = "Colour continuity: gives a clip the colour (and optionally the brightness) of a reference picture, the same on every frame."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reference": ("IMAGE",),
                "amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "colour_only": ("BOOLEAN", {"default": False, "tooltip": "Match colour but keep the clip's own brightness and contrast."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, images, reference, amount, colour_only):
        return (match_color.match(images, reference, float(amount), bool(colour_only), progress=_progress(int(images.shape[0]))),)


class NovaShutter:
    DESCRIPTION = "Shutter motion blur and speed changes. Feed it frames interpolated `multiplier` times (FrameInterpolate)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "multiplier": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "The interpolation multiplier used before this node."}),
                "shutter_angle": ("FLOAT", {"default": 180.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, images, multiplier, shutter_angle, speed):
        out = shutter.retime(images, int(multiplier), float(shutter_angle), float(speed))
        return (out, int(out.shape[0]))


class NovaShotQC:
    DESCRIPTION = "Checks a clip for frozen frames, jumps, flicker, soft frames and exposure problems; returns a JSON report."
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    CATEGORY = CATEGORY

    def check(self, images, fps):
        report = shot_qc.report_json(images, float(fps))
        return {"ui": {"text": [report]}, "result": (report,)}


NODE_CLASS_MAPPINGS = {
    "NovaGhostRig": NovaGhostRig,
    "NovaLensFilm": NovaLensFilm,
    "NovaBoxTrackMask": NovaBoxTrackMask,
    "NovaContactSheet": NovaContactSheet,
    "NovaMatchColor": NovaMatchColor,
    "NovaShutter": NovaShutter,
    "NovaShotQC": NovaShotQC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NovaGhostRig": "Nova Ghost Rig (camera carrier)",
    "NovaLensFilm": "Nova Lens & Film",
    "NovaBoxTrackMask": "Nova Box Track Mask",
    "NovaContactSheet": "Nova Contact Sheet",
    "NovaMatchColor": "Nova Match Colour (continuity)",
    "NovaShutter": "Nova Shutter (motion blur & speed)",
    "NovaShotQC": "Nova Shot QC",
}
