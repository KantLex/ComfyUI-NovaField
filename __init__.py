"""ComfyUI-NovaField: camera-motion carriers, film looks and track masks for NovaField."""

if __package__:  # imported as a package (ComfyUI custom_nodes loader, tests/_pkg.py)
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # imported as a bare file (e.g. pytest walking the rootdir): nothing to register
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
