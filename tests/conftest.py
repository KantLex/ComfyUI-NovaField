import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _pkg  # noqa: E402

_pkg.load()
