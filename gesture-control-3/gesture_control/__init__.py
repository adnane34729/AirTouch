"""Real-time hand-gesture control for volume, brightness, and media keys.

A refactored, hardened rewrite of the original single-file ``main.py`` script.

The package is layered so that everything except :mod:`gesture_control.detector`
and :mod:`gesture_control.overlay` is pure logic that can be unit tested without
a camera, a screen, or a hand::

    geometry  ->  pose  ->  gestures  ->  tracking  ->  actions
                        (pure, deterministic, tested)

Public entry points::

    python -m gesture_control          # run the controller
    python -m gesture_control --doctor # check this machine's capabilities
"""

from __future__ import annotations

__all__ = ["__version__", "MIN_PYTHON", "__version_tuple__"]

__version__ = "2.0.0"

#: MediaPipe does not publish wheels for Python 3.13+ yet, so the supported
#: window is deliberately explicit rather than implied.
MIN_PYTHON: tuple[int, int] = (3, 9)
MAX_PYTHON: tuple[int, int] = (3, 12)

__version_tuple__ = tuple(int(part) for part in __version__.split("."))
