"""Hand detection backends.

The detector is the only component that knows MediaPipe exists. It converts
whatever the backend produces into :class:`~gesture_control.types.HandObservation`
objects in metric space, so nothing downstream depends on a third-party API.

Two backends ship:

``mediapipe``
    The real one, using the legacy ``mp.solutions.hands`` graph. That graph is
    chosen deliberately: the tasks-based ``HandLandmarker`` needs a ``.task``
    model file downloaded at runtime, whereas the legacy graph carries its
    models inside the wheel and therefore works offline with no extra step.

``synthetic``
    Drives the parametric hand model from :mod:`gesture_control.synthetic`
    instead of a camera, so the whole application -- smoothing, pose estimation,
    debouncing, actions, overlay -- can run and be tested end to end with no
    hardware. ``--self-test`` and ``--source synthetic`` both use it.

Environment notes verified against mediapipe 0.10.8:

* ``Hands.process`` requires a **numpy** RGB array in this version; passing an
  ``mp.Image`` raises ``AttributeError: ... has no attribute 'shape'``.
* Inference costs ~20-25 ms per frame on CPU and is essentially independent of
  input resolution (MediaPipe resizes internally to 224x224). Downscaling the
  frame therefore buys very little; skipping frames does.
"""

from __future__ import annotations

import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import DetectorConfig
from .geometry import make_observation
from .synthetic import GestureScript
from .types import HandObservation, Handedness

__all__ = [
    "DetectorError",
    "HandDetector",
    "MediaPipeHandDetector",
    "SyntheticHandDetector",
    "create_detector",
    "mediapipe_available",
    "INSTALL_HINT",
]

_LOG = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Raised when a detector cannot be created or used."""


def _install_hint() -> str:
    """A failure message that accounts for a machine with several Pythons.

    The most common way for this project to fail on Windows is not a missing
    package -- it is the *right* package installed in a *different* interpreter
    than the one running the script.
    """
    return (
        f"MediaPipe is not importable from this interpreter:\n"
        f"    {sys.executable}\n"
        f"Install the dependencies with:\n"
        f'    "{sys.executable}" -m pip install -r requirements.txt\n'
        f"If MediaPipe is already installed elsewhere, run the script with that "
        f"interpreter instead (see run.py / --doctor)."
    )


INSTALL_HINT = _install_hint()


def mediapipe_available() -> bool:
    """True when MediaPipe can be imported in this interpreter."""
    try:
        import mediapipe  # noqa: F401
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


def _quiet_mediapipe_logs() -> None:
    """Silence MediaPipe's one-off INFO chatter about TFLite delegates.

    It is emitted through absl/glog on first inference and is noise in a program
    whose own output matters. Best effort: any failure here is harmless.
    """
    for name in ("mediapipe", "tensorflow", "absl"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:  # pragma: no cover - depends on the absl version present
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class HandDetector(ABC):
    """Finds hands in a frame and returns them in metric space."""

    #: Short identifier used in logs and the HUD.
    name: str = "detector"

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float) -> list[HandObservation]:
        """Return every hand visible in ``frame`` (BGR, HxWx3)."""

    def close(self) -> None:
        """Release any resources. Safe to call more than once."""

    def describe(self) -> str:
        return self.name

    def __enter__(self) -> HandDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass
class DetectionStats:
    """Rolling detector statistics for the HUD."""

    frames: int = 0
    hits: int = 0
    last_inference_ms: float = 0.0
    mean_inference_ms: float = 0.0

    def record(self, elapsed_ms: float, found: bool) -> None:
        self.frames += 1
        if found:
            self.hits += 1
        self.last_inference_ms = elapsed_ms
        # Exponential mean keeps the HUD readable without a history buffer.
        weight = 0.1 if self.frames > 1 else 1.0
        self.mean_inference_ms = (
            elapsed_ms if self.frames == 1 else (1 - weight) * self.mean_inference_ms + weight * elapsed_ms
        )

    @property
    def hit_rate(self) -> float:
        return self.hits / self.frames if self.frames else 0.0


# --------------------------------------------------------------------------- #
# MediaPipe backend
# --------------------------------------------------------------------------- #


class MediaPipeHandDetector(HandDetector):
    """MediaPipe's legacy ``solutions.hands`` graph."""

    name = "mediapipe"

    def __init__(self, config: DetectorConfig | None = None, *, suppress_logs: bool = True) -> None:
        self.config = config or DetectorConfig()
        self.stats = DetectionStats()
        self._closed = False
        self._scale = 1.0

        if suppress_logs:
            _quiet_mediapipe_logs()

        try:
            import mediapipe as mp
        except Exception as exc:  # pragma: no cover - environment dependent
            raise DetectorError(INSTALL_HINT) from exc

        self._mp = mp
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=self.config.static_image_mode,
                max_num_hands=self.config.max_hands,
                model_complexity=self.config.model_complexity,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise DetectorError(f"could not initialise the MediaPipe hand graph: {exc}") from exc

        _LOG.debug(
            "MediaPipe hand graph ready (model_complexity=%d, max_hands=%d)",
            self.config.model_complexity,
            self.config.max_hands,
        )

    def describe(self) -> str:
        return (
            f"mediapipe (complexity {self.config.model_complexity}, "
            f"det {self.config.min_detection_confidence:g}, "
            f"track {self.config.min_tracking_confidence:g})"
        )

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR to the RGB array this MediaPipe version expects."""
        import cv2

        width = self.config.downscale_width
        height, frame_width = frame.shape[:2]
        if width and frame_width > width:
            scale = width / float(frame_width)
            self._scale = scale
            frame = cv2.resize(
                frame, (int(round(frame_width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            self._scale = 1.0
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect(self, frame: np.ndarray, timestamp: float) -> list[HandObservation]:
        if self._closed:
            raise DetectorError("detector has been closed")
        if frame is None or frame.size == 0:
            return []

        started = time.perf_counter()
        rgb = self._prepare(frame)
        height, width = rgb.shape[:2]
        aspect_ratio = width / float(height) if height else 1.0

        try:
            results = self._hands.process(rgb)
        except Exception as exc:
            # A transient decode failure must not kill a running session.
            _LOG.warning("hand detection failed on one frame: %s", exc)
            self.stats.record((time.perf_counter() - started) * 1000.0, False)
            return []

        observations = self._convert(results, aspect_ratio, timestamp)
        self.stats.record((time.perf_counter() - started) * 1000.0, bool(observations))
        return observations

    def _convert(self, results: Any, aspect_ratio: float, timestamp: float) -> list[HandObservation]:
        landmarks = getattr(results, "multi_hand_landmarks", None)
        if not landmarks:
            return []

        handedness_lists = getattr(results, "multi_handedness", None) or []
        observations: list[HandObservation] = []
        for index, hand in enumerate(landmarks):
            handedness = Handedness.UNKNOWN
            score = 0.0
            if index < len(handedness_lists):
                classification = handedness_lists[index].classification
                if classification:
                    label = str(classification[0].label).capitalize()
                    handedness = Handedness(label) if label in ("Left", "Right") else Handedness.UNKNOWN
                    score = float(classification[0].score)

            raw = np.asarray(
                [[point.x, point.y, point.z] for point in hand.landmark], dtype=np.float64
            )
            try:
                observations.append(
                    make_observation(
                        raw,
                        aspect_ratio=aspect_ratio,
                        handedness=handedness,
                        handedness_score=score,
                        timestamp=timestamp,
                    )
                )
            except ValueError as exc:  # pragma: no cover - defensive
                _LOG.warning("discarding malformed hand detection: %s", exc)
        return observations

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._hands.close()
        except Exception:  # pragma: no cover - defensive
            pass


# --------------------------------------------------------------------------- #
# Synthetic backend
# --------------------------------------------------------------------------- #


class SyntheticHandDetector(HandDetector):
    """Replays a scripted hand routine instead of looking at the frame.

    Produces exactly the same :class:`HandObservation` objects as the MediaPipe
    backend, so everything downstream cannot tell the difference. Landmarks are
    generated through the shared hand model, which means the recogniser is
    exercised against geometry it has never been tuned on frame by frame --
    including a slow rotation sweep of +/- 30 degrees.
    """

    name = "synthetic"

    def __init__(
        self,
        script: GestureScript | None = None,
        *,
        speed: float = 1.0,
        aspect_ratio: float = 16 / 9,
    ) -> None:
        self.script = script or GestureScript()
        self.speed = speed
        self.aspect_ratio = aspect_ratio
        self.stats = DetectionStats()
        self._origin: float | None = None
        self._closed = False

    def describe(self) -> str:
        return f"synthetic ({len(self.script.keyframes)} poses, {self.script.duration:.1f}s loop)"

    @property
    def elapsed(self) -> float:
        return 0.0 if self._origin is None else (time.time() - self._origin) * self.speed

    def detect(self, frame: np.ndarray, timestamp: float) -> list[HandObservation]:
        if self._closed:
            raise DetectorError("detector has been closed")
        if self._origin is None:
            self._origin = timestamp

        elapsed = (timestamp - self._origin) * self.speed
        started = time.perf_counter()
        hands = self.script.hands_at(elapsed)
        if not hands:
            self.stats.record((time.perf_counter() - started) * 1000.0, False)
            return []

        # Every hand the script shows, so two-handed poses are exercised exactly
        # like one-handed ones -- including the handedness labels, which the
        # symmetric two-hand gestures must not depend on.
        observations = [
            make_observation(
                self.script.landmarks_for(hand, elapsed, index),
                aspect_ratio=self.aspect_ratio,
                handedness=hand.handedness,
                handedness_score=1.0,
                timestamp=timestamp,
            )
            for index, hand in enumerate(hands)
        ]
        self.stats.record((time.perf_counter() - started) * 1000.0, True)
        return observations

    def close(self) -> None:
        self._closed = True


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def create_detector(kind: str = "auto", config: DetectorConfig | None = None) -> HandDetector:
    """Build a detector.

    Args:
        kind: ``"auto"`` (MediaPipe if available, else a clear error),
            ``"mediapipe"``, or ``"synthetic"``.
        config: detector settings, used by the MediaPipe backend.
    """
    normalised = (kind or "auto").strip().lower()
    if normalised == "synthetic":
        return SyntheticHandDetector()
    if normalised in ("auto", "mediapipe"):
        if not mediapipe_available():
            raise DetectorError(INSTALL_HINT)
        return MediaPipeHandDetector(config)
    raise DetectorError(f"unknown detector {kind!r}; expected auto, mediapipe or synthetic")
