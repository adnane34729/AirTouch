"""Frame sources: webcam, video file, and a generated test pattern.

Why the camera source runs a reader thread
------------------------------------------
The obvious loop is ``read() -> detect() -> show()``, and it is what the original
did. It has a subtle problem: ``cap.read()`` returns the *oldest* buffered frame.
Because inference takes ~25 ms and the camera fills its buffer independently, the
loop falls behind and starts displaying stale frames -- latency grows without
bound and gestures feel disconnected from the image.

:class:`CameraSource` therefore reads in a background thread that keeps only the
most recent frame and discards the rest, so the image on screen is always the
newest one available and latency stays flat even when inference is slow.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .config import CameraConfig

__all__ = [
    "SourceError",
    "FrameSource",
    "CameraSource",
    "VideoFileSource",
    "TestPatternSource",
    "create_source",
]

_LOG = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """Raised when frames cannot be obtained."""


class FrameSource(ABC):
    """A stream of BGR frames."""

    #: Human-readable description for logs and the HUD.
    description: str = "source"

    @abstractmethod
    def open(self) -> None:
        """Acquire the source, raising :class:`SourceError` on failure."""

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return ``(ok, frame)``. ``ok=False`` means temporarily unavailable."""

    def close(self) -> None:
        """Release the source. Safe to call more than once."""

    def reopen(self) -> bool:
        """Try to recover from a dropped connection. Returns success."""
        self.close()
        try:
            self.open()
        except SourceError as exc:
            _LOG.warning("reopen failed: %s", exc)
            return False
        return True

    def __enter__(self) -> FrameSource:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class CameraSource(FrameSource):
    """A webcam, read on a background thread.

    :meth:`read` blocks until a frame *newer* than the last one handed out is
    available. That matters: a reader thread that simply returns the latest frame
    would let the main loop spin at hundreds of frames per second, re-detecting
    the same image over and over and burning CPU for no benefit. Waiting for a
    new frame restores the natural pacing of a blocking ``cap.read()`` while
    keeping the low latency the thread was added for.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._frame: np.ndarray | None = None
        self._serial = 0
        self._delivered = 0
        self._read_errors = 0
        self.description = f"camera {self.config.index}"
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> None:
        import cv2

        if self._capture is not None:
            return

        capture = cv2.VideoCapture(self.config.index)
        if not capture.isOpened():
            capture.release()
            raise SourceError(
                f"could not open camera {self.config.index}. "
                "Check that no other program is using it, that Windows camera "
                "privacy settings allow access, and try --camera 1."
            )

        if self.config.width and self.config.height:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps:
            capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if self.config.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc[:4]))

        # A large driver-side buffer is what causes stale frames; ask for one.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = capture
        self.actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not self.actual_width or not self.actual_height:
            # Some drivers report nothing until a frame has been read.
            ok, frame = capture.read()
            if ok and frame is not None:
                self.actual_height, self.actual_width = frame.shape[:2]
                with self._lock:
                    self._frame = frame
                    self._serial += 1

        self.description = (
            f"camera {self.config.index} "
            f"({self.actual_width}x{self.actual_height}"
            + (f" @ {self.actual_fps:.0f} fps" if self.actual_fps > 1 else "")
            + ")"
        )

        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()
        _LOG.info("opened %s", self.description)

    def _reader(self) -> None:
        """Continuously grab frames, publishing only the newest one."""
        while not self._stop.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                self._read_errors += 1
                if self._read_errors > 30:
                    _LOG.warning("camera read failed %d times in a row", self._read_errors)
                    self._stop.set()
                    with self._condition:
                        self._condition.notify_all()
                time.sleep(0.005)
                continue
            self._read_errors = 0
            with self._condition:
                self._frame = frame
                self._serial += 1
                self._condition.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Wait for and return the newest frame not yet handed out."""
        timeout = max(0.2, 3.0 / self.actual_fps) if self.actual_fps > 1 else 0.5
        with self._condition:
            if self._frame is None:
                self._condition.wait(timeout)
            if self._frame is None:
                return False, None
            if self._serial == self._delivered:
                # Nothing new yet; wait once more rather than returning a frame
                # that has already been processed.
                self._condition.wait(timeout)
            if self._serial == self._delivered:
                return False, None
            self._delivered = self._serial
            # Hand out the frame without copying: the reader replaces the
            # reference rather than mutating the array in place.
            return True, self._frame

    @property
    def serial(self) -> int:
        """Increments per captured frame; lets callers detect duplicate frames."""
        with self._lock:
            return self._serial

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
        with self._lock:
            self._frame = None


class VideoFileSource(FrameSource):
    """Frames from a video file, optionally looping.

    Mainly useful for reproducible testing: a recorded clip replays identically
    on every run, which a webcam cannot.
    """

    def __init__(self, path: str | Path, *, loop: bool = True, pace: bool = True) -> None:
        self.path = Path(path)
        self.loop = loop
        self.pace = pace
        self._capture = None
        self._frames = 0
        self._fps = 0.0
        self._next_frame_time = 0.0
        self.description = f"file {self.path.name}"

    def open(self) -> None:
        import cv2

        if not self.path.is_file():
            raise SourceError(f"video file not found: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise SourceError(f"could not open video file: {self.path}")
        self._capture = capture
        self._fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._next_frame_time = time.monotonic()
        self.description = (
            f"file {self.path.name} "
            f"({int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
            f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))})"
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._capture is None:
            return False, None
        if self.pace and 1.0 < self._fps < 240.0:
            # Replay at roughly the recorded rate so a file behaves like a camera.
            interval = 1.0 / self._fps
            now = time.monotonic()
            sleep_for = self._next_frame_time - now
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.5))
            self._next_frame_time = max(now, self._next_frame_time) + interval
        ok, frame = self._capture.read()
        if ok and frame is not None:
            self._frames += 1
            return True, frame
        if self.loop and self._frames:
            import cv2

            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
            if ok and frame is not None:
                self._frames += 1
                return True, frame
        return False, None

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()


class TestPatternSource(FrameSource):
    """A generated moving pattern, paced to a frame rate, for headless runs.

    Gives the overlay something real to draw over so that
    ``--no-window --save-frames`` produces a meaningful screenshot without a
    camera, and so the display path is exercised during automated runs. Pacing
    matters: without it an automated run would spin as fast as the CPU allows and
    the synthetic hand's scripted timeline (which is driven by wall-clock time)
    would advance far slower than the frames it is meant to accompany.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 30.0) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._index = 0
        self._next_frame_time = 0.0
        self.description = f"test pattern ({width}x{height} @ {fps:.0f} fps)"

    def open(self) -> None:
        self._index = 0
        self._next_frame_time = time.monotonic()

    def read(self) -> tuple[bool, np.ndarray | None]:
        import cv2

        if self.fps > 0:
            now = time.monotonic()
            sleep_for = self._next_frame_time - now
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.5))
            self._next_frame_time = max(now, self._next_frame_time) + 1.0 / self.fps

        self._index += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # A slow vertical gradient plus a moving band: enough structure to judge
        # text contrast and to prove frames are advancing.
        ramp = np.linspace(20, 70, self.height, dtype=np.uint8).reshape(-1, 1, 1)
        frame[:] = ramp
        offset = (self._index * 6) % max(1, self.width)
        cv2.rectangle(frame, (offset, 0), (min(offset + 90, self.width), self.height), (40, 40, 90), -1)
        cv2.putText(
            frame,
            "test pattern",
            (self.width // 2 - 90, self.height - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (120, 120, 140),
            2,
            cv2.LINE_AA,
        )
        return True, frame


def create_source(
    kind: str = "camera",
    *,
    camera: CameraConfig | None = None,
    path: str | Path | None = None,
    width: int = 1280,
    height: int = 720,
) -> FrameSource:
    """Build a frame source.

    Args:
        kind: ``"camera"``, ``"synthetic"``/``"test"``, or ``"file"``.
        camera: camera settings, used when ``kind == "camera"``.
        path: video path, required when ``kind == "file"``.
    """
    normalised = (kind or "camera").strip().lower()
    if normalised == "camera":
        return CameraSource(camera)
    if normalised in ("synthetic", "test", "test-pattern", "pattern"):
        return TestPatternSource(width, height)
    if normalised in ("file", "video"):
        if path is None:
            raise SourceError("a video file path is required for --source file")
        return VideoFileSource(path)
    raise SourceError(f"unknown source {kind!r}; expected camera, test or file")
