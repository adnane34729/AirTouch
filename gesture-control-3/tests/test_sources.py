"""Tests for frame sources.

The webcam cannot be exercised in CI, but its *pacing contract* can: ``read()``
must return each new frame once and block rather than spinning when nothing new
has arrived. That contract is what stops the main loop from re-detecting the same
image hundreds of times a second, so it is worth testing directly by driving the
source's internal state rather than a real device.
"""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from gesture_control.config import CameraConfig
from gesture_control.sources import (
    CameraSource,
    FrameSource,
    SourceError,
    TestPatternSource,
    VideoFileSource,
    create_source,
)

try:
    import cv2  # noqa: F401  (capability probe)

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False


class TestCameraPacing(unittest.TestCase):
    """Drives ``CameraSource`` without a camera by publishing frames directly."""

    def setUp(self) -> None:
        self.source = CameraSource(CameraConfig(reopen_attempts=1))
        self.source.actual_fps = 30.0
        self.source._frame = np.full((4, 4, 3), 1, dtype=np.uint8)
        self.source._serial = 1

    def tearDown(self) -> None:
        self.source.close()

    def _publish(self, value: int) -> None:
        with self.source._condition:
            self.source._frame = np.full((4, 4, 3), value, dtype=np.uint8)
            self.source._serial += 1
            self.source._condition.notify_all()

    def test_first_read_returns_the_frame(self) -> None:
        ok, frame = self.source.read()
        self.assertTrue(ok)
        self.assertEqual(int(frame[0, 0, 0]), 1)

    def test_a_frame_is_not_returned_twice(self) -> None:
        """The same frame must not be handed out again without new data.

        Without this the loop re-processes an identical image as fast as the CPU
        allows, which burns power for nothing.
        """
        self.assertTrue(self.source.read()[0])
        started = time.perf_counter()
        ok, _ = self.source.read()
        elapsed = time.perf_counter() - started
        self.assertFalse(ok, "a stale frame was returned")
        self.assertGreater(elapsed, 0.05, "read did not wait for a new frame")

    def test_a_new_frame_is_returned(self) -> None:
        self.assertTrue(self.source.read()[0])
        self._publish(7)
        ok, frame = self.source.read()
        self.assertTrue(ok)
        self.assertEqual(int(frame[0, 0, 0]), 7)

    def test_read_blocks_until_a_frame_is_published(self) -> None:
        self.assertTrue(self.source.read()[0])

        def publish_soon() -> None:
            time.sleep(0.05)
            self._publish(9)

        thread = threading.Thread(target=publish_soon)
        thread.start()
        ok, frame = self.source.read()
        thread.join(timeout=2.0)
        self.assertTrue(ok, "read timed out instead of waiting for the published frame")
        self.assertEqual(int(frame[0, 0, 0]), 9)

    def test_close_wakes_a_blocked_reader(self) -> None:
        self.assertTrue(self.source.read()[0])
        result: list[bool] = []

        def reader() -> None:
            result.append(self.source.read()[0])

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        time.sleep(0.02)
        self.source.close()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "close() left a reader blocked")

    def test_missing_frame_reports_failure(self) -> None:
        self.source._frame = None
        self.source.actual_fps = 1000.0  # keep the wait short
        ok, frame = self.source.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)

    def test_serial_advances_with_publications(self) -> None:
        self.assertEqual(self.source.serial, 1)
        self._publish(2)
        self.assertEqual(self.source.serial, 2)


class TestSourceFactory(unittest.TestCase):
    def test_camera_source_is_the_default(self) -> None:
        source = create_source("camera", camera=CameraConfig(index=2))
        self.assertIsInstance(source, CameraSource)
        self.assertIn("2", source.description)

    def test_test_pattern(self) -> None:
        source = create_source("test", width=48, height=32)
        self.assertIsInstance(source, TestPatternSource)
        self.assertEqual((source.width, source.height), (48, 32))

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(SourceError):
            create_source("magic")

    def test_file_source_requires_a_path(self) -> None:
        with self.assertRaises(SourceError):
            create_source("file")

    def test_context_manager_opens_and_closes(self) -> None:
        with create_source("test", width=16, height=16) as source:
            ok, frame = source.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (16, 16))

    def test_reopen_reports_failure_rather_than_raising(self) -> None:
        """A failed recovery must be a return value, not an exception."""
        source = create_source("file", path="missing-file.mp4")
        self.assertFalse(source.reopen())


@unittest.skipUnless(HAVE_CV2, "OpenCV is required")
class TestVideoFileSource(unittest.TestCase):
    def _write_video(self, path, frames: int = 12, size=(64, 48), fps: float = 60.0) -> None:
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        if not writer.isOpened():  # pragma: no cover - codec availability
            self.skipTest("no MJPG encoder available")
        for index in range(frames):
            writer.write(np.full((size[1], size[0], 3), index * 10, dtype=np.uint8))
        writer.release()

    def test_frames_are_read_then_looped(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self._write_video(path)
            source = VideoFileSource(path, loop=True, pace=False)
            source.open()
            try:
                seen = 0
                for _ in range(30):
                    ok, _frame = source.read()
                    if ok:
                        seen += 1
                self.assertGreaterEqual(seen, 12, "the clip did not loop")
            finally:
                source.close()

    def test_non_looping_source_ends(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self._write_video(path, frames=5)
            source = VideoFileSource(path, loop=False, pace=False)
            source.open()
            try:
                count = sum(1 for _ in range(20) if source.read()[0])
                self.assertLessEqual(count, 6)
            finally:
                source.close()

    def test_pacing_matches_the_file_rate(self) -> None:
        """A clip replayed at its own rate behaves like a camera."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.avi"
            self._write_video(path, frames=20, fps=100.0)
            source = VideoFileSource(path, loop=False, pace=True)
            source.open()
            try:
                started = time.perf_counter()
                count = 0
                for _ in range(10):
                    if source.read()[0]:
                        count += 1
                elapsed = time.perf_counter() - started
            finally:
                source.close()
        if count >= 5:
            self.assertGreater(elapsed, 0.03, "pacing did not slow the replay down")


class TestFrameSourceBase(unittest.TestCase):
    def test_close_is_safe_before_open(self) -> None:
        FrameSource.close(TestPatternSource(8, 8))
        FrameSource.close(TestPatternSource(8, 8))

    def test_test_pattern_frames_advance(self) -> None:
        source = TestPatternSource(32, 32, fps=0.0)
        source.open()
        first = source.read()[1].copy()
        second = source.read()[1].copy()
        source.close()
        self.assertFalse(np.array_equal(first, second), "the pattern is static")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
