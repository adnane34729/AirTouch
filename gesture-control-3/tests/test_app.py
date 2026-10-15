"""End-to-end tests driving the real application pipeline.

These run the whole chain -- detector, smoothing, pose estimation, debouncing,
action dispatch, overlay -- with the synthetic detector, so no camera, screen or
hand is needed. This is the level at which the original script could not be
tested at all.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gesture_control.app import GestureApp, ReadoutCache
from gesture_control.config import default_config
from gesture_control.detector import SyntheticHandDetector, create_detector
from gesture_control.sources import TestPatternSource, create_source
from gesture_control.synthetic import GestureScript, make_hand
from gesture_control.types import FINGERS, Finger

try:
    import cv2  # noqa: F401  (capability probe)

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False


def make_config(**overrides):
    """A config suited to headless testing: no window, no real side effects."""
    config = default_config("classic")
    config.overlay.show_window = False
    config.actions.dry_run = True
    config.camera.width = 640
    config.camera.height = 360
    for key, value in overrides.items():
        target, _, field = key.partition("__")
        if field:
            setattr(getattr(config, target), field, value)
        else:
            setattr(config, target, value)
    return config


class _ScriptedApp:
    """Helper that builds an app driven by a fixed list of poses."""

    def __init__(self, poses, config=None, *, fps=30.0):
        self.config = config or make_config()
        self.poses = poses
        self.fps = fps

    def run(self):
        app = GestureApp(
            self.config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=_PoseListDetector(self.poses),
        )
        triggers = []
        step = 1.0 / self.fps
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        now = 0.0
        for _ in range(len(self.poses)):
            triggers.extend(app.process(frame, now))
            now += step
        app.shutdown()
        return triggers, app


class _PoseListDetector:
    """Returns one scripted pose per call, then no hand."""

    name = "test-list"

    def __init__(self, poses):
        self.poses = list(poses)
        self.index = 0
        self.stats = None

    def detect(self, frame, timestamp):
        from gesture_control.geometry import make_observation
        from gesture_control.types import Handedness

        if self.index >= len(self.poses):
            return []
        pose = self.poses[self.index]
        self.index += 1
        if pose is None:
            return []
        raw = make_hand(pose, thumb_out=0.0, aspect_ratio=16 / 9)
        return [
            make_observation(
                raw, aspect_ratio=16 / 9, handedness=Handedness.RIGHT,
                handedness_score=1.0, timestamp=timestamp,
            )
        ]

    def describe(self):
        return "test-list"

    def close(self):
        pass


def _hold(curls, frames):
    return [curls] * frames


class TestPipelineIntegration(unittest.TestCase):
    def test_single_gesture_fires_once_when_armed(self) -> None:
        """Exactly one *initial* trigger, then repeats because the action repeats."""
        index_only = {f: (0.0 if f is Finger.INDEX else 1.0) for f in FINGERS}
        triggers, _ = _ScriptedApp(_hold(index_only, 30)).run()  # 1 second
        initials = [t for t in triggers if not t.is_repeat]
        self.assertEqual([t.name for t in initials], ["brightness_up"])
        self.assertTrue(all(t.name == "brightness_up" for t in triggers))

    def test_a_short_hold_fires_exactly_once(self) -> None:
        """Shorter than the repeat delay: one step, not a burst."""
        index_only = {f: (0.0 if f is Finger.INDEX else 1.0) for f in FINGERS}
        triggers, _ = _ScriptedApp(_hold(index_only, 8)).run()  # ~0.27 s
        self.assertEqual([t.name for t in triggers], ["brightness_up"])

    def test_repeatable_gesture_repeats_while_held(self) -> None:
        index_only = {f: (0.0 if f is Finger.INDEX else 1.0) for f in FINGERS}
        triggers, _ = _ScriptedApp(_hold(index_only, 90)).run()  # 3 seconds
        initials = [t for t in triggers if not t.is_repeat]
        repeats = [t for t in triggers if t.is_repeat]
        self.assertEqual(len(initials), 1)
        self.assertGreater(len(repeats), 5, f"only {len(repeats)} repeats in 3 s")

    def test_non_repeatable_gesture_does_not_repeat(self) -> None:
        fist = {f: 1.0 for f in FINGERS}
        triggers, _ = _ScriptedApp(_hold(fist, 120)).run()
        self.assertEqual([t.name for t in triggers], ["space"])

    def test_gesture_sequence_fires_in_order(self) -> None:
        def shape(up):
            return {f: (0.0 if f in up else 1.0) for f in FINGERS}

        poses = (
            _hold(shape({Finger.INDEX}), 20)
            + _hold(shape({Finger.INDEX, Finger.MIDDLE}), 20)
            + _hold(shape({Finger.PINKY}), 20)
            + _hold(shape({Finger.RING, Finger.PINKY}), 20)
            + _hold(shape(set()), 20)
        )
        triggers, _ = _ScriptedApp(poses).run()
        initials = [t.name for t in triggers if not t.is_repeat]
        self.assertEqual(
            initials, ["brightness_up", "left", "volume_down", "volume_up", "space"]
        )

    def test_nothing_fires_without_a_hand(self) -> None:
        triggers, _ = _ScriptedApp([None] * 60).run()
        self.assertEqual(triggers, [])

    def test_open_hand_matches_nothing_in_classic(self) -> None:
        open_hand = {f: 0.0 for f in FINGERS}
        triggers, _ = _ScriptedApp(_hold(open_hand, 60)).run()
        self.assertEqual(triggers, [], "an open hand triggered an action in the classic preset")

    def test_a_single_interrupting_frame_does_not_cancel_a_hold(self) -> None:
        """Detection dropouts are normal; they must not break a held gesture."""
        index_only = {f: (0.0 if f is Finger.INDEX else 1.0) for f in FINGERS}
        poses = _hold(index_only, 20) + [None] + _hold(index_only, 20)
        triggers, _ = _ScriptedApp(poses).run()
        initials = [t for t in triggers if not t.is_repeat]
        self.assertEqual(len(initials), 1, f"a dropout re-triggered: {[t.name for t in triggers]}")

    def test_paused_app_fires_nothing_but_keeps_running(self) -> None:
        index_only = {f: (0.0 if f is Finger.INDEX else 1.0) for f in FINGERS}
        app_config = make_config()
        app = GestureApp(
            app_config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=_PoseListDetector(_hold(index_only, 60)),
        )
        app.paused = True
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        triggers = []
        for step in range(60):
            triggers.extend(app.process(frame, step / 30.0))
        app.shutdown()
        self.assertEqual(triggers, [], "a paused app fired an action")

    def test_dry_run_never_touches_the_system(self) -> None:
        index_only = {f: (0.0 if f is Finger.INDEX else 1.0) for f in FINGERS}
        config = make_config()
        config.actions.dry_run = True
        _, app = _ScriptedApp(_hold(index_only, 40), config=config).run()
        self.assertTrue(app.registry.dry_run)
        self.assertEqual(app.summary.actions_failed, 0)

    def test_hud_state_reflects_the_pipeline(self) -> None:
        index_only = {f: (0.0 if f is Finger.INDEX else 1.0) for f in FINGERS}
        app_config = make_config()
        app = GestureApp(
            app_config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=_PoseListDetector(_hold(index_only, 20)),
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for step in range(20):
            app.process(frame, step / 30.0)
        state = app.build_hud_state(0.7)
        app.shutdown()

        self.assertIsNotNone(state.pose)
        self.assertEqual(state.pose.extended_fingers, frozenset({Finger.INDEX}))
        self.assertIsNotNone(state.active)
        self.assertEqual(state.active.name, "brightness_up")
        self.assertTrue(state.dry_run)
        self.assertEqual(len(state.gesture_set), 5)

    def test_fps_estimate_is_reasonable(self) -> None:
        app_config = make_config()
        app = GestureApp(
            app_config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=_PoseListDetector([]),
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for step in range(31):
            app.process(frame, step / 30.0)
            app._update_fps(step / 30.0)
        app.shutdown()
        self.assertAlmostEqual(app._fps, 30.0, delta=2.0)


class TestSyntheticEndToEnd(unittest.TestCase):
    """The whole application against the scripted synthetic routine.

    A virtual clock is injected so the scripted timeline advances one frame per
    iteration. Without it the loop would be paced by the CPU, so a short run
    would cover only a fraction of the 7.8 second script and never reach most of
    the gestures.
    """

    @staticmethod
    def _virtual_clock(fps: float = 30.0):
        state = {"t": 0.0}

        def clock() -> float:
            value = state["t"]
            state["t"] += 1.0 / fps
            return value

        return clock

    def _build(self, config):
        return GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=SyntheticHandDetector(GestureScript(rotation_span=0.0)),
            clock=self._virtual_clock(),
        )

    def test_scripted_routine_drives_every_gesture(self) -> None:
        app = self._build(make_config())
        summary = app.run(max_frames=260)  # 8.7 virtual seconds: a full script loop
        self.assertGreater(summary.frames, 100)
        self.assertGreater(summary.hands_seen, 50)
        self.assertGreaterEqual(summary.triggers, 5, "the scripted routine fired too little")

    def test_every_classic_gesture_is_reached(self) -> None:
        app = self._build(make_config())
        fired: list[str] = []

        def collect(_frame, state):
            if state.active is not None and state.active.name not in fired:
                fired.append(state.active.name)

        app.run(max_frames=260, on_frame=collect)
        self.assertEqual(
            set(fired),
            {"space", "left", "volume_up", "brightness_up", "volume_down"},
            f"the script did not reach every gesture: {fired}",
        )

    def test_summary_counts_are_consistent(self) -> None:
        app = self._build(make_config())
        summary = app.run(max_frames=120)
        self.assertEqual(summary.triggers, summary.actions_ok + summary.actions_failed)
        self.assertGreater(summary.duration, 0.0)
        self.assertGreater(summary.mean_fps, 0.0)

    def test_hud_rendering_during_a_full_run(self) -> None:
        """The overlay must survive every state the pipeline can reach."""
        if not HAVE_CV2:
            self.skipTest("OpenCV is required")
        app = self._build(make_config())
        shapes = []

        def check(frame, _state):
            shapes.append(frame.shape)

        app.run(max_frames=120, on_frame=check)
        self.assertEqual(len(shapes), 120)
        self.assertTrue(all(shape == (180, 320, 3) for shape in shapes))


class TestFrameSaving(unittest.TestCase):
    def _app(self):
        return GestureApp(
            make_config(),
            source=TestPatternSource(160, 120, fps=0.0),
            detector=SyntheticHandDetector(GestureScript(rotation_span=0.0)),
            clock=TestSyntheticEndToEnd._virtual_clock(),
        )

    @unittest.skipUnless(HAVE_CV2, "OpenCV is required")
    def test_frames_are_written(self) -> None:
        app = self._app()
        with tempfile.TemporaryDirectory() as tmp:
            summary = app.run(max_frames=4, save_frames=tmp, save_every=1)
            written = list(Path(tmp).glob("*.png"))
            self.assertEqual(len(written), 4, f"expected 4 frames, found {len(written)}")
            self.assertEqual(len(summary.saved_frames), 4)
            self.assertGreater(written[0].stat().st_size, 0)

    @unittest.skipUnless(HAVE_CV2, "OpenCV is required")
    def test_save_frames_without_an_interval_saves_every_frame(self) -> None:
        """Asking to save frames and getting none is never the intent."""
        app = self._app()
        with tempfile.TemporaryDirectory() as tmp:
            summary = app.run(max_frames=5, save_frames=tmp)
            self.assertEqual(len(summary.saved_frames), 5)

    @unittest.skipUnless(HAVE_CV2, "OpenCV is required")
    def test_rapid_saves_do_not_overwrite_each_other(self) -> None:
        """A millisecond timestamp is not unique when frames save faster than that."""
        app = self._app()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            paths = {app.save_frame(frame, tmp) for _ in range(12)}
            app.shutdown()
        self.assertNotIn(None, paths)
        self.assertEqual(len(paths), 12, "filenames collided within the same millisecond")

    @unittest.skipUnless(HAVE_CV2, "OpenCV is required")
    def test_save_frame_creates_missing_directories(self) -> None:
        app = self._app()
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "a" / "b" / "c"
            written = app.save_frame(np.zeros((8, 8, 3), dtype=np.uint8), nested)
            app.shutdown()
        self.assertIsNotNone(written)

    @unittest.skipUnless(HAVE_CV2, "OpenCV is required")
    def test_saving_nothing_is_a_no_op(self) -> None:
        app = self._app()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(app.save_frame(None, tmp))
            app.shutdown()


class TestReadoutCache(unittest.TestCase):
    class _Backend:
        def __init__(self):
            self.available = True
            self.reads = 0
            self.level = 0.5

        def get_percent(self):
            self.reads += 1
            return self.level

    def _registry_with(self, backend):
        from gesture_control.actions.base import Action, ActionOutcome, ActionRegistry

        class Stub(Action):
            id = "volume_up"

            def invoke(self) -> ActionOutcome:
                return ActionOutcome(ok=True)

        stub = Stub()
        stub.backend = backend
        return ActionRegistry([stub], async_actions=False)

    def test_reads_are_throttled(self) -> None:
        """Querying volume every frame would be a measurable performance bug."""
        backend = self._Backend()
        registry = self._registry_with(backend)
        cache = ReadoutCache(interval=1.0)
        cache.prime(registry)
        started = backend.reads
        for step in range(100):
            cache.maybe_refresh(registry, step * 0.01)  # 1 second of frames
        self.assertLessEqual(backend.reads - started, 2, "the cache did not throttle reads")

    def test_outcome_updates_immediately(self) -> None:
        backend = self._Backend()
        registry = self._registry_with(backend)
        cache = ReadoutCache(interval=100.0)
        cache.prime(registry)
        cache.note_outcome("volume_up", 0.75)
        self.assertAlmostEqual(cache.volume, 0.75, places=6)

    def test_unavailable_backend_reads_none(self) -> None:
        backend = self._Backend()
        backend.available = False
        registry = self._registry_with(backend)
        cache = ReadoutCache()
        cache.prime(registry)
        self.assertIsNone(cache.volume)


class TestSourceFactory(unittest.TestCase):
    def test_test_pattern_source(self) -> None:
        source = create_source("test", width=64, height=48)
        source.open()
        ok, frame = source.read()
        source.close()
        self.assertTrue(ok)
        self.assertEqual(frame.shape[:2], (48, 64))

    def test_unknown_source_rejected(self) -> None:
        from gesture_control.sources import SourceError

        with self.assertRaises(SourceError):
            create_source("nonsense")

    def test_file_source_needs_a_path(self) -> None:
        from gesture_control.sources import SourceError

        with self.assertRaises(SourceError):
            create_source("file")

    def test_missing_video_file_reported(self) -> None:
        from gesture_control.sources import SourceError

        source = create_source("file", path=Path("definitely-missing.mp4"))
        with self.assertRaises(SourceError):
            source.open()


class TestDetectorFactory(unittest.TestCase):
    def test_synthetic_detector_needs_no_dependencies(self) -> None:
        # The default script opens with the hand absent, so step past that.
        detector = create_detector("synthetic")
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame, 0.0), [], "the script starts with no hand")
        observations = detector.detect(frame, 1.0)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].points.shape, (21, 2))
        detector.close()

    def test_synthetic_detector_matches_its_script(self) -> None:
        from gesture_control.synthetic import ScriptedKeyframe, extended_curls

        always_open = GestureScript(
            (ScriptedKeyframe(name="open", duration=1.0, curls=extended_curls(), thumb_out=1.0),)
        )
        detector = SyntheticHandDetector(always_open)
        observations = detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), 0.0)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].handedness.value, "Right")
        self.assertEqual(observations[0].handedness_score, 1.0)
        detector.close()

    def test_unknown_detector_rejected(self) -> None:
        from gesture_control.detector import DetectorError

        with self.assertRaises(DetectorError):
            create_detector("nonsense")

    def test_closed_detector_refuses(self) -> None:
        from gesture_control.detector import DetectorError

        detector = create_detector("synthetic")
        detector.close()
        with self.assertRaises(DetectorError):
            detector.detect(np.zeros((8, 8, 3), dtype=np.uint8), 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
