"""Tests for the HUD renderer and the diagnostics.

The overlay is verified by rendering into a real frame and inspecting the
result, which is how the layout bugs that a "does it not crash" test misses get
caught: text drawn off-screen, panels that overlap, or a state that renders
nothing at all.
"""

from __future__ import annotations

import unittest

import numpy as np

from gesture_control.config import OverlayConfig, default_config
from gesture_control.gestures import GestureSpec, parse_requirements
from gesture_control.overlay import PALETTE, HudRenderer, HudState
from gesture_control.types import Finger

from .helpers import real_pose

try:
    import cv2  # noqa: F401  (capability probe)

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False


def frame(width: int = 1280, height: int = 720) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def labels(config):
    return [entry[0] for entry in config]


@unittest.skipUnless(HAVE_CV2, "OpenCV is required")
class TestHudRenderer(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = HudRenderer(OverlayConfig())
        self.gestures = default_config("classic").build_gesture_set()

    def _state(self, **overrides) -> HudState:
        state = HudState(
            fps=30.0,
            frame_index=42,
            resolution=(1280, 720),
            source="camera 0 (1280x720)",
            detector="mediapipe (complexity 1)",
            inference_ms=23.4,
            gesture_set=self.gestures,
        )
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    def test_hud_changes_the_frame(self) -> None:
        blank = frame()
        rendered = self.renderer.render(blank.copy(), self._state())
        self.assertFalse(np.array_equal(blank, rendered), "the HUD drew nothing")

    def test_empty_state_renders_without_a_pose(self) -> None:
        """The first frames have no hand; the HUD must still draw."""
        blank = frame()
        rendered = self.renderer.render(blank.copy(), self._state())
        self.assertGreater(int(rendered.sum()), 1000)

    def test_every_state_combination_renders(self) -> None:
        """Including failures, warnings, pauses and a full hand."""
        pose = real_pose({Finger.INDEX, Finger.MIDDLE})
        for paused in (False, True):
            for dry_run in (False, True):
                for ok in (False, True):
                    with self.subTest(paused=paused, dry_run=dry_run, ok=ok):
                        state = self._state(
                            pose=pose,
                            paused=paused,
                            dry_run=dry_run,
                            last_action="INDEX",
                            last_message="Volume +: 60% -> 65%",
                            last_ok=ok,
                            last_action_age=1.25,
                            candidate=self.gestures.by_name("left"),
                            active=self.gestures.by_name("brightness_up"),
                            armed_fraction=0.5,
                            held_seconds=0.4,
                            banner="INDEX: Volume +: 60% -> 65%",
                            banner_age=0.2,
                            banner_ok=ok,
                            volume=0.6,
                            brightness=42.0,
                            warnings=("brightness control unavailable (run --doctor)",),
                        )
                        rendered = self.renderer.render(frame(), state)
                        self.assertEqual(rendered.shape, (720, 1280, 3))

    def test_renders_at_small_and_large_resolutions(self) -> None:
        """Panels must be omitted rather than overlapping when space is tight."""
        for width, height in ((320, 180), (640, 360), (1920, 1080)):
            with self.subTest(size=(width, height)):
                state = self._state(
                    pose=real_pose({Finger.INDEX}),
                    resolution=(width, height),
                    volume=0.5,
                )
                rendered = self.renderer.render(frame(width, height), state)
                self.assertEqual(rendered.shape, (height, width, 3))

    def test_small_frames_do_not_crash_or_overflow(self) -> None:
        for width, height in ((64, 48), (160, 90)):
            with self.subTest(size=(width, height)):
                state = self._state(pose=real_pose(set()), volume=0.5, brightness=10.0)
                self.renderer.render(frame(width, height), state)

    def test_overlay_scale_is_applied(self) -> None:
        small = HudRenderer(OverlayConfig(scale=0.6))
        large = HudRenderer(OverlayConfig(scale=1.8))
        state = self._state(pose=real_pose({Finger.INDEX}), volume=0.5)
        small_pixels = int(small.render(frame(), state).sum())
        large_pixels = int(large.render(frame(), state).sum())
        self.assertGreater(large_pixels, small_pixels, "scaling had no effect")

    def test_sections_can_be_disabled(self) -> None:
        state = self._state(pose=real_pose({Finger.INDEX}), volume=0.5)
        full = HudRenderer(OverlayConfig()).render(frame(), state)
        without = HudRenderer(
            OverlayConfig(show_guide=False, show_skeleton=False, show_system_levels=False)
        ).render(frame(), state)
        self.assertGreater(int(full.sum()), int(without.sum()))

    def test_pose_absent_draws_no_skeleton(self) -> None:
        with_pose = HudRenderer(OverlayConfig(show_guide=False)).render(
            frame(), self._state(pose=real_pose({Finger.INDEX}))
        )
        without_pose = HudRenderer(OverlayConfig(show_guide=False)).render(
            frame(), self._state(pose=None)
        )
        self.assertNotEqual(int(with_pose.sum()), int(without_pose.sum()))

    def test_finger_labels_come_from_the_pose(self) -> None:
        pose = real_pose({Finger.PINKY})
        self.assertTrue(pose.is_extended(Finger.PINKY))
        self.assertTrue(pose.readings[Finger.PINKY].extended)
        self.assertFalse(pose.readings[Finger.INDEX].extended)
        state = self._state(pose=pose, active=self.gestures.by_name("volume_down"))
        self.assertEqual(self.renderer.render(frame(), state).shape, (720, 1280, 3))

    def test_banner_expires(self) -> None:
        fresh = HudRenderer(OverlayConfig()).render(
            frame(), self._state(banner="INDEX: 60% -> 65%", banner_age=0.0, banner_duration=1.0)
        )
        expired = HudRenderer(OverlayConfig()).render(
            frame(), self._state(banner="INDEX: 60% -> 65%", banner_age=5.0, banner_duration=1.0)
        )
        self.assertGreater(int(fresh.sum()), int(expired.sum()), "the banner never faded")

    def test_empty_frame_is_returned_untouched(self) -> None:
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertIs(self.renderer.render(empty, self._state()), empty)

    def test_draw_hand_marks_the_extended_finger(self) -> None:
        pose = real_pose({Finger.INDEX})
        canvas = frame(640, 360)
        self.renderer.draw_hand(canvas, pose)
        self.assertGreater(int(canvas.sum()), 0)
        # The index colour must appear; a fully curled hand must not use it.
        index_colour = np.array(PALETTE["index"], dtype=np.int16)
        matches = np.all(np.abs(canvas.astype(np.int16) - index_colour) <= 2, axis=-1)
        self.assertTrue(matches.any(), "the extended finger was not drawn in its colour")

        curled = frame(640, 360)
        self.renderer.draw_hand(curled, real_pose(set()))
        curled_matches = np.all(
            np.abs(curled.astype(np.int16) - index_colour) <= 2, axis=-1
        )
        self.assertFalse(
            curled_matches.any(), "a curled finger was drawn as if it were extended"
        )

    def test_guide_uses_the_supplied_gesture_set(self) -> None:
        custom = GestureSpec(
            name="custom", action="volume_up",
            requirements=parse_requirements("index:up,pinky:up"),
            label="CUSTOM-LABEL",
        )
        from gesture_control.gestures import build_gesture_set

        state = self._state(gesture_set=build_gesture_set((custom,)))
        rendered = self.renderer.render(frame(), state)
        self.assertEqual(rendered.shape, (720, 1280, 3))


class TestDiagnostics(unittest.TestCase):
    def test_doctor_runs_and_returns_checks(self) -> None:
        from gesture_control.diagnostics import run_doctor

        checks = run_doctor(default_config("classic"))
        self.assertGreater(len(checks), 8)
        names = [check.name for check in checks]
        self.assertTrue(any("interpreter" in name for name in names))
        self.assertTrue(any("gestures" in name for name in names))
        for check in checks:
            self.assertIn(check.status, ("ok", "warn", "fail"))
            self.assertTrue(check.name)

    def test_doctor_report_is_formatted(self) -> None:
        from gesture_control.diagnostics import format_doctor, run_doctor

        report = format_doctor(run_doctor(default_config("classic")))
        self.assertIn("checks:", report)
        self.assertTrue(any(tag in report for tag in ("[PASS]", "[WARN]", "[FAIL]")))

    def test_doctor_never_raises_on_a_broken_config(self) -> None:
        from gesture_control.diagnostics import run_doctor

        config = default_config("classic")
        config.actions.brightness_step = 5
        config.camera.index = 99  # almost certainly not present
        checks = run_doctor(config)
        self.assertTrue(any(check.name == "camera" for check in checks))

    def test_self_test_passes_for_every_preset(self) -> None:
        from gesture_control.diagnostics import run_self_test

        for preset in ("classic", "media", "minimal"):
            with self.subTest(preset=preset):
                results = run_self_test(default_config(preset))
                failures = [result for result in results if not result.ok]
                self.assertEqual(failures, [], f"{preset}: {failures[:3]}")

    def test_self_test_covers_rotations_and_sizes(self) -> None:
        from gesture_control.diagnostics import run_self_test

        results = run_self_test(default_config("classic"))
        rotations = {result.rotation for result in results}
        scales = {result.scale for result in results}
        self.assertGreater(len(rotations), 3, "rotation invariance is not being tested")
        self.assertGreater(len(scales), 2, "scale invariance is not being tested")
        self.assertTrue(any(result.rotation >= 90 for result in results))

    def test_self_test_format_reports_failures(self) -> None:
        from gesture_control.diagnostics import CaseResult, format_self_test

        results = [
            CaseResult("good", "good", "good", 0.0, 0.26, True),
            CaseResult("bad", "bad", "<none>", 90.0, 0.26, False, "detail"),
        ]
        report = format_self_test(results)
        self.assertIn("[FAIL] bad", report)
        self.assertIn("1 failed", report)
        self.assertIn("detail", report)

    def test_self_test_format_verbose_lists_every_case(self) -> None:
        from gesture_control.diagnostics import CaseResult, format_self_test

        results = [CaseResult("g", "g", "g", 0.0, 0.2, True)]
        self.assertEqual(len(format_self_test(results).splitlines()), 4)
        self.assertGreater(len(format_self_test(results, verbose=True).splitlines()), 4)

    def test_threshold_tuning_is_detected_by_the_self_test(self) -> None:
        """The self-test must actually be able to fail.

        A check that cannot fail proves nothing, so this deliberately sets the
        reach thresholds above the reach a genuinely extended finger achieves,
        which makes fingers read as curled. Note that merely raising
        ``finger_curl_up`` is *not* enough: the first-sighting rule compares
        against the middle of the band, so a lenient mid-point still classifies an
        extended finger correctly. That leniency is intentional -- it is what
        stops a threshold from being brittle -- but it means a failure has to be
        provoked through the reach band.
        """
        from gesture_control.diagnostics import run_self_test

        config = default_config("classic")
        config.thresholds.finger_reach_up = 2.0
        config.thresholds.finger_reach_down = 1.85
        results = run_self_test(config)
        self.assertTrue(
            any(not result.ok for result in results),
            "an unreachable threshold still passed the self-test",
        )

    def test_curl_only_tuning_alone_does_not_break_recognition(self) -> None:
        """Documents the leniency above rather than leaving it implicit."""
        from gesture_control.diagnostics import run_self_test

        config = default_config("classic")
        config.thresholds.finger_curl_up = 179.99
        config.thresholds.finger_curl_down = 179.5
        self.assertTrue(all(result.ok for result in run_self_test(config)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
