"""Tests for the geometry layer.

These tests are the load-bearing argument for the rewrite: they demonstrate that
the recognition primitives are invariant under rotation, scale and camera aspect
ratio, and they quantify the failure of the original's approach.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from gesture_control import geometry
from gesture_control.synthetic import curled_curls, extended_curls, make_hand
from gesture_control.types import (
    FINGERS,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    WRIST,
    Finger,
)

from .helpers import ASPECTS, ROTATIONS, SCALES


class TestToMetric(unittest.TestCase):
    def test_scales_x_by_aspect_ratio(self) -> None:
        raw = np.zeros((NUM_LANDMARKS, 3))
        raw[:, 0] = 0.5
        raw[:, 1] = 0.25
        metric = geometry.to_metric(raw, 2.0)
        np.testing.assert_allclose(metric[:, 0], 1.0)
        np.testing.assert_allclose(metric[:, 1], 0.25)

    def test_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            geometry.to_metric(np.zeros((3, 3)))

    def test_rejects_implausible_aspect(self) -> None:
        raw = np.zeros((NUM_LANDMARKS, 3))
        for bad in (0.0, -1.0, 1000.0):
            with self.subTest(aspect=bad), self.assertRaises(ValueError):
                geometry.to_metric(raw, bad)


class TestInvariance(unittest.TestCase):
    """The headline property: hand-relative measurements do not move when the
    hand rotates, changes distance, or the camera aspect changes."""

    def _measure(self, curls, thumb_out, rotation, scale, aspect) -> dict[str, float]:
        raw = make_hand(
            curls, thumb_out=thumb_out, rotation_deg=rotation, scale=scale, aspect_ratio=aspect
        )
        points = geometry.make_observation(raw, aspect_ratio=aspect).points
        values = {
            f"curl_{f.value}": geometry.finger_curl(points, f) for f in FINGERS
        }
        for f in FINGERS:
            values[f"reach_{f.value}"] = geometry.finger_reach(points, f)
        values["palm_size"] = geometry.palm_size(points)
        return values

    #: Tolerances for the invariance comparison. Angles come out of ``arccos``
    #: and reach out of a dot product, so exact equality is not achievable in
    #: floating point; what matters is that the deviation is nothing like the
    #: tens of degrees the uncorrected computation moves by (see
    #: :class:`TestAspectRatioBug`). These bounds are ~1e5 times tighter than
    #: that failure mode.
    TOLERANCES = {"curl_": 1e-3, "reach_": 1e-9}

    def _assert_invariant(self, baseline, measured, context: str) -> None:
        for key, expected in baseline.items():
            if key == "palm_size":
                continue  # scale-dependent by construction
            tolerance = next(
                (value for prefix, value in self.TOLERANCES.items() if key.startswith(prefix)),
                1e-9,
            )
            self.assertAlmostEqual(
                measured[key],
                expected,
                delta=tolerance,
                msg=f"{key} moved by more than {tolerance} under {context}",
            )

    def test_extended_hand_is_invariant(self) -> None:
        baseline = self._measure(extended_curls(), 1.0, 0.0, 0.26, 16 / 9)
        for rotation in ROTATIONS:
            for scale in SCALES:
                for aspect in ASPECTS:
                    with self.subTest(rotation=rotation, scale=scale, aspect=aspect):
                        measured = self._measure(extended_curls(), 1.0, rotation, scale, aspect)
                        self._assert_invariant(
                            baseline,
                            measured,
                            f"rotation={rotation} scale={scale} aspect={aspect}",
                        )

    def test_curled_hand_is_invariant(self) -> None:
        baseline = self._measure(curled_curls(), 0.0, 0.0, 0.26, 16 / 9)
        for rotation in ROTATIONS:
            for scale in SCALES:
                with self.subTest(rotation=rotation, scale=scale):
                    measured = self._measure(curled_curls(), 0.0, rotation, scale, 16 / 9)
                    self._assert_invariant(
                        baseline, measured, f"rotation={rotation} scale={scale}"
                    )

    def test_palm_size_scales_linearly_with_hand_size(self) -> None:
        sizes = [
            geometry.palm_size(
                geometry.make_observation(
                    make_hand(extended_curls(), scale=s), aspect_ratio=1.0
                ).points
            )
            for s in (0.1, 0.2, 0.4)
        ]
        self.assertAlmostEqual(sizes[1] / sizes[0], 2.0, places=6)
        self.assertAlmostEqual(sizes[2] / sizes[0], 4.0, places=6)


class TestAspectRatioBug(unittest.TestCase):
    """Quantifies the original's aspect-ratio blindness.

    The original compared raw normalised landmarks, which is equivalent to
    running this module with ``aspect_ratio=1``. This test asserts that doing so
    produces a materially wrong angle, and that the corrected value does not.
    """

    def test_naive_angles_are_skewed_by_aspect_ratio(self) -> None:
        corrected_seen: set[float] = set()
        naive_values: list[float] = []

        for aspect in ASPECTS:
            for rotation in (0.0, 30.0):
                raw = make_hand(
                    curled_curls(), thumb_out=0.0, rotation_deg=rotation, aspect_ratio=aspect
                )
                corrected = geometry.finger_curl(
                    geometry.make_observation(raw, aspect_ratio=aspect).points, Finger.INDEX
                )
                naive = geometry.finger_curl(
                    geometry.make_observation(raw, aspect_ratio=1.0).points, Finger.INDEX
                )
                corrected_seen.add(round(corrected, 3))
                naive_values.append(naive)

        # The corrected angle is one value regardless of aspect ratio.
        self.assertEqual(len(corrected_seen), 1, f"corrected angles varied: {corrected_seen}")
        # The naive angle is not.
        spread = max(naive_values) - min(naive_values)
        self.assertGreater(
            spread,
            30.0,
            "expected the uncorrected angle to vary widely with aspect ratio; "
            f"observed spread {spread:.1f} deg from {naive_values}",
        )


class TestJointAngle(unittest.TestCase):
    def test_right_angle(self) -> None:
        points = np.zeros((NUM_LANDMARKS, 2))
        points[0] = (1.0, 0.0)
        points[1] = (0.0, 0.0)
        points[2] = (0.0, 1.0)
        self.assertAlmostEqual(geometry.joint_angle(points, 0, 1, 2), 90.0, places=6)

    def test_straight_line(self) -> None:
        points = np.zeros((NUM_LANDMARKS, 2))
        points[0] = (0.0, 1.0)
        points[1] = (0.0, 0.0)
        points[2] = (0.0, -1.0)
        self.assertAlmostEqual(geometry.joint_angle(points, 0, 1, 2), 180.0, places=6)

    def test_degenerate_returns_zero_not_straight(self) -> None:
        """A collapsed joint must not be mistaken for an extended finger.

        Reading a finger as extended fires real actions, so the failure mode has
        to be the conservative one.
        """
        points = np.zeros((NUM_LANDMARKS, 2))
        points[0] = points[1] = points[2] = (0.5, 0.5)
        self.assertEqual(geometry.joint_angle(points, 0, 1, 2), 0.0)

    def test_clamped_for_duplicate_points(self) -> None:
        points = np.random.default_rng(0).random((NUM_LANDMARKS, 2))
        angle = geometry.joint_angle(points, 0, 1, 1)
        self.assertGreaterEqual(angle, 0.0)
        self.assertLessEqual(angle, 180.0)


class TestSeparability(unittest.TestCase):
    """The measured margins the default thresholds rely on."""

    def test_extended_and_curled_do_not_overlap(self) -> None:
        worst_extended_curl = 180.0
        best_curled_curl = 0.0
        worst_extended_reach = math.inf
        best_curled_reach = 0.0

        for finger in FINGERS:
            for rotation in ROTATIONS:
                for scale in SCALES:
                    for curls, thumb_out, table in (
                        (extended_curls(), 1.0, "extended"),
                        (curled_curls(), 0.0, "curled"),
                    ):
                        points = geometry.make_observation(
                            make_hand(
                                curls,
                                thumb_out=thumb_out,
                                rotation_deg=rotation,
                                scale=scale,
                                aspect_ratio=16 / 9,
                            ),
                            aspect_ratio=16 / 9,
                        ).points
                        curl = geometry.finger_curl(points, finger)
                        reach = geometry.finger_reach(points, finger)
                        if table == "extended" and finger is not Finger.THUMB:
                            worst_extended_curl = min(worst_extended_curl, curl)
                            worst_extended_reach = min(worst_extended_reach, reach)
                        elif table == "curled" and finger is not Finger.THUMB:
                            best_curled_curl = max(best_curled_curl, curl)
                            best_curled_reach = max(best_curled_reach, reach)

        # The default bands are [120, 150] deg and [0.95, 1.15] palm lengths;
        # assert real daylight on both sides.
        self.assertGreater(worst_extended_curl - 150.0, 20.0, "curl up-margin too small")
        self.assertGreater(120.0 - best_curled_curl, 20.0, "curl down-margin too small")
        self.assertGreater(worst_extended_reach - 1.15, 0.2, "reach up-margin too small")
        self.assertGreater(0.95 - best_curled_reach, 0.1, "reach down-margin too small")


class TestMisc(unittest.TestCase):
    def test_palm_axis_points_from_wrist_to_middle_knuckle(self) -> None:
        points = geometry.make_observation(
            make_hand(extended_curls(), rotation_deg=0.0), aspect_ratio=16 / 9
        ).points
        axis = geometry.palm_axis(points)
        direction = points[MIDDLE_MCP] - points[WRIST]
        direction = direction / np.linalg.norm(direction)
        np.testing.assert_allclose(axis, direction, atol=1e-12)

    def test_palm_axis_survives_coincident_points(self) -> None:
        points = np.zeros((NUM_LANDMARKS, 2))
        axis = geometry.palm_axis(points)
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=9)

    def test_palm_size_never_zero(self) -> None:
        self.assertGreater(geometry.palm_size(np.zeros((NUM_LANDMARKS, 2))), 0.0)

    def test_hand_extent(self) -> None:
        points = geometry.make_observation(make_hand(extended_curls())).points
        x0, y0, x1, y1 = geometry.hand_extent(points)
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
