"""Tests for pose estimation: finger states, hysteresis and the thumb rules."""

from __future__ import annotations

import math
import unittest

from gesture_control.geometry import make_observation
from gesture_control.pose import (
    DEFAULT_THRESHOLDS,
    FingerThresholds,
    PoseEstimator,
    Thresholds,
    ThumbThresholds,
    estimate_pose,
)
from gesture_control.synthetic import curled_curls, extended_curls, make_hand
from gesture_control.types import FINGERS, Finger, Handedness

from .helpers import ASPECTS, ROTATIONS, SCALES, extended, pose_from_curls


class TestFingerDecisions(unittest.TestCase):
    def test_open_hand_reads_all_extended(self) -> None:
        pose = pose_from_curls(extended_curls(), thumb_out=1.0)
        self.assertEqual(pose.extended_fingers, frozenset(FINGERS))
        self.assertEqual(pose.signature, "open")

    def test_fist_reads_nothing_extended(self) -> None:
        pose = pose_from_curls(curled_curls(), thumb_out=0.0)
        self.assertEqual(pose.extended_fingers, frozenset())
        self.assertEqual(pose.signature, "fist")

    def test_each_single_finger(self) -> None:
        for finger in FINGERS:
            thumb_out = 1.0 if finger is Finger.THUMB else 0.0
            with self.subTest(finger=finger.value):
                pose = pose_from_curls(extended([finger]), thumb_out=thumb_out)
                self.assertEqual(pose.extended_fingers, frozenset({finger}))
                self.assertEqual(pose.signature, finger.value)

    def test_every_combination_of_long_fingers(self) -> None:
        """All 16 subsets of the four long fingers must be read exactly."""
        long_fingers = (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
        for mask in range(16):
            wanted = {f for bit, f in enumerate(long_fingers) if mask & (1 << bit)}
            with self.subTest(extended=sorted(f.value for f in wanted)):
                pose = pose_from_curls(extended(wanted), thumb_out=0.0)
                self.assertEqual(pose.extended_fingers, frozenset(wanted))

    def test_recognition_survives_rotation_scale_and_aspect(self) -> None:
        """The whole point of the rewrite, asserted end to end."""
        cases = [
            (extended_curls(), 1.0, frozenset(FINGERS)),
            (curled_curls(), 0.0, frozenset()),
            (extended([Finger.INDEX]), 0.0, frozenset({Finger.INDEX})),
            (extended([Finger.INDEX, Finger.MIDDLE]), 0.0,
             frozenset({Finger.INDEX, Finger.MIDDLE})),
            (extended([Finger.RING, Finger.PINKY]), 0.0,
             frozenset({Finger.RING, Finger.PINKY})),
            (extended([Finger.PINKY]), 0.0, frozenset({Finger.PINKY})),
            (extended([Finger.THUMB]), 1.0, frozenset({Finger.THUMB})),
        ]
        for curls, thumb_out, expected in cases:
            for rotation in ROTATIONS:
                for scale in SCALES:
                    for aspect in ASPECTS:
                        with self.subTest(
                            expected=sorted(f.value for f in expected),
                            rotation=rotation,
                            scale=scale,
                            aspect=aspect,
                        ):
                            pose = pose_from_curls(
                                curls,
                                thumb_out=thumb_out,
                                rotation=rotation,
                                scale=scale,
                                aspect=aspect,
                            )
                            self.assertEqual(pose.extended_fingers, expected)


class TestThumbRules(unittest.TestCase):
    """The three thumb cases that a "is the thumb straight" test cannot tell
    apart, and which the two-metric rule separates."""

    def test_straight_and_out_is_up(self) -> None:
        pose = pose_from_curls(extended_curls(), thumb_out=1.0)
        self.assertTrue(pose.is_extended(Finger.THUMB))

    def test_straight_but_tucked_is_down(self) -> None:
        """Straight, but lying against the index finger -- this is the case the
        original's `lm[4].x < lm[3].x` test got wrong."""
        pose = pose_from_curls(extended_curls(), thumb_out=0.0)
        reading = pose.readings[Finger.THUMB]
        self.assertGreater(reading.curl, 170.0, "fixture should be geometrically straight")
        self.assertFalse(reading.extended, "a tucked thumb must not read as extended")

    def test_folded_across_palm_is_down(self) -> None:
        pose = pose_from_curls(curled_curls(), thumb_out=0.0)
        self.assertFalse(pose.is_extended(Finger.THUMB))

    def test_abduction_recorded_for_thumb_only(self) -> None:
        pose = pose_from_curls(extended_curls(), thumb_out=1.0)
        self.assertFalse(math.isnan(pose.readings[Finger.THUMB].abduction))
        self.assertTrue(math.isnan(pose.readings[Finger.INDEX].abduction))

    def test_thumb_up_survives_rotation(self) -> None:
        for rotation in ROTATIONS:
            with self.subTest(rotation=rotation):
                pose = pose_from_curls(
                    extended([Finger.THUMB]), thumb_out=1.0, rotation=rotation
                )
                self.assertEqual(pose.extended_fingers, frozenset({Finger.THUMB}))


class TestHysteresis(unittest.TestCase):
    def _index_at(self, curl: float, estimator: PoseEstimator) -> object:
        """A hand with every finger closed except the index, at a given curl."""
        curls = dict.fromkeys(FINGERS, 1.0)
        curls[Finger.INDEX] = curl
        return pose_from_curls(curls, thumb_out=0.0, estimator=estimator)

    @staticmethod
    def _in_band(reading, thresholds) -> bool:
        """True when neither the up nor the down rule fires for this reading."""
        definitely_down = (
            reading.curl <= thresholds.curl_down or reading.reach <= thresholds.reach_down
        )
        definitely_up = (
            reading.curl >= thresholds.curl_up and reading.reach >= thresholds.reach_up
        )
        return not definitely_down and not definitely_up

    def test_state_is_sticky_inside_the_band(self) -> None:
        """A finger between the thresholds keeps its previous answer.

        This is what stops a hand resting near a threshold from flickering and
        machine-gunning actions.
        """
        estimator = PoseEstimator()
        thresholds = DEFAULT_THRESHOLDS.finger

        first = self._index_at(0.0, estimator)
        self.assertTrue(first.is_extended(Finger.INDEX), "fixture should start extended")

        in_band = 0
        for step in range(1, 21):
            pose = self._index_at(step / 20.0, estimator)
            reading = pose.readings[Finger.INDEX]
            if self._in_band(reading, thresholds):
                in_band += 1
                self.assertTrue(
                    pose.is_extended(Finger.INDEX),
                    f"finger entered the band (curl={reading.curl:.1f}, "
                    f"reach={reading.reach:.3f}) but flipped state",
                )
        self.assertGreater(in_band, 0, "fixture never entered the hysteresis band")

    def test_hysteresis_does_not_freeze_a_real_transition(self) -> None:
        """Sticky must not mean stuck: a genuine close must still register."""
        estimator = PoseEstimator()
        self._index_at(0.0, estimator)
        final = self._index_at(1.0, estimator)
        self.assertFalse(final.is_extended(Finger.INDEX))

    def test_first_sighting_uses_the_midpoint(self) -> None:
        estimator = PoseEstimator()
        pose = estimator.estimate(
            make_observation(make_hand(extended([Finger.INDEX]), aspect_ratio=16 / 9),
                             aspect_ratio=16 / 9)
        )
        self.assertTrue(pose.is_extended(Finger.INDEX))

    def test_reset_clears_memory(self) -> None:
        estimator = PoseEstimator()
        estimator.estimate(
            make_observation(make_hand(extended_curls(), aspect_ratio=16 / 9), aspect_ratio=16 / 9)
        )
        estimator.reset()
        self.assertEqual(estimator.reset_count, 1)
        pose = estimator.estimate(
            make_observation(make_hand(curled_curls(), thumb_out=0.0, aspect_ratio=16 / 9),
                             aspect_ratio=16 / 9)
        )
        self.assertEqual(pose.extended_fingers, frozenset())

    def test_memory_is_per_handedness(self) -> None:
        """Two hands in frame must not share hysteresis state."""
        estimator = PoseEstimator()
        raw = make_hand(extended_curls(), aspect_ratio=16 / 9)
        left = make_observation(raw, aspect_ratio=16 / 9, handedness=Handedness.LEFT)
        right = make_observation(raw, aspect_ratio=16 / 9, handedness=Handedness.RIGHT)
        estimator.estimate(left)
        estimator.estimate(right)
        self.assertEqual(len(estimator._memory), 2)
        estimator.reset(Handedness.LEFT)
        self.assertEqual(len(estimator._memory), 1)


class TestThresholdValidation(unittest.TestCase):
    def test_midpoints(self) -> None:
        finger = FingerThresholds(curl_up=150.0, curl_down=120.0, reach_up=1.2, reach_down=1.0)
        self.assertAlmostEqual(finger.curl_mid, 135.0)
        self.assertAlmostEqual(finger.reach_mid, 1.1)
        thumb = ThumbThresholds()
        self.assertGreater(thumb.curl_mid, thumb.curl_down)
        self.assertLess(thumb.curl_mid, thumb.curl_up)

    def test_custom_thresholds_are_used(self) -> None:
        strict = Thresholds(
            finger=FingerThresholds(curl_up=179.9, curl_down=179.0, reach_up=99.0, reach_down=98.0),
            thumb=ThumbThresholds(),
        )
        pose = estimate_pose(
            make_observation(make_hand(extended_curls(), aspect_ratio=16 / 9), aspect_ratio=16 / 9),
            strict,
        )
        # An impossible reach requirement means nothing reads as extended.
        self.assertFalse(pose.is_extended(Finger.INDEX))

    def test_default_thresholds_are_the_documented_ones(self) -> None:
        self.assertEqual(DEFAULT_THRESHOLDS.finger.curl_up, 150.0)
        self.assertEqual(DEFAULT_THRESHOLDS.finger.reach_up, 1.15)
        self.assertEqual(DEFAULT_THRESHOLDS.thumb.abduction_up, 0.35)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
