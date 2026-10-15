"""Tests for the One Euro landmark filter."""

from __future__ import annotations

import math
import unittest

import numpy as np

from gesture_control.smoothing import LandmarkSmoother, OneEuroFilter


class TestOneEuroFilter(unittest.TestCase):
    def test_first_sample_passes_through(self) -> None:
        filt = OneEuroFilter()
        self.assertEqual(filt(3.5, 0.0), 3.5)
        self.assertTrue(filt.initialised)
        self.assertEqual(filt.samples, 1)

    def test_constant_signal_is_reproduced_exactly(self) -> None:
        filt = OneEuroFilter()
        filt(2.0, 0.0)
        for step in range(1, 50):
            self.assertAlmostEqual(filt(2.0, step * 0.02), 2.0, places=12)

    def test_converges_to_a_step_without_overshoot(self) -> None:
        filt = OneEuroFilter(min_cutoff=2.0, beta=0.0)
        filt(0.0, 0.0)
        values = [filt(1.0, step * 0.02) for step in range(1, 200)]
        self.assertGreater(values[-1], 0.99)
        self.assertLessEqual(max(values), 1.0 + 1e-12, "low-pass filter overshot")

    def test_reduces_jitter_at_rest(self) -> None:
        rng = np.random.default_rng(1234)
        noise = rng.normal(0.0, 0.01, size=400)
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        outputs = [filt(1.0 + value, step * 0.02) for step, value in enumerate(noise)]
        self.assertLess(
            float(np.std(outputs[50:])),
            float(np.std(noise[50:])) * 0.5,
            "filter did not meaningfully reduce jitter",
        )

    def test_beta_reduces_lag_on_fast_motion(self) -> None:
        """The defining property of the filter.

        With ``beta > 0`` the cutoff rises with velocity, so a fast ramp is
        tracked more closely than the same filter with ``beta = 0``.
        """
        adaptive = OneEuroFilter(min_cutoff=1.0, beta=0.6)
        fixed = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        adaptive(0.0, 0.0)
        fixed(0.0, 0.0)

        target = 0.0
        adaptive_error = fixed_error = 0.0
        for step in range(1, 100):
            target = step * 0.03  # a fast, steady ramp
            time = step * 0.02
            adaptive_error += abs(adaptive(target, time) - target)
            fixed_error += abs(fixed(target, time) - target)

        self.assertLess(
            adaptive_error,
            fixed_error,
            "velocity adaptation did not improve tracking of fast motion",
        )

    def test_velocity_adaptation_is_gentle_at_rest(self) -> None:
        """Adaptivity must not defeat smoothing when the hand is still."""
        rng = np.random.default_rng(7)
        adaptive = OneEuroFilter(min_cutoff=1.0, beta=0.035)
        noise = rng.normal(0.0, 0.01, size=300)
        outputs = [adaptive(value, step * 0.02) for step, value in enumerate(noise)]
        self.assertLess(float(np.std(outputs[100:])), 0.006)

    def test_non_increasing_timestamps_hold_the_estimate(self) -> None:
        filt = OneEuroFilter()
        filt(1.0, 0.0)
        value = filt(50.0, 0.0)
        self.assertEqual(value, 1.0, "duplicate timestamp caused a jump")
        self.assertEqual(filt(50.0, -1.0), 1.0, "backwards clock caused a jump")

    def test_nan_timestamp_is_handled(self) -> None:
        filt = OneEuroFilter()
        filt(1.0, 0.0)
        self.assertEqual(filt(9.0, math.nan), 1.0)

    def test_reset_restarts(self) -> None:
        filt = OneEuroFilter()
        filt(1.0, 0.0)
        filt(2.0, 0.1)
        filt.reset()
        self.assertFalse(filt.initialised)
        self.assertEqual(filt(5.0, 5.0), 5.0)

    def test_invalid_parameters_rejected(self) -> None:
        for kwargs in (
            {"min_cutoff": 0.0},
            {"min_cutoff": -1.0},
            {"d_cutoff": 0.0},
            {"beta": -0.1},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                OneEuroFilter(**kwargs)


class TestLandmarkSmoother(unittest.TestCase):
    def _landmarks(self, value: float = 0.5, count: int = 21) -> np.ndarray:
        array = np.zeros((count, 3), dtype=np.float64)
        array[:, 0] = value
        array[:, 1] = value
        array[:, 2] = 0.25
        return array

    def test_shape_is_preserved(self) -> None:
        smoother = LandmarkSmoother()
        out = smoother.smooth(self._landmarks(), 0.0)
        self.assertEqual(out.shape, (21, 3))

    def test_input_is_not_mutated(self) -> None:
        smoother = LandmarkSmoother()
        original = self._landmarks()
        snapshot = original.copy()
        smoother.smooth(original, 0.0)
        smoother.smooth(self._landmarks(0.9), 0.05)
        np.testing.assert_array_equal(original, snapshot)

    def test_z_channel_is_passed_through(self) -> None:
        smoother = LandmarkSmoother()
        out = smoother.smooth(self._landmarks(), 0.0)
        np.testing.assert_allclose(out[:, 2], 0.25)

    def test_disabled_smoother_returns_input(self) -> None:
        smoother = LandmarkSmoother(enabled=False)
        array = self._landmarks()
        np.testing.assert_array_equal(smoother.smooth(array, 0.0), array)
        self.assertFalse(smoother.enabled)

    def test_converges_on_a_still_hand(self) -> None:
        smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.0)
        smoother.smooth(self._landmarks(0.0), 0.0)
        for step in range(1, 300):
            out = smoother.smooth(self._landmarks(1.0), step * 0.02)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=3)

    def test_reset_drops_history(self) -> None:
        smoother = LandmarkSmoother()
        smoother.smooth(self._landmarks(0.0), 0.0)
        smoother.reset()
        out = smoother.smooth(self._landmarks(0.9), 0.02)
        self.assertAlmostEqual(float(out[0, 0]), 0.9, places=9)

    def test_jitter_is_reduced_end_to_end(self) -> None:
        rng = np.random.default_rng(99)
        smoother = LandmarkSmoother(min_cutoff=1.0, beta=0.0)
        outputs = [
            float(smoother.smooth(self._landmarks(0.5 + rng.normal(0, 0.01)), step * 0.02)[0, 0])
            for step in range(400)
        ]
        self.assertLess(float(np.std(outputs[100:])), 0.005)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
