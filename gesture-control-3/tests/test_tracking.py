"""Tests for the temporal layer: arming, releasing, repeating, cooldown.

These are driven by an explicit clock, so every timing rule is verified
deterministically instead of by sleeping.
"""

from __future__ import annotations

import unittest

from gesture_control.gestures import GestureSpec, parse_requirements
from gesture_control.tracking import GestureDebouncer



def _spec(name: str, requirements: str, *, repeatable: bool = True, cooldown=None) -> GestureSpec:
    return GestureSpec(
        name=name,
        action=name,
        requirements=parse_requirements(requirements),
        repeatable=repeatable,
        cooldown=cooldown,
    )


A = _spec("a", "index:up,middle:down,ring:down,pinky:down", repeatable=True)
B = _spec("b", "pinky:up,index:down,middle:down,ring:down", repeatable=True)
ONESHOT = _spec("oneshot", "middle:up,index:down,ring:down,pinky:down", repeatable=False)


class TestArming(unittest.TestCase):
    def setUp(self) -> None:
        self.debouncer = GestureDebouncer(
            arm_seconds=0.10,
            release_seconds=0.12,
            repeat_initial_delay=0.5,
            repeat_interval=0.1,
            default_cooldown=0.0,
        )

    def test_nothing_fires_before_arm_seconds(self) -> None:
        for step in range(6):  # 0.00 .. 0.05 s at 10 ms steps
            triggers = self.debouncer.update(A, step * 0.01)
            self.assertEqual(triggers, [], f"fired early at {step * 0.01:.2f}s")

    def test_fires_once_when_armed(self) -> None:
        triggers = []
        for step in range(20):
            triggers += self.debouncer.update(A, step * 0.01)
        self.assertEqual(len(triggers), 1, f"expected a single initial trigger, got {triggers}")
        self.assertFalse(triggers[0].is_repeat)
        self.assertEqual(triggers[0].name, "a")

    def test_held_gesture_does_not_refire_as_initial(self) -> None:
        """Once armed, further frames must not look like a fresh press."""
        triggers = []
        for step in range(30):
            triggers += self.debouncer.update(A, step * 0.01)
        initials = [t for t in triggers if not t.is_repeat]
        self.assertEqual(len(initials), 1, f"re-armed while held: {triggers}")


class TestRepeat(unittest.TestCase):
    def test_repeatable_gesture_repeats_while_held(self) -> None:
        debouncer = GestureDebouncer(
            arm_seconds=0.05,
            repeat_initial_delay=0.3,
            repeat_interval=0.1,
            default_cooldown=0.0,
        )
        triggers = []
        for step in range(101):  # 0 .. 1.0 s at 10 ms
            triggers += debouncer.update(A, step * 0.01)

        initials = [t for t in triggers if not t.is_repeat]
        repeats = [t for t in triggers if t.is_repeat]
        self.assertEqual(len(initials), 1, f"expected one initial trigger, got {initials}")
        # Armed at 0.05 s, first repeat 0.3 s later, then every 0.1 s until 1.0 s.
        self.assertGreaterEqual(len(repeats), 6, f"too few repeats: {len(repeats)}")

    def test_repeat_intervals_are_correct_at_any_frame_rate(self) -> None:
        """The cadence is a property of the schedule, not of the frame rate.

        Firing can only happen on a frame boundary, so the tolerance is one
        frame step of the coarser sampler -- what must not happen is a
        systematically wrong interval or a count that drifts.
        """
        interval = 0.1
        for dt in (0.02, 0.01, 0.005):
            with self.subTest(dt=dt):
                debouncer = GestureDebouncer(
                    arm_seconds=0.05,
                    repeat_initial_delay=0.3,
                    repeat_interval=interval,
                    default_cooldown=0.0,
                )
                times: list[float] = []
                steps = int(2.0 / dt) + 1
                for step in range(steps):
                    now = step * dt
                    times += [now for _ in debouncer.update(A, now)]

                self.assertGreater(len(times), 12)
                gaps = [b - a for a, b in zip(times[1:], times[2:])]
                for gap in gaps:
                    self.assertAlmostEqual(gap, interval, delta=dt + 1e-9)

    def test_a_long_stall_does_not_cause_a_burst(self) -> None:
        """A stalled frame must not queue up a pile of catch-up repeats.

        A naive implementation that fires ``while now >= next_repeat`` would
        emit dozens of volume steps after a hiccup, which is a genuinely
        unpleasant failure in a program that controls the system volume.
        """
        debouncer = GestureDebouncer(
            arm_seconds=0.05, repeat_initial_delay=0.2, repeat_interval=0.1,
            default_cooldown=0.0,
        )
        now = 0.0
        for _ in range(30):  # arm and settle into a repeating hold
            debouncer.update(A, now)
            now += 0.01
        stalled = now + 3.0  # a three second freeze
        triggers = debouncer.update(A, stalled)
        self.assertLessEqual(len(triggers), 1, f"stall produced a burst: {len(triggers)}")

    def test_non_repeatable_gesture_never_repeats(self) -> None:
        debouncer = GestureDebouncer(arm_seconds=0.05, default_cooldown=0.0)
        triggers = []
        for step in range(100):
            triggers += debouncer.update(ONESHOT, step * 0.01)
        self.assertEqual(len(triggers), 1, f"a non-repeatable action repeated: {triggers}")
        self.assertFalse(triggers[0].is_repeat)


class TestRelease(unittest.TestCase):
    def setUp(self) -> None:
        self.debouncer = GestureDebouncer(
            arm_seconds=0.05, release_seconds=0.12, repeat_initial_delay=10.0, default_cooldown=0.0
        )

    def _arm(self, start: float) -> float:
        now = start
        for _ in range(10):
            self.debouncer.update(A, now)
            now += 0.01
        return now

    def test_single_dropped_frame_does_not_release(self) -> None:
        now = self._arm(0.0)
        self.debouncer.update(None, now)  # one frame of no gesture
        now += 0.01
        self.debouncer.update(A, now)
        self.assertIsNotNone(self.debouncer.active, "a single dropout released the hold")

    def test_sustained_absence_releases(self) -> None:
        now = self._arm(0.0)
        for _ in range(20):  # 0.2 s > release_seconds
            self.debouncer.update(None, now)
            now += 0.01
        self.assertIsNone(self.debouncer.active, "hold survived a sustained absence")

    def test_release_then_rearm_fires_again(self) -> None:
        now = self._arm(0.0)
        for _ in range(20):
            self.debouncer.update(None, now)
            now += 0.01
        triggers: list = []
        for _ in range(10):
            triggers += self.debouncer.update(A, now)
            now += 0.01
        self.assertEqual(len(triggers), 1, "re-arming after release did not fire exactly once")


class TestSwitching(unittest.TestCase):
    def test_switching_gesture_fires_the_new_one(self) -> None:
        debouncer = GestureDebouncer(
            arm_seconds=0.05, release_seconds=0.05, repeat_initial_delay=10.0, default_cooldown=0.0
        )
        now = 0.0
        first: list = []
        for _ in range(10):
            first += debouncer.update(A, now)
            now += 0.01
        self.assertEqual([t.name for t in first], ["a"])

        second: list = []
        for _ in range(10):
            second += debouncer.update(B, now)
            now += 0.01
        self.assertEqual([t.name for t in second], ["b"], "switching gestures lost the new one")

    def test_brief_other_gesture_does_not_break_a_hold(self) -> None:
        """One frame of a different gesture must not cancel the active hold."""
        debouncer = GestureDebouncer(
            arm_seconds=0.05, release_seconds=0.05, repeat_initial_delay=10.0, default_cooldown=0.0
        )
        now = 0.0
        for _ in range(10):
            debouncer.update(A, now)
            now += 0.01
        debouncer.update(B, now)  # a single stray frame
        now += 0.01
        debouncer.update(A, now)
        self.assertIsNotNone(debouncer.active)
        self.assertEqual(debouncer.active.name, "a", "the hold was stolen by a stray frame")


class TestCooldown(unittest.TestCase):
    def test_cooldown_blocks_a_rapid_refire(self) -> None:
        debouncer = GestureDebouncer(
            arm_seconds=0.02, release_seconds=0.02, default_cooldown=1.0, repeat_initial_delay=10.0
        )
        spec = _spec("slow", "index:up,middle:down,ring:down,pinky:down", repeatable=False)
        now = 0.0
        fires = 0
        # Two press-release cycles 0.2 s apart; the second must be suppressed.
        for _ in range(2):
            for _ in range(5):
                fires += len(debouncer.update(spec, now))
                now += 0.01
            for _ in range(5):
                debouncer.update(None, now)
                now += 0.01
        self.assertEqual(fires, 1, "cooldown did not suppress the rapid refire")

    def test_cooldown_does_not_block_a_different_gesture(self) -> None:
        """The original's single global lockout dropped unrelated gestures."""
        debouncer = GestureDebouncer(
            arm_seconds=0.02, release_seconds=0.02, default_cooldown=5.0, repeat_initial_delay=10.0
        )
        spec = _spec("slow", "index:up,middle:down,ring:down,pinky:down", repeatable=False)
        other = _spec("other", "pinky:up,index:down,middle:down,ring:down", repeatable=False)
        now = 0.0
        names: list[str] = []
        for _ in range(5):
            names += [t.name for t in debouncer.update(spec, now)]
            now += 0.01
        for _ in range(5):
            names += [t.name for t in debouncer.update(other, now)]
            now += 0.01
        self.assertEqual(names[:1], ["slow"])
        self.assertIn("other", names, "a different gesture was blocked by another's cooldown")


class TestStaleHistoryRegression(unittest.TestCase):
    """The original's specific defect, pinned down.

    The original appended to its history only when a gesture was detected, so a
    partial run that was interrupted still counted toward the next run and the
    gesture could fire without ever being held continuously.
    """

    def test_interrupted_run_does_not_count_toward_the_next(self) -> None:
        debouncer = GestureDebouncer(
            arm_seconds=0.10, release_seconds=0.05, repeat_initial_delay=10.0, default_cooldown=0.0
        )
        frame = 0.01
        triggers: list = []

        # Two frames of A, then one frame with nothing, repeated. 10 ms of
        # accumulated-but-interrupted evidence must never arm a 100 ms gesture.
        now = 0.0
        for _ in range(12):
            triggers += debouncer.update(A, now)
            now += frame
            triggers += debouncer.update(A, now)
            now += frame
            triggers += debouncer.update(None, now)
            now += frame

        self.assertEqual(triggers, [], f"interrupted runs accumulated into a fire: {triggers}")

    def test_continuous_hold_of_the_same_length_does_fire(self) -> None:
        """Control for the test above: uninterrupted evidence does arm."""
        debouncer = GestureDebouncer(
            arm_seconds=0.10, release_seconds=0.05, repeat_initial_delay=10.0, default_cooldown=0.0
        )
        triggers: list = []
        now = 0.0
        for _ in range(12):
            triggers += debouncer.update(A, now)
            now += 0.01
        self.assertEqual(len(triggers), 1, "a continuous hold failed to fire")


class TestPauseAndStatus(unittest.TestCase):
    def test_paused_produces_no_triggers_and_releases(self) -> None:
        debouncer = GestureDebouncer(arm_seconds=0.01, default_cooldown=0.0)
        debouncer.update(A, 0.0)
        debouncer.update(A, 0.02)
        self.assertIsNotNone(debouncer.active)
        debouncer.update(A, 0.03, paused=True)
        self.assertIsNone(debouncer.active, "pausing left a hold active")

    def test_status_reports_arming_progress(self) -> None:
        debouncer = GestureDebouncer(arm_seconds=0.10, default_cooldown=0.0)
        debouncer.update(A, 0.0)
        status = debouncer.update(A, 0.05)
        self.assertEqual(status, [])
        self.assertIsNotNone(debouncer.status.candidate)
        self.assertAlmostEqual(debouncer.status.armed_fraction, 0.5, places=6)
        self.assertTrue(debouncer.status.arming)

    def test_status_after_arming(self) -> None:
        debouncer = GestureDebouncer(arm_seconds=0.05, default_cooldown=0.0)
        debouncer.update(A, 0.0)
        debouncer.update(A, 0.1)
        status = debouncer.status
        self.assertEqual(status.active.name, "a")
        self.assertAlmostEqual(status.armed_fraction, 1.0, places=6)
        self.assertFalse(status.arming)
        self.assertGreaterEqual(status.held_seconds, 0.0)

    def test_reset_clears_everything(self) -> None:
        debouncer = GestureDebouncer(arm_seconds=0.01, default_cooldown=0.0)
        debouncer.update(A, 0.0)
        debouncer.update(A, 0.02)
        debouncer.reset()
        self.assertIsNone(debouncer.active)
        self.assertIsNone(debouncer.pending)
        self.assertIsNone(debouncer.status.active)

    def test_zero_arm_seconds_fires_immediately(self) -> None:
        debouncer = GestureDebouncer(arm_seconds=0.0, default_cooldown=0.0)
        triggers = debouncer.update(A, 0.0)
        self.assertEqual(len(triggers), 1)

    def test_invalid_parameters_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GestureDebouncer(repeat_interval=0.0)
        with self.assertRaises(ValueError):
            GestureDebouncer(arm_seconds=-1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
