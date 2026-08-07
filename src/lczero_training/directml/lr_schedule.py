"""Learning-rate schedules for the DirectML port.

A plain-Python mirror of training/lr_schedule.py. The JAX version is written
against traced arrays because it runs inside a jitted step; here the rate is
computed once per step on the host, so the same logic reads as ordinary
control flow. Segment selection, the CONSTANT/LINEAR/COSINE transitions,
looping, gaps between segments, and the clamp past the end all match.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Sequence

from proto.training_config_pb2 import LrSchedule


def _create_rule_fn(rule: LrSchedule) -> Callable[[float], float]:
    start_step = float(rule.starting_step)
    durations = [float(d) for d in rule.duration_steps]
    rates = [float(x) for x in rule.lr]
    is_looping = bool(rule.loop)

    # A rule with no segments, or no rates, is constant.
    if not durations or not rates:
        constant = rates[-1] if rates else 0.0
        return lambda step: constant

    period = sum(durations)
    if period == 0.0:
        constant = rates[-1]
        return lambda step: constant

    transitions = [
        (
            rule.transition[index]
            if index < len(rule.transition)
            else LrSchedule.Transition.CONSTANT
        )
        for index in range(len(durations))
    ]
    ends = list(itertools.accumulate(durations))
    starts = [end - duration for end, duration in zip(ends, durations)]
    # Each segment interpolates from lr[i] to lr[i+1], holding the last value
    # when the rate list is shorter than the segment list.
    from_rates = [
        rates[index] if index < len(rates) else rates[-1]
        for index in range(len(durations))
    ]
    to_rates = [
        rates[index + 1] if index + 1 < len(rates) else from_rates[index]
        for index in range(len(durations))
    ]
    last_rate = rates[-1]

    def rule_fn(step: float) -> float:
        relative = step - start_step
        if is_looping:
            # `%`, not math.fmod: it matches jnp.mod for negative steps.
            relative = relative % period

        for index, (start, end, duration) in enumerate(
            zip(starts, ends, durations)
        ):
            if duration > 0 and start <= relative < end:
                progress = min(max((relative - start) / duration, 0.0), 1.0)
                transition = transitions[index]
                low, high = from_rates[index], to_rates[index]
                if transition == LrSchedule.Transition.LINEAR:
                    return low + (high - low) * progress
                if transition == LrSchedule.Transition.COSINE:
                    return low + 0.5 * (1.0 - math.cos(math.pi * progress)) * (
                        high - low
                    )
                return low

        # Outside every segment: either a gap between them or past the end.
        return last_rate

    return rule_fn


def make_lr_schedule(
    schedules: Sequence[LrSchedule],
) -> Callable[[int], float]:
    """Combine the configured rules into a step -> learning rate function."""
    rules = list(schedules)
    if not rules:
        return lambda step: 0.0

    rule_fns = [_create_rule_fn(rule) for rule in rules]
    start_steps = [float(rule.starting_step) for rule in rules]
    first_rates = [float(rule.lr[0]) if rule.lr else 0.0 for rule in rules]

    # Ties resolve to the first rule, matching jnp.argmin/argmax.
    earliest = min(range(len(start_steps)), key=lambda i: start_steps[i])
    pre_start_rate = first_rates[earliest]
    min_start_step = start_steps[earliest]

    def schedule(step: int) -> float:
        position = float(step)
        if position < min_start_step:
            return pre_start_rate
        # The active rule is the one with the largest starting_step at or
        # below the current step.
        eligible = [
            start if position >= start else -1.0 for start in start_steps
        ]
        active = max(range(len(eligible)), key=lambda i: eligible[i])
        return float(rule_fns[active](position))

    return schedule
