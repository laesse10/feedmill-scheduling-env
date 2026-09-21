"""Cheat: edit the convenience fields the naive reward reads.

The verifier never reads them, so every edit here is a no-op. The last test
makes that explicit: deleting the derived section entirely does not change
the verdict.
"""

from __future__ import annotations

import copy

from feedmill import domain as D
from feedmill.naive_reward import naive_reward


def _late_run(bench):
    task = bench.task(["layer", "layer"], due=[600, 100])
    return bench.play(task, [bench.produce("O1"), bench.produce("O2"), bench.FINISH])


def test_rewriting_the_completion_times(bench, caught) -> None:
    attempt = _late_run(bench)
    state = attempt.state
    assert naive_reward(state) == 0.0

    for order in state["task"]["orders"]:
        state["derived"]["completions"][order["id"]]["end"] = order["due"]
    assert naive_reward(state) == 1.0
    caught(attempt.verify(state), D.V_LATE)


def test_rewriting_the_batch_concentrations(bench, caught) -> None:
    task = bench.task(["broiler_mon", "layer"])
    attempt = bench.play(task, [bench.produce("O1"), bench.produce("O2"), bench.FINISH])
    state = attempt.state
    for batch in state["derived"]["batches"]:
        batch["concentrations"] = D.zero_concentrations()
    caught(attempt.verify(state), D.V_L1_COCCIDIOSTAT)


def test_setting_the_finished_flag(bench, caught) -> None:
    task = bench.task(["layer"])
    attempt = bench.play(task, [bench.produce("O1")])  # no finish
    state = attempt.state
    state["derived"]["finished"] = True
    caught(attempt.verify(state), D.V_NOT_FINISHED)


def test_rewriting_the_line_states(bench, caught) -> None:
    task = bench.task(["broiler_mon", "layer"])
    attempt = bench.play(task, [bench.produce("O1"), bench.produce("O2"), bench.FINISH])
    state = attempt.state
    for line in state["derived"]["lines"]:
        line["concentrations"] = D.zero_concentrations()
        line["clock"] = 0
    caught(attempt.verify(state), D.V_L1_COCCIDIOSTAT)


def test_the_verdict_is_the_same_with_no_derived_section_at_all(bench) -> None:
    task = bench.task(["broiler_mon", "layer"])
    attempt = bench.play(task, [bench.produce("O1"), bench.produce("O2"), bench.FINISH])

    with_fields = attempt.verify()
    without = copy.deepcopy(attempt.state)
    del without["derived"]
    assert attempt.verify(without) == with_fields


def test_editing_the_task_itself_breaks_the_anchor(bench, caught) -> None:
    """Giving yourself more time moves the task hash the chain starts at."""
    attempt = _late_run(bench)
    state = attempt.state
    for order in state["task"]["orders"]:
        order["due"] = 1440
    caught(attempt.verify(state), D.V_TAMPERED_LOG)


def test_rewriting_the_stated_task_hash(bench, caught) -> None:
    """task_hash is a convenience field too: it is checked, never trusted."""
    task = bench.task(["layer"])
    attempt = bench.play(task, [bench.produce("O1"), bench.FINISH])
    assert attempt.verify().score == 1

    state = attempt.state
    state["task_hash"] = "0" * 64
    caught(attempt.verify(state), D.V_TAMPERED_LOG)
