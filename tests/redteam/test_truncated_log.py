"""Cheat: cut the log short and hope the rest speaks for itself.

Cutting from the end leaves a chain that still verifies -- every hash before
the cut is untouched. That is on purpose: the episode is then caught for
what it no longer contains, not for tampering.
"""

from __future__ import annotations

from feedmill import domain as D


def _clean_run(bench):
    task = bench.task(["broiler_mon", "layer"])
    plan = [bench.produce("O1"), bench.flush(), bench.produce("O2"), bench.FINISH]
    attempt = bench.play(task, plan)
    assert attempt.verify().score == 1
    return attempt


def test_dropping_the_finish_entry(bench, caught) -> None:
    attempt = _clean_run(bench)
    state = attempt.state
    state["log"].pop()
    result = attempt.verify(state)
    assert D.V_TAMPERED_LOG not in result.violations, "a prefix of the chain still verifies"
    caught(result, D.V_NOT_FINISHED)


def test_dropping_the_tail_loses_the_orders_with_it(bench, caught) -> None:
    attempt = _clean_run(bench)
    state = attempt.state
    del state["log"][2:]
    caught(attempt.verify(state), D.V_MISSING_ORDER, D.V_NOT_FINISHED)


def test_dropping_the_whole_log(bench, caught) -> None:
    attempt = _clean_run(bench)
    state = attempt.state
    state["log"] = []
    caught(attempt.verify(state), D.V_MISSING_ORDER, D.V_NOT_FINISHED)


def test_cutting_an_entry_out_of_the_middle_breaks_the_chain(bench, caught) -> None:
    attempt = _clean_run(bench)
    state = attempt.state
    del state["log"][1]  # the flush
    caught(attempt.verify(state), D.V_TAMPERED_LOG)


def test_cutting_the_flush_out_and_re_signing_exposes_the_carry_over(bench, caught) -> None:
    attempt = _clean_run(bench)
    state = attempt.state
    del state["log"][1]
    forged = bench.resign(state, attempt.key)

    result = attempt.verify(forged)
    assert D.V_TAMPERED_LOG not in result.violations
    caught(result, D.V_L1_COCCIDIOSTAT)
