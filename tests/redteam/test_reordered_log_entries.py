"""Cheat: shuffle the log so the schedule reads better than it ran."""

from __future__ import annotations

from feedmill import domain as D


def _legal_run(bench):
    task = bench.task(["broiler_mon", "layer"])
    plan = [bench.produce("O1"), bench.flush(), bench.produce("O2"), bench.FINISH]
    attempt = bench.play(task, plan)
    assert attempt.verify().score == 1
    return attempt


def test_swapping_two_entries_breaks_the_chain(bench, caught) -> None:
    attempt = _legal_run(bench)
    state = attempt.state
    state["log"][1], state["log"][2] = state["log"][2], state["log"][1]
    caught(attempt.verify(state), D.V_TAMPERED_LOG)


def test_reversing_the_log_breaks_the_chain(bench, caught) -> None:
    attempt = _legal_run(bench)
    state = attempt.state
    state["log"].reverse()
    caught(attempt.verify(state), D.V_TAMPERED_LOG)


def test_renumbering_the_entries_does_not_repair_the_chain(bench, caught) -> None:
    attempt = _legal_run(bench)
    state = attempt.state
    state["log"][1], state["log"][2] = state["log"][2], state["log"][1]
    for index, entry in enumerate(state["log"]):
        entry["i"] = index
    caught(attempt.verify(state), D.V_TAMPERED_LOG)


def test_reordering_with_the_stolen_key_changes_what_was_replayed(bench, caught) -> None:
    """Re-signing makes the chain valid, and the schedule illegal.

    The flush moves behind the layer batch, so the verifier replays exactly
    what the log now says: monensin straight into layer feed.
    """
    attempt = _legal_run(bench)
    state = attempt.state
    state["log"][1], state["log"][2] = state["log"][2], state["log"][1]
    forged = bench.resign(state, attempt.key)

    result = attempt.verify(forged)
    assert D.V_TAMPERED_LOG not in result.violations, "the chain itself is valid again"
    caught(result, D.V_L1_COCCIDIOSTAT)
