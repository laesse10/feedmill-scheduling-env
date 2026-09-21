"""Cheat: relabel an entry so the verifier treats it as a no-op.

The environment marks an action ``invalid`` when the press could not have
done it, and leaves the state untouched. An agent that could rewrite the log
might mark a real action ``invalid`` to have it skipped on replay. SPEC
section 9 rule 2 closes that off from the other side: any invalid entry
scores 0, whatever else the episode did.
"""

from __future__ import annotations

from feedmill import domain as D


def _legal_run(bench):
    task = bench.task(["broiler_mon", "layer"], due=600)
    attempt = bench.play(
        task,
        [
            bench.produce("O1"),
            bench.wait(5),
            bench.flush(),
            bench.produce("O2"),
            bench.FINISH,
        ],
    )
    assert attempt.verify().score == 1
    return attempt


def test_relabelling_a_harmless_entry_still_costs_the_run(bench, caught) -> None:
    """Skipping the wait changes nothing physically. It is still an invalid entry."""
    attempt = _legal_run(bench)
    state = attempt.state
    assert state["log"][1]["action"]["tool"] == D.TOOL_WAIT
    state["log"][1]["status"] = D.STATUS_INVALID
    forged = bench.resign(state, attempt.key)

    result = attempt.verify(forged)
    assert result.violations == [D.V_INVALID_ACTION]
    caught(result, D.V_INVALID_ACTION)


def test_relabelling_the_flush_skips_it_and_exposes_the_carry_over(bench, caught) -> None:
    attempt = _legal_run(bench)
    state = attempt.state
    assert state["log"][2]["action"]["tool"] == D.TOOL_FLUSH
    state["log"][2]["status"] = D.STATUS_INVALID
    forged = bench.resign(state, attempt.key)

    caught(attempt.verify(forged), D.V_INVALID_ACTION, D.V_L1_COCCIDIOSTAT)


def test_relabelling_an_invalid_entry_as_ok_does_not_help_either(bench, caught) -> None:
    """The verifier decides possibility itself, so the label buys nothing."""
    task = bench.task(["layer"], initial_die=6)  # layer needs a 3 mm die
    attempt = bench.play(task, [bench.produce("O1"), bench.FINISH])
    state = attempt.state
    assert state["log"][0]["status"] == D.STATUS_INVALID

    state["log"][0]["status"] = D.STATUS_OK
    forged = bench.resign(state, attempt.key)
    caught(attempt.verify(forged), D.V_INVALID_ACTION)
