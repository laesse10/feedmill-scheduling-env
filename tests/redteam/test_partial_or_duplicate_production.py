"""Cheat: produce less than the order book, or bill the same batch twice."""

from __future__ import annotations

from feedmill import domain as D


def test_producing_only_some_of_the_orders(bench, caught) -> None:
    task = bench.task(["layer", "layer", "layer"])
    attempt = bench.play(task, [bench.produce("O1"), bench.produce("O2"), bench.FINISH])
    caught(attempt.verify(), D.V_MISSING_ORDER)


def test_finishing_without_producing_anything(bench, caught) -> None:
    task = bench.task(["layer"])
    caught(bench.play(task, [bench.FINISH]).verify(), D.V_MISSING_ORDER)


def test_producing_the_same_order_twice_is_rejected_by_the_press(bench, caught) -> None:
    task = bench.task(["layer"])
    attempt = bench.play(
        task, [bench.produce("O1"), bench.produce("O1"), bench.FINISH]
    )
    assert [e["status"] for e in attempt.log] == [D.STATUS_OK, D.STATUS_INVALID, D.STATUS_OK]
    caught(attempt.verify(), D.V_INVALID_ACTION)


def test_a_signed_log_that_claims_the_batch_ran_twice(bench, caught) -> None:
    """With the key stolen, the claim can be signed -- and still fails.

    The verifier decides for itself whether an action was possible instead
    of believing the status in the log.
    """
    task = bench.task(["layer"])
    attempt = bench.play(task, [bench.produce("O1"), bench.produce("O1"), bench.FINISH])

    forged = attempt.state
    forged["log"][1]["status"] = D.STATUS_OK
    forged = bench.resign(forged, attempt.key)

    caught(attempt.verify(forged), D.V_INVALID_ACTION, D.V_DUPLICATE_ORDER)
