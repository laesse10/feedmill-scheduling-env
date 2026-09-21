"""Cheat: flush, but not where it helps -- a batch runs in between."""

from __future__ import annotations

from feedmill import domain as D


def test_flushing_before_the_dirty_batch_instead_of_after_it(bench, caught) -> None:
    """flush -> broiler_mon -> layer: the line is clean at the wrong moment."""
    task = bench.task(["broiler_mon", "layer"])
    attempt = bench.play(
        task, [bench.flush(), bench.produce("O1"), bench.produce("O2"), bench.FINISH]
    )
    caught(attempt.verify(), D.V_L1_COCCIDIOSTAT)


def test_a_house_rule_flush_with_a_batch_in_between(bench, caught) -> None:
    """H1 wants the flush directly before that customer's order."""
    rule = D.HouseRule(
        D.H1_CUSTOMER_FLUSH, {"customer": "Zuber Farms"}, "orders of Zuber Farms follow a flush"
    )
    task = bench.task(
        ["layer", "layer"],
        customers=["Hof Meier", "Zuber Farms"],
        house_rules=[rule],
    )
    attempt = bench.play(
        task, [bench.flush(), bench.produce("O1"), bench.produce("O2"), bench.FINISH]
    )
    caught(attempt.verify(), D.V_H1_CUSTOMER_FLUSH)


def test_the_same_plan_with_the_flush_in_the_right_place_passes(bench) -> None:
    rule = D.HouseRule(
        D.H1_CUSTOMER_FLUSH, {"customer": "Zuber Farms"}, "orders of Zuber Farms follow a flush"
    )
    task = bench.task(
        ["layer", "layer"],
        customers=["Hof Meier", "Zuber Farms"],
        house_rules=[rule],
    )
    attempt = bench.play(
        task, [bench.produce("O1"), bench.flush(), bench.produce("O2"), bench.FINISH]
    )
    assert attempt.verify().score == 1
