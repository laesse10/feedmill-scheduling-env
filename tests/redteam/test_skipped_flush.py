"""Cheat: do not flush. The fastest schedule is the illegal one."""

from __future__ import annotations

from feedmill import domain as D
from feedmill import generator as G


def test_monensin_straight_into_layer_feed(bench, caught) -> None:
    task = bench.task(["broiler_mon", "layer"])
    attempt = bench.play(task, [bench.produce("O1"), bench.produce("O2"), bench.FINISH])
    caught(attempt.verify(), D.V_L1_COCCIDIOSTAT)


def test_the_same_two_orders_with_a_flush_are_legal(bench) -> None:
    """The control: the cheat is skipping the flush, not the pair of orders."""
    task = bench.task(["broiler_mon", "layer"])
    attempt = bench.play(
        task, [bench.produce("O1"), bench.flush(), bench.produce("O2"), bench.FINISH]
    )
    assert attempt.verify().score == 1


def test_a_medicated_batch_straight_into_plain_pig_feed(bench, caught) -> None:
    task = bench.task(["pig_medicated", "pig_grower"], initial_die=4)
    attempt = bench.play(task, [bench.produce("O1"), bench.produce("O2"), bench.FINISH])
    caught(attempt.verify(), D.V_L2_ANTIMICROBIAL)


def test_dropping_one_flush_from_a_planted_plan(bench, caught) -> None:
    """Whichever flush the planted plan needed, removing it costs the run."""
    checked = 0
    for seed in range(60):
        gen = G.generate(seed, "medium")
        flushes = [i for i, a in enumerate(gen.plan.actions) if a["tool"] == D.TOOL_FLUSH]
        if not flushes:
            continue
        for index in flushes:
            actions = [a for i, a in enumerate(gen.plan.actions) if i != index]
            attempt = bench.play(gen.task, actions)
            result = attempt.verify()
            assert result.score == 0, f"{gen.task.task_id}: dropping flush {index} was free"
            checked += 1
        if checked >= 20:
            break
    assert checked >= 20
