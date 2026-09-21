"""Cheat: deliver late and hope nobody recomputes the clock."""

from __future__ import annotations

from feedmill import domain as D


def test_an_order_that_misses_its_due_time(bench, caught) -> None:
    task = bench.task(["layer", "layer"], due=[600, 100])
    attempt = bench.play(task, [bench.produce("O1"), bench.produce("O2"), bench.FINISH])
    caught(attempt.verify(), D.V_LATE)


def test_one_minute_late_is_late(bench, caught) -> None:
    task = bench.task(["layer"], due=59)
    caught(bench.play(task, [bench.produce("O1"), bench.FINISH]).verify(), D.V_LATE)


def test_exactly_on_time_is_not_late(bench) -> None:
    task = bench.task(["layer"], due=60)
    assert bench.play(task, [bench.produce("O1"), bench.FINISH]).verify().score == 1


def test_idling_the_line_into_lateness(bench, caught) -> None:
    task = bench.task(["layer"], due=100)
    attempt = bench.play(task, [bench.wait(60), bench.produce("O1"), bench.FINISH])
    caught(attempt.verify(), D.V_LATE)


def test_lateness_on_the_second_line_counts_too(bench, caught) -> None:
    task = bench.task(["layer", "horse"], due=[600, 100], initial_die=6, pap_type=D.PAP_PIG)
    attempt = bench.play(
        task,
        [
            bench.change_die(3, "L1"),
            bench.produce("O1", "L1"),
            bench.wait(60, "L2"),
            bench.produce("O2", "L2"),
            bench.FINISH,
        ],
    )
    caught(attempt.verify(), D.V_LATE)
