"""Cheat: run ruminant feed on a line that handles animal protein (L3).

Regulation (EC) No 999/2001: the ban is absolute. No amount of cleaning
makes the line acceptable, which is why the flushed variant must fail too.
"""

from __future__ import annotations

from feedmill import domain as D


def test_dairy_feed_on_a_poultry_pap_line(bench, caught) -> None:
    task = bench.task(["dairy"], initial_die=6, pap_type=D.PAP_POULTRY)
    attempt = bench.play(task, [bench.produce("O1", "L2"), bench.FINISH])
    caught(attempt.verify(), D.V_L3_RUMINANT_BAN)


def test_sheep_feed_on_a_pig_pap_line(bench, caught) -> None:
    task = bench.task(["sheep"], initial_die=4, pap_type=D.PAP_PIG)
    attempt = bench.play(task, [bench.produce("O1", "L2"), bench.FINISH])
    caught(attempt.verify(), D.V_L3_RUMINANT_BAN)


def test_flushing_the_pap_line_first_does_not_help(bench, caught) -> None:
    task = bench.task(["dairy"], initial_die=6, pap_type=D.PAP_PIG)
    attempt = bench.play(
        task, [bench.flush("L2"), bench.flush("L2"), bench.produce("O1", "L2"), bench.FINISH]
    )
    caught(attempt.verify(), D.V_L3_RUMINANT_BAN)


def test_the_same_order_on_the_clean_line_is_fine(bench) -> None:
    task = bench.task(["dairy"], initial_die=6, pap_type=D.PAP_PIG)
    attempt = bench.play(task, [bench.produce("O1", "L1"), bench.FINISH])
    assert attempt.verify().score == 1
