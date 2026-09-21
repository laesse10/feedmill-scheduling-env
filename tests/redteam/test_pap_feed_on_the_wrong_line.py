"""Cheat: produce a PAP feed off its dedicated line (L4).

Regulation (EU) 2021/1372 allows porcine PAP in poultry feed and poultry PAP
in pig feed only on lines dedicated to that PAP type.
"""

from __future__ import annotations

from feedmill import domain as D


def test_pig_pap_broiler_feed_on_the_plain_line(bench, caught) -> None:
    task = bench.task(["broiler_pigpap"], pap_type=D.PAP_PIG)
    attempt = bench.play(task, [bench.produce("O1", "L1"), bench.FINISH])
    caught(attempt.verify(), D.V_L4_PAP_LINE)


def test_poultry_pap_pig_feed_on_the_plain_line(bench, caught) -> None:
    task = bench.task(["pig_poultrypap"], initial_die=4, pap_type=D.PAP_POULTRY)
    attempt = bench.play(task, [bench.produce("O1", "L1"), bench.FINISH])
    caught(attempt.verify(), D.V_L4_PAP_LINE)


def test_pig_pap_broiler_feed_on_a_poultry_line_breaks_two_rules(bench, caught) -> None:
    """Wrong PAP type, and broiler feed on a poultry line on top of it."""
    task = bench.task(["broiler_pigpap"], pap_type=D.PAP_POULTRY)
    attempt = bench.play(task, [bench.produce("O1", "L2"), bench.FINISH])
    caught(attempt.verify(), D.V_L4_PAP_LINE, D.V_L5_INTRA_SPECIES)


def test_the_dedicated_line_is_accepted(bench) -> None:
    task = bench.task(["broiler_pigpap"], pap_type=D.PAP_PIG)
    attempt = bench.play(task, [bench.produce("O1", "L2"), bench.FINISH])
    assert attempt.verify().score == 1
