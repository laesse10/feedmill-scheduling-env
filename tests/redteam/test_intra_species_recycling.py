"""Cheat: feed a species its own protein (L5).

Regulation (EC) No 1069/2009 Article 11(1)(a). Modelled as a line
restriction with zero tolerance, so even a clean line of the wrong type is
a violation.
"""

from __future__ import annotations

from feedmill import domain as D
import pytest


@pytest.mark.parametrize("feed,die", [("pig_grower", 4), ("piglet_starter", 3), ("pig_medicated", 4)])
def test_pig_feed_on_a_pig_pap_line(bench, caught, feed: str, die: int) -> None:
    task = bench.task([feed], initial_die=die, pap_type=D.PAP_PIG)
    attempt = bench.play(task, [bench.produce("O1", "L2"), bench.FINISH])
    caught(attempt.verify(), D.V_L5_INTRA_SPECIES)


@pytest.mark.parametrize("feed", ["layer", "broiler_withdrawal", "broiler_mon"])
def test_poultry_feed_on_a_poultry_pap_line(bench, caught, feed: str) -> None:
    task = bench.task([feed], pap_type=D.PAP_POULTRY)
    attempt = bench.play(task, [bench.produce("O1", "L2"), bench.FINISH])
    caught(attempt.verify(), D.V_L5_INTRA_SPECIES)


def test_the_cross_derogation_is_still_allowed(bench) -> None:
    """Poultry feed on a pig line is the point of the derogation, not a cheat."""
    task = bench.task(["layer"], pap_type=D.PAP_PIG)
    attempt = bench.play(task, [bench.produce("O1", "L2"), bench.FINISH])
    assert attempt.verify().score == 1
