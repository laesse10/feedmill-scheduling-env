"""Domain-level unit tests: the carry-over model of SPEC section 3."""

from __future__ import annotations

import pytest

from feedmill import domain as D

R = D.CARRY_OVER_RATE


def _produce(previous: dict[str, float], feed_id: str) -> dict[str, float]:
    return D.next_concentrations(previous, D.FEED_BY_ID[feed_id].substances, R)


def _flush(previous: dict[str, float]) -> dict[str, float]:
    return D.next_concentrations(previous, (), R)


def _limit(feed_id: str) -> float:
    """L1 limit for monensin in a feed that does not contain it."""
    return D.COCCIDIOSTAT_LIMIT[D.FEED_BY_ID[feed_id].carry_over_class]


def test_carry_over_rate_is_two_percent() -> None:
    assert R == 0.02


def test_clean_line_has_no_residue() -> None:
    assert D.zero_concentrations() == {s: 0.0 for s in D.SUBSTANCES}


def test_target_feed_carries_its_own_substance_at_full_level() -> None:
    conc = _produce(D.zero_concentrations(), "broiler_mon")
    assert conc[D.MONENSIN] == pytest.approx(1.0)


# -- the three examples of SPEC section 3 ----------------------------------


def test_example_1_monensin_then_layer_violates_the_one_percent_limit() -> None:
    """broiler_mon -> layer: monensin in layer = 2 % > 1 % limit -> violation."""
    after_mon = _produce(D.zero_concentrations(), "broiler_mon")
    layer = _produce(after_mon, "layer")

    assert layer[D.MONENSIN] == pytest.approx(0.02)
    assert _limit("layer") == 0.01
    assert layer[D.MONENSIN] > _limit("layer") + D.LIMIT_TOL


def test_example_2_flush_between_makes_it_legal() -> None:
    """broiler_mon -> flush -> layer: 0.04 % -> ok."""
    after_mon = _produce(D.zero_concentrations(), "broiler_mon")
    after_flush = _flush(after_mon)
    layer = _produce(after_flush, "layer")

    assert after_flush[D.MONENSIN] == pytest.approx(0.02)
    assert layer[D.MONENSIN] == pytest.approx(0.0004)
    assert layer[D.MONENSIN] <= _limit("layer") + D.LIMIT_TOL


def test_example_3_less_sensitive_batch_acts_as_a_natural_flush() -> None:
    """broiler_mon -> pig_grower -> layer: 2 % <= 3 %, then 0.04 % -> both ok."""
    after_mon = _produce(D.zero_concentrations(), "broiler_mon")
    grower = _produce(after_mon, "pig_grower")
    layer = _produce(grower, "layer")

    assert grower[D.MONENSIN] == pytest.approx(0.02)
    assert _limit("pig_grower") == 0.03
    assert grower[D.MONENSIN] <= _limit("pig_grower") + D.LIMIT_TOL

    assert layer[D.MONENSIN] == pytest.approx(0.0004)
    assert layer[D.MONENSIN] <= _limit("layer") + D.LIMIT_TOL


# -- other domain invariants ----------------------------------------------


def test_one_flush_always_clears_any_residue_below_the_strictest_limit() -> None:
    """The planted plan relies on this: a single flush repairs any carry-over."""
    worst = {s: 1.0 + R for s in D.SUBSTANCES}  # highest reachable concentration
    after_flush = _flush(worst)
    clean_batch = D.next_concentrations(after_flush, (), R)
    strictest = min(min(D.COCCIDIOSTAT_LIMIT.values()), D.ANTIMICROBIAL_LIMIT)
    assert max(clean_batch.values()) <= strictest + D.LIMIT_TOL


def test_production_minutes_rounds_up() -> None:
    assert D.production_minutes(20, 20.0) == 60
    assert D.production_minutes(25, 20.0) == 75
    assert D.production_minutes(10.5, 20.0) == 32  # 31.5 -> 32


def test_feed_catalog_matches_spec_table() -> None:
    expected = {
        "broiler_mon": ("broiler", False, (D.MONENSIN,), D.CLASS_TARGET, 3),
        "broiler_withdrawal": ("broiler", False, (), D.CLASS_SENSITIVE, 3),
        "broiler_pigpap": ("broiler", False, (D.PAP_PIG_SUBSTANCE,), D.CLASS_SENSITIVE, 3),
        "layer": ("laying_hen", False, (), D.CLASS_SENSITIVE, 3),
        "piglet_starter": ("pig", False, (D.COPPER_HIGH,), D.CLASS_LESS_SENSITIVE, 3),
        "pig_grower": ("pig", False, (), D.CLASS_LESS_SENSITIVE, 4),
        "pig_medicated": ("pig", False, (D.ANTIMICROBIAL,), D.CLASS_LESS_SENSITIVE, 4),
        "pig_poultrypap": ("pig", False, (D.PAP_POULTRY_SUBSTANCE,), D.CLASS_LESS_SENSITIVE, 4),
        "dairy": ("cattle", True, (), D.CLASS_SENSITIVE, 6),
        "sheep": ("sheep", True, (), D.CLASS_LESS_SENSITIVE, 4),
        "horse": ("horse", False, (), D.CLASS_SENSITIVE, 6),
    }
    assert set(D.FEED_BY_ID) == set(expected)
    for feed_id, (species, ruminant, subs, klass, die) in expected.items():
        feed = D.FEED_BY_ID[feed_id]
        assert (feed.species, feed.ruminant, feed.substances, feed.carry_over_class, feed.die_mm) == (
            species,
            ruminant,
            subs,
            klass,
            die,
        )


def test_canonical_json_is_sorted_and_compact() -> None:
    assert D.canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
