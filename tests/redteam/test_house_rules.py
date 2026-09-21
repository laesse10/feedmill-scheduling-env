"""Cheat: break the company's own rules, one at a time.

H3 has no violation code: it makes a die change take 90 minutes instead of
45 on one line, so it cannot be broken, only paid for. The last two tests
check that the verifier makes an agent pay it.
"""

from __future__ import annotations

from feedmill import domain as D


def test_h1_an_order_that_is_not_preceded_by_a_flush(bench, caught) -> None:
    rule = D.HouseRule(
        D.H1_CUSTOMER_FLUSH, {"customer": "Zuber Farms"}, "orders of Zuber Farms follow a flush"
    )
    task = bench.task(["layer"], customers=["Zuber Farms"], house_rules=[rule])
    attempt = bench.play(task, [bench.produce("O1"), bench.FINISH])
    caught(attempt.verify(), D.V_H1_CUSTOMER_FLUSH)


def test_h2_the_banned_pair_even_with_a_flush_between(bench, caught) -> None:
    rule = D.HouseRule(
        D.H2_SEQUENCE_BAN,
        {"feed_a": "layer", "feed_b": "horse"},
        "horse never directly after layer",
    )
    task = bench.task(["layer", "horse"], house_rules=[rule])
    for middle in ([], [bench.flush()]):
        attempt = bench.play(
            task,
            [bench.produce("O1"), bench.change_die(6), *middle, bench.produce("O2"), bench.FINISH],
        )
        caught(attempt.verify(), D.V_H2_SEQUENCE_BAN)


def test_h2_is_satisfied_by_putting_a_batch_in_between(bench) -> None:
    rule = D.HouseRule(
        D.H2_SEQUENCE_BAN,
        {"feed_a": "layer", "feed_b": "horse"},
        "horse never directly after layer",
    )
    task = bench.task(["layer", "horse", "broiler_withdrawal"], house_rules=[rule])
    attempt = bench.play(
        task,
        [
            bench.produce("O1"),
            bench.produce("O3"),
            bench.change_die(6),
            bench.produce("O2"),
            bench.FINISH,
        ],
    )
    assert attempt.verify().score == 1


def test_h4_a_die_change_started_after_22_00(bench, caught) -> None:
    rule = D.HouseRule(
        D.H4_DIE_CHANGE_DAY_ONLY, {"latest_start": 960}, "die changes only between 06:00 and 22:00"
    )
    task = bench.task(["layer", "pig_grower"], due=1440, house_rules=[rule])
    attempt = bench.play(
        task,
        [
            bench.produce("O1"),
            bench.wait(950),
            bench.change_die(4),
            bench.produce("O2"),
            bench.FINISH,
        ],
    )
    caught(attempt.verify(), D.V_H4_DIE_CHANGE_TIME)


def test_h5_a_sensitive_feed_before_the_lab_released_the_line(bench, caught) -> None:
    rule = D.HouseRule(D.H5_LAB_HOLD, {"minutes": 120}, "120 minutes after a medicated batch")
    task = bench.task(["pig_medicated", "layer"], initial_die=4, house_rules=[rule])
    plan = [
        bench.produce("O1"),
        bench.change_die(3),
        bench.flush(),
        bench.produce("O2"),
        bench.FINISH,
    ]
    caught(bench.play(task, plan).verify(), D.V_H5_LAB_HOLD)

    # the same plan, waiting out the hold, is clean
    patient = plan[:2] + [bench.wait(60)] + plan[2:]
    assert bench.play(task, patient).verify().score == 1


def test_h6_sheep_feed_after_piglet_feed_without_a_flush(bench, caught) -> None:
    rule = D.HouseRule(
        D.H6_COPPER_SHEEP_FLUSH,
        {"after_feed": "piglet_starter", "before_feed": "sheep"},
        "flush after piglet_starter before sheep",
    )
    task = bench.task(["piglet_starter", "sheep"], house_rules=[rule])
    plan = [bench.produce("O1"), bench.change_die(4), bench.produce("O2"), bench.FINISH]
    caught(bench.play(task, plan).verify(), D.V_H6_COPPER_SHEEP_FLUSH)

    flushed = plan[:2] + [bench.flush()] + plan[2:]
    assert bench.play(task, flushed).verify().score == 1


def test_h3_cannot_be_broken_only_paid_for(bench, caught) -> None:
    """The slow press is physics: the verifier advances the clock by 90."""
    rule = D.HouseRule(D.H3_SLOW_DIE, {"line": "L1", "minutes": 90}, "L1 has an old press")
    plan = [bench.produce("O1"), bench.change_die(4), bench.produce("O2"), bench.FINISH]

    fast = bench.task(["layer", "pig_grower"], due=[600, 180])
    assert bench.play(fast, plan).verify().score == 1, "45 minutes: O2 ends at 165"

    slow = bench.task(["layer", "pig_grower"], due=[600, 180], house_rules=[rule])
    caught(bench.play(slow, plan).verify(), D.V_LATE)  # 90 minutes: O2 ends at 210
