"""Generator tests: determinism, difficulty shape, and the planted solution.

The planted solution is checked with ``generator.plan_violations``, which
re-simulates an action list from scratch instead of reusing the scheduling
code. ``tests/test_verifier_required.py`` checks the same plans a second time
with the independent verifier.
"""

from __future__ import annotations

import pytest

from feedmill import domain as D
from feedmill import generator as G

DIFFICULTIES = ("easy", "medium", "hard")
SEEDS = range(0, 100)


@pytest.fixture(scope="module")
def generated() -> dict[str, list[G.GeneratedTask]]:
    return {d: [G.generate(seed, d) for seed in SEEDS] for d in DIFFICULTIES}


# -- determinism -----------------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_same_seed_gives_the_same_task_and_plan(difficulty: str) -> None:
    for seed in (0, 7, 42, 999, 10000):
        a = G.generate(seed, difficulty)
        b = G.generate(seed, difficulty)
        assert a.task.to_json() == b.task.to_json()
        assert a.task.task_hash == b.task.task_hash
        assert a.plan.actions == b.plan.actions


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_different_seeds_give_different_tasks(
    difficulty: str, generated: dict[str, list[G.GeneratedTask]]
) -> None:
    hashes = {g.task.task_hash for g in generated[difficulty]}
    assert len(hashes) == len(SEEDS)


def test_difficulties_are_disjoint() -> None:
    hashes = {G.generate(0, d).task.task_hash for d in DIFFICULTIES}
    assert len(hashes) == len(DIFFICULTIES)


def test_task_json_round_trip() -> None:
    for difficulty in DIFFICULTIES:
        task = G.generate(3, difficulty).task
        assert D.Task.from_json(task.to_json()) == task
        assert D.Task.from_json(task.to_json()).task_hash == task.task_hash


# -- difficulty shape (SPEC section 11) ------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_difficulty_shape(difficulty: str, generated: dict[str, list[G.GeneratedTask]]) -> None:
    cfg = G.DIFFICULTIES[difficulty]
    for gen in generated[difficulty]:
        task = gen.task
        assert task.difficulty == difficulty
        assert len(task.lines) == cfg.n_lines
        assert cfg.n_orders[0] <= len(task.orders) <= cfg.n_orders[1]
        assert cfg.n_house_rules[0] <= len(task.house_rules) <= cfg.n_house_rules[1]
        for order in task.orders:
            assert cfg.tonnes[0] <= order.tonnes <= cfg.tonnes[1]
            assert order.feed in task.feed_map
        assert task.lines[0].pap_type == D.PAP_NONE
        assert {r.id for r in task.house_rules} <= set(D.HOUSE_RULE_TEMPLATES)
        assert len({r.id for r in task.house_rules}) == len(task.house_rules)


@pytest.mark.parametrize("difficulty", ("easy", "medium"))
def test_single_line_tasks_have_no_pap_feeds(
    difficulty: str, generated: dict[str, list[G.GeneratedTask]]
) -> None:
    for gen in generated[difficulty]:
        assert len(gen.task.lines) == 1
        for order in gen.task.orders:
            assert order.feed not in D.PAP_FEED_IDS


def test_hard_tasks_have_a_pap_line_and_at_least_one_pap_order(
    generated: dict[str, list[G.GeneratedTask]]
) -> None:
    seen_types = set()
    for gen in generated["hard"]:
        task = gen.task
        pap_lines = [ln for ln in task.lines if ln.pap_type != D.PAP_NONE]
        assert len(pap_lines) == 1
        pap_type = pap_lines[0].pap_type
        seen_types.add(pap_type)
        pap_orders = [o for o in task.orders if o.feed in D.PAP_FEED_IDS]
        assert pap_orders, "a hard task should exercise the PAP rules"
        for order in pap_orders:
            feed = task.feed_map[order.feed]
            assert D.PAP_SUBSTANCE_LINE[feed.pap_substance] == pap_type
    assert seen_types == {D.PAP_PIG, D.PAP_POULTRY}


def test_lines_start_clean(generated: dict[str, list[G.GeneratedTask]]) -> None:
    for difficulty in DIFFICULTIES:
        for gen in generated[difficulty]:
            for line in gen.task.lines:
                assert line.initial_concentrations == D.zero_concentrations()
                assert line.initial_die_mm in D.DIE_SIZES


# -- the planted solution --------------------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_planted_solution_obeys_every_rule(
    difficulty: str, generated: dict[str, list[G.GeneratedTask]]
) -> None:
    for gen in generated[difficulty]:
        violations = G.plan_violations(gen.task, gen.plan.actions)
        assert violations == [], f"{gen.task.task_id}: {violations}"


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_planted_actions_are_well_formed(
    difficulty: str, generated: dict[str, list[G.GeneratedTask]]
) -> None:
    for gen in generated[difficulty]:
        actions = gen.plan.actions
        line_ids = set(gen.task.line_map)
        assert actions[-1] == {"tool": D.TOOL_FINISH}
        assert len(actions) <= gen.task.max_steps
        produced = [a["order"] for a in actions if a["tool"] == D.TOOL_PRODUCE]
        assert sorted(produced) == sorted(o.id for o in gen.task.orders)
        for action in actions[:-1]:
            assert action["tool"] in D.TOOLS
            assert action["line"] in line_ids
            if action["tool"] == D.TOOL_WAIT:
                assert action["minutes"] > 0
            if action["tool"] == D.TOOL_CHANGE_DIE:
                assert action["die_mm"] in D.DIE_SIZES


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_due_times_follow_from_the_planted_completion_times(
    difficulty: str, generated: dict[str, list[G.GeneratedTask]]
) -> None:
    slack = G.DIFFICULTIES[difficulty].slack
    for gen in generated[difficulty]:
        for order in gen.task.orders:
            completion = gen.plan.completion[order.id]
            assert completion <= order.due, "the planted plan must meet every due time"
            # due = completion * slack + jitter, jitter >= 0
            assert order.due >= int(round(completion * slack))
            assert order.due - int(round(completion * slack)) <= G.MAX_JITTER_MIN
            assert order.due <= gen.task.horizon_min


def test_planted_plan_uses_the_line_assignment_rules(
    generated: dict[str, list[G.GeneratedTask]]
) -> None:
    """Every produce action of the planted plan sits on a line that L3-L5 allow."""
    for gen in generated["hard"]:
        task = gen.task
        for action in gen.plan.actions:
            if action["tool"] != D.TOOL_PRODUCE:
                continue
            feed = task.feed_map[task.order_map[action["order"]].feed]
            line = task.line_map[action["line"]]
            assert not (feed.ruminant and line.pap_type != D.PAP_NONE)
            assert D.SPECIES_PAP_GROUP[feed.species] != line.pap_type
            if feed.pap_substance:
                assert line.pap_type == D.PAP_SUBSTANCE_LINE[feed.pap_substance]


# -- house rules -----------------------------------------------------------


def test_house_rules_have_text_and_valid_parameters(
    generated: dict[str, list[G.GeneratedTask]]
) -> None:
    for difficulty in DIFFICULTIES:
        for gen in generated[difficulty]:
            task = gen.task
            for rule in task.house_rules:
                assert rule.text.strip()
                if rule.id == D.H1_CUSTOMER_FLUSH:
                    assert rule.params["customer"] in {o.customer for o in task.orders}
                if rule.id == D.H2_SEQUENCE_BAN:
                    assert rule.params["feed_a"] in task.feed_map
                    assert rule.params["feed_b"] in task.feed_map
                    assert rule.params["feed_a"] != rule.params["feed_b"]
                if rule.id == D.H3_SLOW_DIE:
                    assert rule.params["line"] in task.line_map
                    assert rule.params["minutes"] == D.SLOW_DIE_CHANGE_MINUTES
                if rule.id == D.H5_LAB_HOLD:
                    assert rule.params["minutes"] == D.LAB_HOLD_MINUTES


def test_generator_rejects_unknown_difficulty() -> None:
    with pytest.raises(ValueError):
        G.generate(0, "impossible")


# -- the independent plan checker catches real mistakes --------------------


def test_plan_violations_flags_a_missing_flush() -> None:
    """Dropping a flush from a planted plan must be detected."""
    found = False
    for seed in range(200):
        gen = G.generate(seed, "medium")
        flushes = [i for i, a in enumerate(gen.plan.actions) if a["tool"] == D.TOOL_FLUSH]
        if not flushes:
            continue
        broken = [a for i, a in enumerate(gen.plan.actions) if i != flushes[0]]
        assert G.plan_violations(gen.task, broken), f"{gen.task.task_id}: removing a flush was not noticed"
        found = True
        break
    assert found, "no medium task needed a flush"


def test_plan_violations_flags_a_missing_order() -> None:
    gen = G.generate(1, "easy")
    broken = [a for a in gen.plan.actions if a["tool"] != D.TOOL_PRODUCE or a["order"] != "O1"]
    assert D.V_MISSING_ORDER in G.plan_violations(gen.task, broken)


def test_plan_violations_flags_a_late_order() -> None:
    gen = G.generate(2, "medium")
    task = gen.task
    tight = D.Task(
        **{
            **{f: getattr(task, f) for f in task.__dataclass_fields__},
            "orders": tuple(D.Order(o.id, o.customer, o.feed, o.tonnes, 1) for o in task.orders),
        }
    )
    assert D.V_LATE in G.plan_violations(tight, gen.plan.actions)


def test_plan_violations_flags_a_duplicate_production() -> None:
    gen = G.generate(4, "easy")
    doubled = list(gen.plan.actions)
    first_produce = next(a for a in doubled if a["tool"] == D.TOOL_PRODUCE)
    doubled.insert(len(doubled) - 1, dict(first_produce))
    assert D.V_DUPLICATE_ORDER in G.plan_violations(gen.task, doubled)


def test_plan_violations_flags_actions_after_finish() -> None:
    gen = G.generate(4, "easy")
    trailing = list(gen.plan.actions) + [{"tool": D.TOOL_FLUSH, "line": "L1"}]
    assert D.V_INVALID_ACTION in G.plan_violations(gen.task, trailing)


def test_plan_violations_flags_a_wrong_die() -> None:
    """Producing without the die change the feed needs breaks L6."""
    found = False
    for seed in range(50):
        gen = G.generate(seed, "medium")
        changes = [i for i, a in enumerate(gen.plan.actions) if a["tool"] == D.TOOL_CHANGE_DIE]
        if not changes:
            continue
        broken = [a for i, a in enumerate(gen.plan.actions) if i != changes[0]]
        assert D.V_L6_WRONG_DIE in G.plan_violations(gen.task, broken)
        found = True
        break
    assert found, "no medium task needed a die change"


def test_determinism_survives_a_fresh_interpreter() -> None:
    """Same seed -> same task_hash in another process, whatever PYTHONHASHSEED is."""
    import os
    import subprocess
    import sys

    script = (
        "from feedmill import generator as G;"
        "print(' '.join(G.generate(s, d).task.task_hash"
        " for s in (0, 7, 10000) for d in ('easy', 'medium', 'hard')))"
    )
    outputs = set()
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        outputs.add(out.stdout.strip())
    assert len(outputs) == 1
