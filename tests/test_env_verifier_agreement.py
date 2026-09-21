"""Differential test: the verifier's replay must match what the env did.

The verifier implements the physics and the rules a second time, from SPEC,
without importing the environment. That only helps if the two agree, so this
file drives random episodes -- valid actions, illegal schedules and garbage
mixed together -- and compares the two implementations on every episode:

1. the verdict on every single action (possible or not),
2. the reconstructed clocks, dies, concentrations and completion times,
3. the violation codes, against the generator's independent plan checker.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import pytest

from feedmill import domain as D
from feedmill import generator as G
from feedmill.env import FeedMillEnv
from feedmill.verifier import replay, verify

DIFFICULTIES = ("easy", "medium", "hard")
SEEDS_PER_DIFFICULTY = 25

GARBAGE: tuple[Any, ...] = (
    {"tool": "jump", "line": "L1"},
    {"tool": "produce", "line": "LX", "order": "O1"},
    {"tool": "produce", "line": "L1", "order": "O404"},
    {"tool": "wait", "line": "L1", "minutes": 0},
    {"tool": "wait", "line": "L1", "minutes": True},
    {"tool": "change_die", "line": "L1", "die_mm": 5},
    {"tool": "flush", "line": "L1", "extra": "x"},
    "not even a dict",
    42,
    None,
)


def _random_action(rng: random.Random, task: D.Task) -> Any:
    """A mix an agent could plausibly emit, plus outright nonsense."""
    roll = rng.random()
    line = rng.choice(task.lines)
    if roll < 0.08:
        return rng.choice(GARBAGE)
    if roll < 0.20:
        return {"tool": D.TOOL_FLUSH, "line": line.id}
    if roll < 0.30:
        return {"tool": D.TOOL_WAIT, "line": line.id, "minutes": rng.randint(1, 90)}
    if roll < 0.45:
        return {"tool": D.TOOL_CHANGE_DIE, "line": line.id, "die_mm": rng.choice(D.DIE_SIZES)}
    return {"tool": D.TOOL_PRODUCE, "line": line.id, "order": rng.choice(task.orders).id}


def _random_episode(task: D.Task, rng: random.Random, steps: int = 60) -> FeedMillEnv:
    env = FeedMillEnv()
    env.reset(task=task)
    for _ in range(steps):
        env.step(_random_action(rng, task))
    if rng.random() < 0.7:
        env.step({"tool": D.TOOL_FINISH})
    return env


def _awkward(task: D.Task) -> D.Task:
    """The same task on a mill whose batches do not last a whole number of minutes.

    Every generated task has integer production times (integer tonnes at
    20 t/h), so rounding is invisible there. A real mill from a task file
    (SPEC section 13) need not be that tidy, and the environment and the
    verifier round with separate code.
    """
    return dataclasses.replace(
        task,
        lines=tuple(dataclasses.replace(ln, rate_t_per_h=22.0) for ln in task.lines),
        orders=tuple(dataclasses.replace(o, tonnes=o.tonnes + 0.5) for o in task.orders),
    )


@pytest.mark.parametrize("awkward", (False, True))
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_the_replay_reproduces_the_environment_state(difficulty: str, awkward: bool) -> None:
    for seed in range(SEEDS_PER_DIFFICULTY):
        task = G.generate(seed, difficulty).task
        if awkward:
            task = _awkward(task)
        env = _random_episode(task, random.Random(f"{difficulty}|{seed}"))
        state = env.final_state()
        derived = state["derived"]
        result = replay(state)

        assert result.statuses == [e["status"] for e in state["log"]], (
            f"{task.task_id}: env and verifier disagree on which actions are possible"
        )
        assert result.completions == derived["completions"]
        assert result.batches == derived["batches"]
        assert result.finished == derived["finished"]
        for line in derived["lines"]:
            recomputed = result.lines[line["id"]]
            assert recomputed.clock == line["clock"]
            assert recomputed.die_mm == line["die_mm"]
            assert recomputed.concentrations == line["concentrations"]


@pytest.mark.parametrize("awkward", (False, True))
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_the_violation_codes_match_the_independent_plan_checker(
    difficulty: str, awkward: bool
) -> None:
    """Two implementations, written from SPEC separately, on the same schedule."""
    compared = 0
    for seed in range(SEEDS_PER_DIFFICULTY):
        task = G.generate(seed, difficulty).task
        if awkward:
            task = _awkward(task)
        env = _random_episode(task, random.Random(f"codes|{difficulty}|{seed}"))

        accepted = [
            e["action"] for e in env.final_state()["log"] if e["status"] == D.STATUS_OK
        ]
        clean = FeedMillEnv()
        clean.reset(task=task)
        for action in accepted:
            _, _, _, _, info = clean.step(action)
            assert info["status"] == D.STATUS_OK, "replaying accepted actions must stay accepted"

        result = verify(clean.final_state(), clean.secret_key)
        expected = G.plan_violations(task, accepted)
        assert set(result.violations) == set(expected), (
            f"{task.task_id}: verifier {sorted(result.violations)} vs "
            f"plan checker {sorted(expected)}"
        )
        assert result.score == (0 if expected else 1)
        compared += 1
    assert compared == SEEDS_PER_DIFFICULTY


def test_random_episodes_do_break_every_comparable_rule() -> None:
    """Guard against the differential test comparing two empty lists forever.

    Four codes cannot appear in a log where every action was accepted:
    L6_WRONG_DIE, DUPLICATE_ORDER and INVALID_ACTION are refused by the
    environment before they reach the log, and TAMPERED_LOG is not a rule.
    Every other code is compared between the two implementations here.
    """
    seen: set[str] = set()
    for difficulty in DIFFICULTIES:
        for seed in range(SEEDS_PER_DIFFICULTY):
            task = G.generate(seed, difficulty).task
            env = _random_episode(task, random.Random(f"codes|{difficulty}|{seed}"))
            accepted = [
                e["action"] for e in env.final_state()["log"] if e["status"] == D.STATUS_OK
            ]
            seen.update(G.plan_violations(task, accepted))
    assert seen == {
        D.V_L1_COCCIDIOSTAT,
        D.V_L2_ANTIMICROBIAL,
        D.V_L3_RUMINANT_BAN,
        D.V_L4_PAP_LINE,
        D.V_L5_INTRA_SPECIES,
        D.V_H1_CUSTOMER_FLUSH,
        D.V_H2_SEQUENCE_BAN,
        D.V_H4_DIE_CHANGE_TIME,
        D.V_H5_LAB_HOLD,
        D.V_H6_COPPER_SHEEP_FLUSH,
        D.V_LATE,
        D.V_MISSING_ORDER,
        D.V_NOT_FINISHED,
    }, sorted(seen)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_a_planted_plan_replayed_from_the_log_matches_the_plan(difficulty: str) -> None:
    for seed in range(10):
        gen = G.generate(seed, difficulty)
        env = FeedMillEnv()
        env.reset(task=gen.task)
        for action in gen.plan.actions:
            env.step(action)
        result = replay(env.final_state())
        assert {k: v["end"] for k, v in result.completions.items()} == gen.plan.completion
        assert result.violations == []
