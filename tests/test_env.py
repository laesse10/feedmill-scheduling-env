"""Environment tests: physics, invalid actions and the tamper-evident log.

The environment enforces only what is physically possible (SPEC section 6).
Legal and house rule violations must stay possible here -- they are the
verifier's job -- so these tests check that breaking a rule is *allowed*.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest

from feedmill import domain as D
from feedmill import generator as G
from feedmill.env import FeedMillEnv


def _env(seed: int = 0, difficulty: str = "easy", **kwargs: Any) -> tuple[FeedMillEnv, dict]:
    env = FeedMillEnv(**kwargs)
    obs = env.reset(seed, difficulty)
    return env, obs


def _line(obs: dict, line_id: str = "L1") -> dict:
    return next(ln for ln in obs["lines"] if ln["id"] == line_id)


# -- reset and observation (SPEC section 7) --------------------------------


def test_reset_returns_a_complete_observation() -> None:
    env, obs = _env(0, "medium")
    for key in (
        "task_id",
        "difficulty",
        "step",
        "max_steps",
        "horizon_min",
        "carry_over_rate",
        "lines",
        "feeds",
        "open_orders",
        "completed_orders",
        "legal_rules",
        "house_rules",
    ):
        assert key in obs, key
    assert obs["step"] == 0
    assert obs["carry_over_rate"] == D.CARRY_OVER_RATE
    assert len(obs["open_orders"]) == len(env.task.orders)
    assert obs["completed_orders"] == []
    line = _line(obs)
    assert line["clock"] == 0
    assert line["concentrations"] == D.zero_concentrations()
    assert line["die_mm"] == env.task.lines[0].initial_die_mm


def test_rules_come_structured_and_as_text() -> None:
    env, obs = _env(0, "hard")
    assert {r["id"] for r in obs["legal_rules"]} == {"L1", "L2", "L3", "L4", "L5", "L6"}
    for rule in obs["legal_rules"]:
        assert rule["text"].strip() and rule["params"] is not None and rule["basis"].strip()
    for rule in obs["house_rules"]:
        assert rule["id"] in D.HOUSE_RULE_TEMPLATES
        assert rule["text"].strip()
    assert [r["id"] for r in obs["house_rules"]] == [r.id for r in env.task.house_rules]


def test_observation_never_contains_the_solution_or_the_key() -> None:
    env, obs = _env(3, "hard")
    text = json.dumps(obs)
    assert env.secret_key.hex() not in text
    assert "planted" not in text and "solution" not in text
    # the planted plan is not reachable from the environment at all
    assert not hasattr(env, "plan")


def test_same_seed_gives_the_same_task() -> None:
    a, _ = _env(11, "medium")
    b, _ = _env(11, "medium")
    assert a.task.task_hash == b.task.task_hash
    assert a.secret_key != b.secret_key, "the key is per instance (SPEC section 8)"


def test_reset_accepts_an_external_task() -> None:
    task = G.generate(5, "easy").task
    env = FeedMillEnv()
    obs = env.reset(task=task)
    assert obs["task_id"] == task.task_id
    assert env.task == task


# -- physics (SPEC sections 2 and 3) ---------------------------------------


def test_produce_advances_the_clock_and_carries_substances_over() -> None:
    """conc = (1.0 if the batch contains the substance) + r * previous batch."""
    task = _monensin_task(feeds=("broiler_mon", "broiler_mon", "layer"))
    env = FeedMillEnv()
    env.reset(task=task)

    obs, reward, terminated, truncated, info = env.step(
        {"tool": "produce", "line": "L1", "order": "O1"}
    )
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info["status"] == D.STATUS_OK
    line = _line(obs)
    assert line["clock"] == D.production_minutes(20, 20.0) == 60
    assert line["concentrations"][D.MONENSIN] == pytest.approx(1.0)
    assert [o["id"] for o in obs["completed_orders"]] == ["O1"]
    assert "O1" not in [o["id"] for o in obs["open_orders"]]

    # a second medicated batch carries its own monensin plus 2 % of the first
    obs, *_ = env.step({"tool": "produce", "line": "L1", "order": "O2"})
    line = _line(obs)
    assert line["concentrations"][D.MONENSIN] == pytest.approx(1.02)
    assert line["clock"] == 2 * D.production_minutes(20, 20.0)

    # a feed without monensin only picks up the carry-over
    obs, *_ = env.step({"tool": "produce", "line": "L1", "order": "O3"})
    assert _line(obs)["concentrations"][D.MONENSIN] == pytest.approx(0.0204)


def test_flush_takes_fifteen_minutes_and_dilutes_by_the_carry_over_rate() -> None:
    env = FeedMillEnv()
    env.reset(task=_monensin_task())
    env.step({"tool": "produce", "line": "L1", "order": "O1"})
    obs, *_ = env.step({"tool": "flush", "line": "L1"})
    line = _line(obs)
    assert line["clock"] == 60 + D.FLUSH_MINUTES
    assert line["concentrations"][D.MONENSIN] == pytest.approx(0.02)
    assert line["last_batch"]["kind"] == D.TOOL_FLUSH


def test_change_die_takes_45_minutes_and_90_under_house_rule_h3() -> None:
    env = FeedMillEnv()
    env.reset(task=_monensin_task())
    obs, *_ = env.step({"tool": "change_die", "line": "L1", "die_mm": 4})
    assert _line(obs)["clock"] == D.DIE_CHANGE_MINUTES
    assert _line(obs)["die_mm"] == 4

    slow = D.HouseRule("H3", {"line": "L1", "minutes": D.SLOW_DIE_CHANGE_MINUTES}, "slow press")
    env = FeedMillEnv()
    env.reset(task=_monensin_task(house_rules=(slow,)))
    obs, *_ = env.step({"tool": "change_die", "line": "L1", "die_mm": 4})
    assert _line(obs)["clock"] == D.SLOW_DIE_CHANGE_MINUTES
    assert _line(obs)["die_change_minutes"] == D.SLOW_DIE_CHANGE_MINUTES


def test_wait_advances_only_its_own_line() -> None:
    env, _ = _env(0, "hard")
    obs, *_ = env.step({"tool": "wait", "line": "L2", "minutes": 30})
    assert _line(obs, "L2")["clock"] == 30
    assert _line(obs, "L1")["clock"] == 0


def test_the_environment_allows_illegal_schedules() -> None:
    """monensin -> layer with no flush is 2 % carry-over: physically possible."""
    env = FeedMillEnv()
    env.reset(task=_monensin_task(feeds=("broiler_mon", "layer")))
    env.step({"tool": "produce", "line": "L1", "order": "O1"})
    obs, _, _, _, info = env.step({"tool": "produce", "line": "L1", "order": "O2"})
    assert info["status"] == D.STATUS_OK
    assert _line(obs)["concentrations"][D.MONENSIN] == pytest.approx(0.02)


# -- invalid actions (SPEC section 6) --------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        {"tool": "fly", "line": "L1"},
        {"tool": "produce", "line": "L9", "order": "O1"},
        {"tool": "produce", "line": "L1", "order": "O99"},
        {"tool": "wait", "line": "L1", "minutes": 0},
        {"tool": "wait", "line": "L1", "minutes": -5},
        {"tool": "wait", "line": "L1", "minutes": 1.5},
        {"tool": "change_die", "line": "L1", "die_mm": 5},
        {"tool": "produce", "line": "L1"},
        {"tool": "produce", "line": "L1", "order": "O1", "extra": 1},
        {"line": "L1"},
        "flush",
        None,
        42,
    ],
)
def test_invalid_actions_are_logged_and_change_nothing(action: Any) -> None:
    env = FeedMillEnv()
    env.reset(task=_monensin_task(initial_die=3))
    before = env.observation()

    obs, reward, terminated, truncated, info = env.step(action)

    assert info["status"] == D.STATUS_INVALID
    assert info["error"]
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert env.final_state()["log"][-1]["status"] == D.STATUS_INVALID
    assert obs["step"] == before["step"] + 1, "an invalid action still costs a step"
    for key in ("lines", "open_orders", "completed_orders"):
        assert obs[key] == before[key], f"{key} changed after an invalid action"


def test_producing_with_the_wrong_die_is_physically_impossible() -> None:
    env = FeedMillEnv()
    env.reset(task=_monensin_task(initial_die=6))  # order O1 needs a 3 mm die
    _, _, _, _, info = env.step({"tool": "produce", "line": "L1", "order": "O1"})
    assert info["status"] == D.STATUS_INVALID
    assert "die" in info["error"]


def test_an_order_cannot_be_produced_twice() -> None:
    env = FeedMillEnv()
    env.reset(task=_monensin_task())
    env.step({"tool": "produce", "line": "L1", "order": "O1"})
    _, _, _, _, info = env.step({"tool": "produce", "line": "L1", "order": "O1"})
    assert info["status"] == D.STATUS_INVALID
    assert "already" in info["error"]


def test_actions_after_finish_are_invalid_but_do_not_crash() -> None:
    env = FeedMillEnv(reward_fn=lambda state, key: 1.0)
    env.reset(task=_monensin_task())
    _, reward, terminated, _, _ = env.step({"tool": "finish"})
    assert terminated and reward == 1.0

    _, reward, terminated, _, info = env.step({"tool": "flush", "line": "L1"})
    assert info["status"] == D.STATUS_INVALID
    assert terminated and reward == 0.0
    statuses = [e["status"] for e in env.final_state()["log"]]
    assert statuses == [D.STATUS_OK, D.STATUS_INVALID]


def test_the_episode_truncates_after_max_steps() -> None:
    env = FeedMillEnv(max_steps=3)
    env.reset(task=_monensin_task())
    for _ in range(2):
        _, _, terminated, truncated, _ = env.step({"tool": "wait", "line": "L1", "minutes": 1})
        assert not (terminated or truncated)
    _, _, terminated, truncated, _ = env.step({"tool": "wait", "line": "L1", "minutes": 1})
    assert truncated and not terminated


# -- the tamper-evident log (SPEC section 8) -------------------------------


def _expected_hash(key: bytes, prev_hash: str, entry: dict) -> str:
    payload = {k: v for k, v in entry.items() if k != "hash"}
    message = prev_hash + D.canonical_json(payload)
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()


def test_the_log_is_an_hmac_chain_that_starts_at_the_task_hash() -> None:
    env, _ = _env(2, "easy")
    for action in ({"tool": "flush", "line": "L1"}, {"tool": "nope"}, {"tool": "finish"}):
        env.step(action)

    state = env.final_state()
    assert state["task_hash"] == env.task.task_hash
    prev = state["task_hash"]
    for i, entry in enumerate(state["log"]):
        assert entry["i"] == i
        assert entry["prev_hash"] == prev
        assert entry["hash"] == _expected_hash(env.secret_key, prev, entry)
        prev = entry["hash"]


def test_the_chain_does_not_verify_under_a_different_key() -> None:
    env, _ = _env(2, "easy")
    env.step({"tool": "flush", "line": "L1"})
    entry = env.final_state()["log"][0]
    assert entry["hash"] != _expected_hash(b"x" * 32, entry["prev_hash"], entry)


def test_the_log_records_the_action_as_submitted() -> None:
    env, _ = _env(0, "easy")
    env.step({"tool": "wait", "line": "L1", "minutes": 7})
    env.step({"tool": "produce", "line": "L1", "order": object()})
    log = env.final_state()["log"]
    assert log[0]["action"] == {"tool": "wait", "line": "L1", "minutes": 7}
    assert log[0]["status"] == D.STATUS_OK
    assert log[1]["status"] == D.STATUS_INVALID
    json.dumps(log)  # an unserialisable action must not poison the log


def test_the_secret_key_appears_nowhere_the_agent_can_look() -> None:
    env, _ = _env(0, "medium")
    env.step({"tool": "flush", "line": "L1"})
    env.step({"tool": "finish"})
    key_hex = env.secret_key.hex()
    assert key_hex not in json.dumps(env.observation())
    assert key_hex not in json.dumps(env.final_state())


def test_final_state_carries_the_task_and_the_derived_fields() -> None:
    env, _ = _env(0, "easy")
    order = env.task.orders[0]
    env.reset(task=_monensin_task())
    env.step({"tool": "produce", "line": "L1", "order": "O1"})
    env.step({"tool": "finish"})

    state = env.final_state()
    assert D.Task.from_json(state["task"]) == env.task
    assert state["derived"]["completions"]["O1"] == {"line": "L1", "start": 0, "end": 60}
    assert state["derived"]["batches"][0]["feed"] == "broiler_mon"
    assert state["derived"]["finished"] is True
    json.dumps(state)


# -- the planted plan runs cleanly through the environment -----------------


@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_the_planted_plan_produces_a_clean_log(difficulty: str) -> None:
    for seed in range(20):
        gen = G.generate(seed, difficulty)
        env = FeedMillEnv()
        env.reset(task=gen.task)
        for action in gen.plan.actions:
            _, _, terminated, truncated, info = env.step(action)
            assert info["status"] == D.STATUS_OK, (seed, action, info)
            assert not truncated
        assert terminated
        state = env.final_state()
        assert state["derived"]["finished"] is True
        completions = {k: v["end"] for k, v in state["derived"]["completions"].items()}
        assert completions == gen.plan.completion


# -- helper task -----------------------------------------------------------


def _monensin_task(
    feeds: tuple[str, ...] = ("broiler_mon", "broiler_mon"),
    house_rules: tuple[D.HouseRule, ...] = (),
    initial_die: int = 3,
) -> D.Task:
    """A tiny hand-made task, so the physics tests do not depend on a seed."""
    return D.Task(
        task_id="unit-test",
        seed=-1,
        difficulty="easy",
        horizon_min=D.HORIZON_MIN,
        carry_over_rate=D.CARRY_OVER_RATE,
        flush_tonnes=D.FLUSH_TONNES,
        flush_minutes=D.FLUSH_MINUTES,
        die_change_minutes=D.DIE_CHANGE_MINUTES,
        max_steps=D.DEFAULT_MAX_STEPS,
        feeds=D.FEED_CATALOG,
        lines=(
            D.LineSpec("L1", 20.0, D.PAP_NONE, initial_die, D.zero_concentrations()),
        ),
        orders=tuple(
            D.Order(f"O{i + 1}", "Hof Meier", feed, 20.0, 1440) for i, feed in enumerate(feeds)
        ),
        house_rules=house_rules,
    )


def test_a_truncated_episode_cannot_be_finished_afterwards() -> None:
    """Otherwise an agent could ignore truncation and still collect a reward."""
    env = FeedMillEnv(max_steps=2, reward_fn=lambda state, key: 1.0)
    env.reset(task=_monensin_task())
    env.step({"tool": "wait", "line": "L1", "minutes": 1})
    _, _, _, truncated, _ = env.step({"tool": "wait", "line": "L1", "minutes": 1})
    assert truncated

    _, reward, terminated, truncated, info = env.step({"tool": "finish"})
    assert info["status"] == D.STATUS_INVALID
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert env.final_state()["derived"]["finished"] is False


def test_die_sizes_come_from_the_task_catalog() -> None:
    """An external task may describe a mill with different dies (SPEC 13)."""
    task = _monensin_task()
    small = tuple(
        D.Feed(f.id, f.species, f.ruminant, f.substances, f.carry_over_class, 8 if f.die_mm == 4 else f.die_mm)
        for f in task.feeds
    )
    env = FeedMillEnv()
    env.reset(task=D.Task(**{**{k: getattr(task, k) for k in task.__dataclass_fields__}, "feeds": small}))
    assert env.step({"tool": "change_die", "line": "L1", "die_mm": 8})[4]["status"] == D.STATUS_OK
    assert env.step({"tool": "change_die", "line": "L1", "die_mm": 4})[4]["status"] == D.STATUS_INVALID
