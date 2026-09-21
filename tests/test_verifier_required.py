"""The four verifier tests the case study requires (SPEC section 14).

    initial state           -> 0
    planted solution        -> 1   (100 seeds x 3 difficulties)
    invalid action          -> 0
    directly written state  -> 0
"""

from __future__ import annotations

import pytest

from feedmill import domain as D
from feedmill import generator as G
from feedmill.env import FeedMillEnv
from feedmill.verifier import verify

DIFFICULTIES = ("easy", "medium", "hard")
SEEDS = range(100)


def _play(task: D.Task, actions) -> tuple[FeedMillEnv, dict]:
    env = FeedMillEnv()
    env.reset(task=task)
    for action in actions:
        env.step(action)
    return env, env.final_state()


# -- 1. the initial state scores 0 -----------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_the_initial_state_scores_zero(difficulty: str) -> None:
    env = FeedMillEnv()
    env.reset(0, difficulty)
    result = verify(env.final_state(), env.secret_key)
    assert result.score == 0
    assert D.V_NOT_FINISHED in result.violations
    assert D.V_MISSING_ORDER in result.violations


# -- 2. the planted solution scores 1 --------------------------------------


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_the_planted_solution_scores_one(difficulty: str) -> None:
    for seed in SEEDS:
        gen = G.generate(seed, difficulty)
        env, state = _play(gen.task, gen.plan.actions)
        result = verify(state, env.secret_key)
        assert result.score == 1, f"{gen.task.task_id}: {result.violations}"
        assert result.violations == []


def test_the_environment_pays_the_verifier_score_as_terminal_reward() -> None:
    """SPEC section 7: step reward 0, terminal reward = verifier score."""
    gen = G.generate(0, "medium")
    env = FeedMillEnv()
    env.reset(task=gen.task)
    for action in gen.plan.actions[:-1]:
        _, reward, *_ = env.step(action)
        assert reward == 0.0
    _, reward, terminated, _, _ = env.step(gen.plan.actions[-1])
    assert terminated and reward == 1.0

    # the same task finished without producing anything pays nothing
    env = FeedMillEnv()
    env.reset(task=gen.task)
    _, reward, terminated, _, _ = env.step({"tool": "finish"})
    assert terminated and reward == 0.0


# -- 3. an invalid action scores 0 -----------------------------------------


@pytest.mark.parametrize(
    "bad_action",
    [
        {"tool": "produce", "line": "L1", "order": "O404"},
        {"tool": "wait", "line": "L1", "minutes": 0},
        {"tool": "change_die", "line": "L1", "die_mm": 5},
        {"tool": "teleport", "line": "L1"},
        {"tool": "flush"},
    ],
)
def test_an_invalid_action_anywhere_scores_zero(bad_action: dict) -> None:
    gen = G.generate(1, "easy")
    actions = list(gen.plan.actions)
    actions.insert(len(actions) // 2, bad_action)

    env, state = _play(gen.task, actions)
    result = verify(state, env.secret_key)
    assert result.score == 0
    assert D.V_INVALID_ACTION in result.violations


def test_an_otherwise_perfect_episode_still_fails_on_one_invalid_action() -> None:
    """The invalid action changes nothing physically -- it must still cost the run."""
    gen = G.generate(2, "medium")
    actions = [{"tool": "wait", "line": "L1", "minutes": -1}] + list(gen.plan.actions)

    env, state = _play(gen.task, actions)
    assert [e["status"] for e in state["log"]].count(D.STATUS_INVALID) == 1
    assert verify(state, env.secret_key).violations == [D.V_INVALID_ACTION]
    assert verify(state, env.secret_key).score == 0


# -- 4. a directly written goal state scores 0 -----------------------------


def test_a_goal_state_written_without_a_log_scores_zero() -> None:
    """Everything delivered on time in the derived fields, no log to back it."""
    gen = G.generate(3, "medium")
    task = gen.task
    state = {
        "task": task.to_json(),
        "task_hash": task.task_hash,
        "log": [],
        "derived": {
            "completions": {
                o.id: {"line": "L1", "start": 0, "end": o.due} for o in task.orders
            },
            "batches": [],
            "lines": [
                {
                    "id": "L1",
                    "clock": max(o.due for o in task.orders),
                    "die_mm": 3,
                    "concentrations": D.zero_concentrations(),
                }
            ],
            "steps": 0,
            "finished": True,
        },
    }
    result = verify(state, b"any key at all")
    assert result.score == 0
    assert D.V_NOT_FINISHED in result.violations
    assert D.V_MISSING_ORDER in result.violations


def test_derived_fields_of_a_real_episode_are_ignored() -> None:
    """A real but unfinished episode, dressed up as a finished one."""
    gen = G.generate(4, "easy")
    env, state = _play(gen.task, gen.plan.actions[:2])

    state["derived"]["finished"] = True
    state["derived"]["completions"] = {
        o.id: {"line": "L1", "start": 0, "end": o.due} for o in gen.task.orders
    }
    result = verify(state, env.secret_key)
    assert result.score == 0
    assert D.V_MISSING_ORDER in result.violations


def test_a_forged_log_entry_scores_zero() -> None:
    """Appending a finish entry without the key cannot work."""
    gen = G.generate(5, "easy")
    env, state = _play(gen.task, gen.plan.actions[:-1])  # everything but the finish
    last = state["log"][-1]
    state["log"].append(
        {
            "i": last["i"] + 1,
            "action": {"tool": "finish"},
            "status": D.STATUS_OK,
            "prev_hash": last["hash"],
            "hash": "0" * 64,
        }
    )
    result = verify(state, env.secret_key)
    assert result.score == 0
    assert result.violations == [D.V_TAMPERED_LOG]


# -- the verifier is a separate checker, not a view on the environment -----


def test_the_verifier_imports_nothing_but_the_domain() -> None:
    """SPEC section 9: independent of env.py, sharing only domain.py."""
    import ast
    import pathlib

    from feedmill import verifier

    tree = ast.parse(pathlib.Path(verifier.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:  # a relative import inside the package
                imported.update(f".{alias.name}" for alias in node.names)
            else:
                imported.add(module)

    assert ".domain" in imported
    assert not {name for name in imported if name.startswith(".")} - {".domain"}
    assert not any("env" in name or "generator" in name for name in imported)
