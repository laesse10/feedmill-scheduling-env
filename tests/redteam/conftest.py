"""Shared bench for the red team: build a mill, play a cheat, check the score.

Every test in this directory does the same three things, so they are done
here once: build a small hand-made task where exactly one thing can go wrong,
play an action list through the real environment, and verify the result.

``Bench.resign`` re-signs a modified log with the environment's key. That
simulates the limit SPEC section 8 documents: an agent in the same process
can read the key. The tests that use it show what still holds when it does --
the verifier recomputes the whole episode, so a forged log only passes if the
schedule it describes was legal anyway.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pytest

from feedmill import domain as D
from feedmill.env import FeedMillEnv
from feedmill.verifier import VerificationResult, verify

Action = dict[str, Any]


@dataclass
class Attempt:
    """One played episode, ready to be tampered with and verified."""

    env: FeedMillEnv
    state: dict[str, Any]

    @property
    def key(self) -> bytes:
        return self.env.secret_key

    @property
    def log(self) -> list[dict[str, Any]]:
        return self.state["log"]

    def verify(
        self, state: dict[str, Any] | None = None, key: bytes | str | None = None
    ) -> VerificationResult:
        # `key or self.key` would quietly fall back to the real key for an
        # empty one, which is the bug test_wrong_hmac_key.py exists to catch.
        return verify(
            state if state is not None else self.state,
            self.key if key is None else key,
        )


class Bench:
    """Builds tasks, plays action lists, and forges states."""

    FINISH: Action = {"tool": D.TOOL_FINISH}

    # -- building ---------------------------------------------------------

    def task(
        self,
        feeds: Sequence[str],
        *,
        due: int | Sequence[int] = 600,
        customers: Sequence[str] | None = None,
        tonnes: float = 20.0,
        initial_die: int = 3,
        pap_type: str | None = None,
        house_rules: Iterable[D.HouseRule] = (),
    ) -> D.Task:
        """One line, plus a PAP line as L2 when ``pap_type`` asks for one."""
        dues = [due] * len(feeds) if isinstance(due, int) else list(due)
        names = list(customers) if customers else ["Hof Meier"] * len(feeds)
        lines = [D.LineSpec("L1", 20.0, D.PAP_NONE, initial_die, D.zero_concentrations())]
        if pap_type is not None:
            lines.append(D.LineSpec("L2", 20.0, pap_type, initial_die, D.zero_concentrations()))
        return D.Task(
            task_id="redteam",
            seed=-1,
            difficulty="easy",
            horizon_min=D.HORIZON_MIN,
            carry_over_rate=D.CARRY_OVER_RATE,
            flush_tonnes=D.FLUSH_TONNES,
            flush_minutes=D.FLUSH_MINUTES,
            die_change_minutes=D.DIE_CHANGE_MINUTES,
            max_steps=D.DEFAULT_MAX_STEPS,
            feeds=D.FEED_CATALOG,
            lines=tuple(lines),
            orders=tuple(
                D.Order(f"O{i + 1}", names[i], feed, tonnes, dues[i])
                for i, feed in enumerate(feeds)
            ),
            house_rules=tuple(house_rules),
        )

    def play(self, task: D.Task, actions: Iterable[Action]) -> Attempt:
        env = FeedMillEnv()
        env.reset(task=task)
        for action in actions:
            env.step(action)
        return Attempt(env=env, state=env.final_state())

    def resign(self, state: dict[str, Any], key: bytes) -> dict[str, Any]:
        """Rebuild the whole chain, as an agent that stole the key would."""
        forged = copy.deepcopy(state)
        previous = D.sha256_hex(D.canonical_json(forged["task"]))
        forged["task_hash"] = previous
        for index, entry in enumerate(forged["log"]):
            entry["i"] = index
            entry["prev_hash"] = previous
            payload = {k: v for k, v in entry.items() if k != "hash"}
            entry["hash"] = hmac.new(
                key, (previous + D.canonical_json(payload)).encode("utf-8"), hashlib.sha256
            ).hexdigest()
            previous = entry["hash"]
        return forged

    # -- action builders, so the tests read like schedules ----------------

    @staticmethod
    def produce(order: str, line: str = "L1") -> Action:
        return {"tool": D.TOOL_PRODUCE, "line": line, "order": order}

    @staticmethod
    def flush(line: str = "L1") -> Action:
        return {"tool": D.TOOL_FLUSH, "line": line}

    @staticmethod
    def change_die(die_mm: int, line: str = "L1") -> Action:
        return {"tool": D.TOOL_CHANGE_DIE, "line": line, "die_mm": die_mm}

    @staticmethod
    def wait(minutes: int, line: str = "L1") -> Action:
        return {"tool": D.TOOL_WAIT, "line": line, "minutes": minutes}


@pytest.fixture
def bench() -> Bench:
    return Bench()


@pytest.fixture
def caught():
    """Assert a cheat scored 0 and was named by the right violation code."""

    def check(result: VerificationResult, *codes: str) -> None:
        assert result.score == 0, f"the cheat was not caught: {result}"
        for code in codes:
            assert code in result.violations, f"expected {code}, got {result.violations}"

    return check
