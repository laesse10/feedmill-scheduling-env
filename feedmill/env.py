"""The scheduling environment: reset, step, and the tamper-evident log.

Two properties matter here.

**The environment enforces only physical possibility** (SPEC section 6). A
wrong die, an unknown order or a negative wait is rejected, because a press
cannot do those things. Running layer feed straight after a coccidiostat
batch is accepted, because a press *can* do that -- it is simply illegal.
Legal and house rule violations are the verifier's job, so that an agent can
make real mistakes and be graded on them.

**Every step is appended to an HMAC chain** (SPEC section 8) keyed with a
secret created per environment instance. The chain starts at the task hash,
so neither the task nor any earlier entry can be changed afterwards without
breaking every later hash.

Known limit: an agent running inside this Python process can read the key by
introspection. In a real deployment the environment runs in its own process
or container and hands the key to the verifier out of band (see README).
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import math
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from . import domain as D
from .generator import generate

Action = Any
RewardFn = Callable[[Mapping[str, Any], bytes], float]

#: The keys an action may carry, per tool (SPEC section 6). Anything else is
#: malformed input, which is an invalid action.
ACTION_KEYS: dict[str, frozenset[str]] = {
    D.TOOL_PRODUCE: frozenset({"tool", "line", "order"}),
    D.TOOL_FLUSH: frozenset({"tool", "line"}),
    D.TOOL_CHANGE_DIE: frozenset({"tool", "line", "die_mm"}),
    D.TOOL_WAIT: frozenset({"tool", "line", "minutes"}),
    D.TOOL_FINISH: frozenset({"tool"}),
}


@dataclass
class LineState:
    """Physical state of one line: its own clock, die and residue."""

    id: str
    clock: int = 0
    die_mm: int = 3
    concentrations: dict[str, float] = field(default_factory=D.zero_concentrations)
    last_batch_kind: str | None = None
    last_batch_feed: str | None = None


class FeedMillEnv:
    """Gymnasium-style environment, without the gymnasium dependency."""

    def __init__(
        self,
        *,
        secret_key: bytes | None = None,
        max_steps: int | None = None,
        reward_fn: RewardFn | None = None,
    ) -> None:
        self._secret_key = secret_key if secret_key is not None else secrets.token_bytes(32)
        self._max_steps_override = max_steps
        self._reward_fn = reward_fn
        self._task: D.Task | None = None

    # -- out-of-band channel to the verifier -------------------------------

    @property
    def secret_key(self) -> bytes:
        """The HMAC key. Handed to the verifier out of band, never observed."""
        return self._secret_key

    @property
    def task(self) -> D.Task:
        if self._task is None:
            raise RuntimeError("call reset() before using the environment")
        return self._task

    @property
    def max_steps(self) -> int:
        if self._max_steps_override is not None:
            return self._max_steps_override
        return self.task.max_steps

    # -- episode -----------------------------------------------------------

    def reset(
        self, seed: int = 0, difficulty: str = "easy", *, task: D.Task | None = None
    ) -> dict[str, Any]:
        """Start a new episode on a generated task, or on one given directly.

        The planted solution of a generated task is dropped here: the
        environment never holds it, so it cannot leak into an observation.
        """
        self._task = task if task is not None else generate(seed, difficulty).task
        self._lines = {
            line.id: LineState(
                id=line.id,
                die_mm=line.initial_die_mm,
                concentrations=dict(line.initial_concentrations or D.zero_concentrations()),
            )
            for line in self._task.lines
        }
        self._log: list[dict[str, Any]] = []
        self._prev_hash = self._task.task_hash
        self._completions: dict[str, dict[str, Any]] = {}
        self._batches: list[dict[str, Any]] = []
        self._steps = 0
        self._finished = False
        self._truncated = False
        self._last: dict[str, Any] = {"action": None, "status": None, "error": None}
        return self.observation()

    def step(self, action: Action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one JSON tool call. Returns (obs, reward, terminated, truncated, info)."""
        task = self.task  # raises if reset() was not called
        self._steps += 1

        error = self._validate(action)
        status = D.STATUS_INVALID if error else D.STATUS_OK
        if status == D.STATUS_OK:
            self._apply(action)

        self._append_to_log(action, status)
        self._last = {"action": _json_safe(action), "status": status, "error": error}

        if not self._finished and self._steps >= self.max_steps:
            self._truncated = True

        reward = 0.0
        if status == D.STATUS_OK and action["tool"] == D.TOOL_FINISH:
            reward = self._terminal_reward()

        info = {
            "status": status,
            "error": error,
            "step": self._steps,
            "score": reward,
        }
        return self.observation(), reward, self._finished, self._truncated, info

    def _terminal_reward(self) -> float:
        """Terminal reward = verifier score (SPEC section 7).

        The verifier is wired in as ``reward_fn`` from outside, because
        ``verifier.py`` must stay independent of this module: it may never
        import ``env.py``, only the other way round.
        """
        if self._reward_fn is None:
            return 0.0
        return float(self._reward_fn(self.final_state(), self._secret_key))

    # -- validation (SPEC section 6) ---------------------------------------

    def _validate(self, action: Action) -> str | None:
        """None if the action is physically possible, else why it is not."""
        if self._finished:
            return "the episode is already finished"
        if self._truncated:
            return f"the episode was truncated after {self.max_steps} steps"
        if not isinstance(action, Mapping):
            return "malformed action: expected a JSON object"

        tool = action.get("tool")
        if not isinstance(tool, str) or tool not in D.TOOLS:
            return f"unknown tool {tool!r}"
        if set(action) != ACTION_KEYS[tool]:
            expected = ", ".join(sorted(ACTION_KEYS[tool]))
            return f"malformed action: {tool} takes exactly the keys {expected}"
        if tool == D.TOOL_FINISH:
            return None

        line_id = action.get("line")
        if line_id not in self._lines:
            return f"unknown line {line_id!r}"

        if tool == D.TOOL_WAIT:
            minutes = action["minutes"]
            if isinstance(minutes, bool) or not isinstance(minutes, int):
                return "wait minutes must be a whole number of minutes"
            if minutes <= 0:
                return "wait minutes must be positive"
            return None

        if tool == D.TOOL_CHANGE_DIE:
            die_mm = action["die_mm"]
            if isinstance(die_mm, bool) or not isinstance(die_mm, int):
                return "die_mm must be an integer"
            if die_mm not in {feed.die_mm for feed in self.task.feeds}:
                return f"the mill has no {die_mm} mm die"
            return None

        if tool == D.TOOL_PRODUCE:
            order_id = action["order"]
            order = self.task.order_map.get(order_id) if isinstance(order_id, str) else None
            if order is None:
                return f"unknown order {order_id!r}"
            if order_id in self._completions:
                return f"order {order_id} has already been produced"
            feed = self.task.feed_map[order.feed]
            if self._lines[line_id].die_mm != feed.die_mm:
                return (
                    f"wrong die: {order.feed} needs a {feed.die_mm} mm die, line "
                    f"{line_id} has {self._lines[line_id].die_mm} mm"
                )
            return None

        return None  # pragma: no cover - every tool is handled above

    # -- physics -----------------------------------------------------------

    def _apply(self, action: Mapping[str, Any]) -> None:
        task = self.task
        tool = action["tool"]

        if tool == D.TOOL_FINISH:
            self._finished = True
            return

        state = self._lines[action["line"]]

        if tool == D.TOOL_WAIT:
            state.clock += int(action["minutes"])
            return

        if tool == D.TOOL_CHANGE_DIE:
            state.clock += D.die_change_minutes(task, state.id)
            state.die_mm = int(action["die_mm"])
            return

        if tool == D.TOOL_FLUSH:
            start = state.clock
            state.concentrations = D.next_concentrations(
                state.concentrations, (), task.carry_over_rate
            )
            state.clock += task.flush_minutes
            state.last_batch_kind = D.TOOL_FLUSH
            state.last_batch_feed = None
            self._batches.append(
                {
                    "line": state.id,
                    "kind": D.TOOL_FLUSH,
                    "order": None,
                    "feed": None,
                    "start": start,
                    "end": state.clock,
                    "concentrations": dict(state.concentrations),
                }
            )
            return

        if tool == D.TOOL_PRODUCE:
            order = task.order_map[action["order"]]
            feed = task.feed_map[order.feed]
            line = task.line_map[state.id]
            start = state.clock
            state.concentrations = D.next_concentrations(
                state.concentrations, feed.substances, task.carry_over_rate
            )
            state.clock += D.production_minutes(order.tonnes, line.rate_t_per_h)
            state.last_batch_kind = D.TOOL_PRODUCE
            state.last_batch_feed = feed.id
            self._completions[order.id] = {
                "line": state.id,
                "start": start,
                "end": state.clock,
            }
            self._batches.append(
                {
                    "line": state.id,
                    "kind": D.TOOL_PRODUCE,
                    "order": order.id,
                    "feed": feed.id,
                    "start": start,
                    "end": state.clock,
                    "concentrations": dict(state.concentrations),
                }
            )

    # -- the log chain (SPEC section 8) ------------------------------------

    def _append_to_log(self, action: Action, status: str) -> None:
        """Append ``{i, action, status, prev_hash, hash}`` to the chain.

        ``hash = HMAC-SHA256(key, prev_hash + canonical_json(entry))`` where
        *entry* is the entry without its own hash. The first ``prev_hash`` is
        the task hash, so the chain is anchored to the task it was played on.
        """
        entry = {
            "i": len(self._log),
            "action": _json_safe(action),
            "status": status,
            "prev_hash": self._prev_hash,
        }
        digest = hmac.new(
            self._secret_key,
            (self._prev_hash + D.canonical_json(entry)).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        entry["hash"] = digest
        self._log.append(entry)
        self._prev_hash = digest

    # -- what the agent sees (SPEC section 7) ------------------------------

    def observation(self) -> dict[str, Any]:
        task = self.task
        return {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "step": self._steps,
            "max_steps": self.max_steps,
            "horizon_min": task.horizon_min,
            "carry_over_rate": task.carry_over_rate,
            "flush": {"tonnes": task.flush_tonnes, "minutes": task.flush_minutes},
            "lines": [
                {
                    "id": line.id,
                    "pap_type": line.pap_type,
                    "rate_t_per_h": line.rate_t_per_h,
                    "die_change_minutes": D.die_change_minutes(task, line.id),
                    "clock": self._lines[line.id].clock,
                    "die_mm": self._lines[line.id].die_mm,
                    "concentrations": dict(self._lines[line.id].concentrations),
                    "last_batch": {
                        "kind": self._lines[line.id].last_batch_kind,
                        "feed": self._lines[line.id].last_batch_feed,
                    },
                }
                for line in task.lines
            ],
            "feeds": [feed.to_json() for feed in task.feeds],
            "open_orders": [
                order.to_json() for order in task.orders if order.id not in self._completions
            ],
            "completed_orders": [
                {"id": order_id, "due": task.order_map[order_id].due, **done}
                for order_id, done in self._completions.items()
            ],
            "legal_rules": [rule.to_json() for rule in D.LEGAL_RULES],
            "house_rules": [rule.to_json() for rule in task.house_rules],
            "last_action": self._last["action"],
            "last_status": self._last["status"],
            "last_error": self._last["error"],
            "finished": self._finished,
        }

    # -- what the verifier sees (SPEC section 8) ---------------------------

    def final_state(self) -> dict[str, Any]:
        """Task, task hash, log chain and derived convenience fields.

        The derived fields are exactly what the naive reward reads and what
        the verifier ignores: the verifier recomputes them from the log.
        """
        task = self.task
        return copy.deepcopy(
            {
                "task": task.to_json(),
                "task_hash": task.task_hash,
                "log": self._log,
                "derived": {
                    "completions": self._completions,
                    "batches": self._batches,
                    "lines": [
                        {
                            "id": state.id,
                            "clock": state.clock,
                            "die_mm": state.die_mm,
                            "concentrations": dict(state.concentrations),
                        }
                        for state in self._lines.values()
                    ],
                    "steps": self._steps,
                    "finished": self._finished,
                },
            }
        )


def _json_safe(value: Any) -> Any:
    """A JSON-serialisable copy of anything an agent may submit.

    The log has to record malformed actions faithfully without ever failing
    to serialise, so values that JSON cannot express are kept as their repr.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    return repr(value)
