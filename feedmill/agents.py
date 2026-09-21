"""The three baselines and a runner for them (SPEC section 12).

    edd_naive   earliest due date first, changes dies as needed, never flushes
    law_aware   the same, plus a flush whenever the next batch would break
                L1 or L2, and line assignment that respects L3-L5
    full_aware  law_aware plus every house rule of the task

The three share one policy and differ only in what they are allowed to know:
``Knowledge(line_rules, limits, house_rules)``. That matters for the
experiment in SPEC section 12, where the gap between ``full_aware`` and
``law_aware`` is supposed to measure the value of company knowledge. If the
two were separate implementations, the gap would also measure which one was
written better.

The agents are reactive policies: ``act(observation) -> action``. They read
the limits and the line rules out of ``observation["legal_rules"]`` and the
house rules out of ``observation["house_rules"]`` rather than importing them,
which keeps them honest about what an agent can actually see, and checks that
the observation carries enough to act on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import domain as D
from .env import FeedMillEnv
from .verifier import VerificationResult, verify

Action = dict[str, Any]
Observation = Mapping[str, Any]


@dataclass(frozen=True)
class Knowledge:
    """What a baseline is allowed to take into account."""

    line_rules: bool  # L3, L4, L5
    limits: bool  # L1, L2
    house_rules: bool  # H1 to H6


@dataclass
class LineMemory:
    """What the agent remembers about a line beyond what it can see."""

    last_production_feed: str | None = None
    antimicrobial_end: int | None = None
    copper_pending: bool = False


class SchedulingAgent:
    """Earliest due date first, with as much rule knowledge as it is given."""

    def __init__(self, knowledge: Knowledge, name: str) -> None:
        self.knowledge = knowledge
        self.name = name
        self._memory: dict[str, LineMemory] = {}

    # -- episode -----------------------------------------------------------

    def reset(self, observation: Observation) -> None:
        self._memory = {line["id"]: LineMemory() for line in observation["lines"]}
        self._orders: dict[str, Mapping[str, Any]] = {}

    def act(self, observation: Observation) -> Action:
        """One action per call, re-derived from the observation every time."""
        self._absorb(observation)

        if not observation["open_orders"]:
            return {"tool": D.TOOL_FINISH}

        order, line = self._choose(observation)
        feed = _feed(observation, order["feed"])

        if line["die_mm"] != feed["die_mm"]:
            return {"tool": D.TOOL_CHANGE_DIE, "line": line["id"], "die_mm": feed["die_mm"]}

        flush = self._wants_flush(observation, line, order, feed)
        wait = self._required_wait(observation, line, feed, after_flush=flush)
        if wait > 0:
            return {"tool": D.TOOL_WAIT, "line": line["id"], "minutes": wait}
        if flush:
            return {"tool": D.TOOL_FLUSH, "line": line["id"]}
        return {"tool": D.TOOL_PRODUCE, "line": line["id"], "order": order["id"]}

    # -- memory ------------------------------------------------------------

    def _absorb(self, observation: Observation) -> None:
        """Update memory from what the last action did, as reported back.

        A produced order leaves ``open_orders``, and ``completed_orders``
        carries the times but not the feed, so the agent keeps the order
        records it has seen.
        """
        for order in observation["open_orders"]:
            self._orders.setdefault(order["id"], order)

        action = observation.get("last_action")
        if observation.get("last_status") != D.STATUS_OK or not isinstance(action, Mapping):
            return
        memory = self._memory.get(action.get("line"))
        if memory is None:
            return

        if action["tool"] == D.TOOL_FLUSH:
            memory.copper_pending = False
            return
        if action["tool"] != D.TOOL_PRODUCE:
            return

        order = self._orders.get(action["order"])
        if order is None:
            return
        feed = _feed(observation, order["feed"])
        memory.last_production_feed = feed["id"]

        if D.ANTIMICROBIAL in feed["substances"]:
            done = next(
                (c for c in observation["completed_orders"] if c["id"] == action["order"]), None
            )
            if done is not None:
                memory.antimicrobial_end = done["end"]

        rule = _house_rule(observation, D.H6_COPPER_SHEEP_FLUSH)
        if rule and feed["id"] == str(rule["params"].get("after_feed")):
            memory.copper_pending = True

    # -- choosing what to run next -----------------------------------------

    def _choose(self, observation: Observation) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Earliest due date first; the line that would finish it soonest.

        With house rule knowledge, a combination that would break a rule no
        flush or wait can repair (H2, H4) is skipped in favour of the next
        order. If every combination is blocked, the earliest one is taken and
        the violation is accepted -- an agent that stops working scores 0 too.
        """
        candidates = sorted(observation["open_orders"], key=lambda o: (o["due"], o["id"]))
        fallback: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None

        for order in candidates:
            feed = _feed(observation, order["feed"])
            lines = self._legal_lines(observation, feed)
            if not lines:
                lines = list(observation["lines"])
            lines = sorted(lines, key=lambda ln: (self._completion(ln, order, feed), ln["id"]))
            if fallback is None:
                fallback = (order, lines[0])
            for line in lines:
                if not self._blocked(observation, line, order, feed):
                    return order, line

        assert fallback is not None  # open_orders was not empty
        return fallback

    def _legal_lines(self, observation: Observation, feed: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        """L3, L4 and L5, read out of the observation's legal rules."""
        if not self.knowledge.line_rules:
            return list(observation["lines"])
        rules = _legal_rules(observation)
        allowed = []
        for line in observation["lines"]:
            pap = line["pap_type"]
            if feed["ruminant"] and pap not in rules["L3"]["params"]["ruminant_feeds_allowed_pap_types"]:
                continue
            required = {
                substance: pap_type
                for substance, pap_type in rules["L4"]["params"]["required_line_pap_type"].items()
            }
            if any(required.get(s) not in (None, pap) for s in feed["substances"]):
                continue
            group = rules["L5"]["params"]["species_pap_group"].get(feed["species"])
            if group is not None and group == pap:
                continue
            allowed.append(line)
        return allowed

    def _blocked(
        self,
        observation: Observation,
        line: Mapping[str, Any],
        order: Mapping[str, Any],
        feed: Mapping[str, Any],
    ) -> bool:
        """House rules that no inserted flush or wait can repair."""
        if not self.knowledge.house_rules:
            return False
        memory = self._memory[line["id"]]

        rule = _house_rule(observation, D.H2_SEQUENCE_BAN)
        if rule and memory.last_production_feed == rule["params"].get("feed_a"):
            if feed["id"] == rule["params"].get("feed_b"):
                return True

        rule = _house_rule(observation, D.H4_DIE_CHANGE_DAY_ONLY)
        if rule and line["die_mm"] != feed["die_mm"]:
            if line["clock"] > int(rule["params"]["latest_start"]):
                return True
        return False

    def _completion(
        self, line: Mapping[str, Any], order: Mapping[str, Any], feed: Mapping[str, Any]
    ) -> int:
        setup = 0 if line["die_mm"] == feed["die_mm"] else line["die_change_minutes"]
        return line["clock"] + setup + D.production_minutes(order["tonnes"], line["rate_t_per_h"])

    # -- flushing and waiting ----------------------------------------------

    def _wants_flush(
        self,
        observation: Observation,
        line: Mapping[str, Any],
        order: Mapping[str, Any],
        feed: Mapping[str, Any],
    ) -> bool:
        if self.knowledge.limits and self._would_break_a_limit(observation, line, feed):
            return True
        if not self.knowledge.house_rules:
            return False

        rule = _house_rule(observation, D.H1_CUSTOMER_FLUSH)
        if rule and order["customer"] == rule["params"].get("customer"):
            if line["last_batch"]["kind"] != D.TOOL_FLUSH:
                return True

        rule = _house_rule(observation, D.H6_COPPER_SHEEP_FLUSH)
        if rule and feed["id"] == str(rule["params"].get("before_feed")):
            if self._memory[line["id"]].copper_pending:
                return True
        return False

    def _would_break_a_limit(
        self, observation: Observation, line: Mapping[str, Any], feed: Mapping[str, Any]
    ) -> bool:
        """L1 and L2, with the limits taken from the observation."""
        rules = _legal_rules(observation)
        rate = observation["carry_over_rate"]
        after = {
            substance: (1.0 if substance in feed["substances"] else 0.0)
            + rate * line["concentrations"].get(substance, 0.0)
            for substance in line["concentrations"]
        }

        l1 = rules["L1"]["params"]
        if l1["substance"] not in feed["substances"]:
            limit = l1["limits"].get(feed["carry_over_class"])
            if limit is not None and after[l1["substance"]] > limit + D.LIMIT_TOL:
                return True

        l2 = rules["L2"]["params"]
        if l2["substance"] not in feed["substances"]:
            if after[l2["substance"]] > l2["limit"] + D.LIMIT_TOL:
                return True
        return False

    def _required_wait(
        self,
        observation: Observation,
        line: Mapping[str, Any],
        feed: Mapping[str, Any],
        *,
        after_flush: bool,
    ) -> int:
        """House rule H5: the lab release hold after a medicated batch."""
        if not self.knowledge.house_rules:
            return 0
        rule = _house_rule(observation, D.H5_LAB_HOLD)
        if not rule or feed["carry_over_class"] != D.CLASS_SENSITIVE:
            return 0
        released = self._memory[line["id"]].antimicrobial_end
        if released is None:
            return 0
        flush_minutes = observation["flush"]["minutes"] if after_flush else 0
        return max(0, released + int(rule["params"]["minutes"]) - line["clock"] - flush_minutes)


# --------------------------------------------------------------------------
# The three baselines
# --------------------------------------------------------------------------


def edd_naive() -> SchedulingAgent:
    """Earliest due date first. No flushes, no rules. Exploit A as a policy."""
    return SchedulingAgent(Knowledge(False, False, False), "edd_naive")


def law_aware() -> SchedulingAgent:
    """Knows the law (L1-L5), knows nothing about the company."""
    return SchedulingAgent(Knowledge(True, True, False), "law_aware")


def full_aware() -> SchedulingAgent:
    """Knows the law and the house rules."""
    return SchedulingAgent(Knowledge(True, True, True), "full_aware")


AGENTS: dict[str, Callable[[], SchedulingAgent]] = {
    "edd_naive": edd_naive,
    "law_aware": law_aware,
    "full_aware": full_aware,
}


# --------------------------------------------------------------------------
# Running them
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeResult:
    agent: str
    task_id: str
    score: int
    violations: list[str]
    actions: list[Action]
    steps: int
    final_state: dict[str, Any]

    @property
    def solved(self) -> bool:
        return self.score == 1


def run_episode(agent: SchedulingAgent, task: D.Task) -> EpisodeResult:
    """Play one task to the end and verify the result."""
    env = FeedMillEnv()
    observation = env.reset(task=task)
    agent.reset(observation)

    actions: list[Action] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(observation)
        actions.append(action)
        observation, _, terminated, truncated, _ = env.step(action)

    state = env.final_state()
    result = verify(state, env.secret_key)
    return EpisodeResult(
        agent=agent.name,
        task_id=task.task_id,
        score=result.score,
        violations=result.violations,
        actions=actions,
        steps=len(actions),
        final_state=state,
    )


def _feed(observation: Observation, feed_id: str) -> Mapping[str, Any]:
    return next(f for f in observation["feeds"] if f["id"] == feed_id)


def _legal_rules(observation: Observation) -> dict[str, Mapping[str, Any]]:
    return {rule["id"]: rule for rule in observation["legal_rules"]}


def _house_rule(observation: Observation, template: str) -> Mapping[str, Any] | None:
    for rule in observation["house_rules"]:
        if rule["id"] == template:
            return rule
    return None
