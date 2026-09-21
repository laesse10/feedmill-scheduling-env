"""The verifier: reads only the final state and replays it from scratch.

Independence is the point of this module (SPEC section 9). It never imports
``env.py``, and from ``domain.py`` it takes only constants and data classes.
Every formula it needs -- canonical JSON, the HMAC chain, the carry-over
update, production time, the die change duration -- is implemented here a
second time, from SPEC. If this module shared its arithmetic with the
environment, a bug in that arithmetic would be invisible to both.

It also never trusts the derived convenience fields of the state. Completion
times, batch concentrations and the "finished" flag are recomputed by walking
the log; the fields themselves are not read at all. That is what makes
exploit B (writing the goal state directly) score 0.

    verify(final_state, key) -> VerificationResult(score, violations)

Score 1 requires all of: an intact chain anchored at the task hash, no
invalid entry and a closing ``finish``, every order produced exactly once and
in full, every order on time, L1-L6 for every batch, and every house rule of
the task.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import domain as D

#: The exact key set of a log entry (SPEC section 8).
_ENTRY_KEYS = frozenset({"i", "action", "status", "prev_hash", "hash"})

#: The keys each tool takes (SPEC section 6).
_ACTION_KEYS: dict[str, frozenset[str]] = {
    D.TOOL_PRODUCE: frozenset({"tool", "line", "order"}),
    D.TOOL_FLUSH: frozenset({"tool", "line"}),
    D.TOOL_CHANGE_DIE: frozenset({"tool", "line", "die_mm"}),
    D.TOOL_WAIT: frozenset({"tool", "line", "minutes"}),
    D.TOOL_FINISH: frozenset({"tool"}),
}


@dataclass(frozen=True)
class VerificationResult:
    score: int
    violations: list[str]


@dataclass
class LineReplay:
    """The line state as recomputed from the log."""

    id: str
    clock: int = 0
    die_mm: int = 0
    concentrations: dict[str, float] = field(default_factory=D.zero_concentrations)
    last_batch_kind: str | None = None
    last_batch_feed: str | None = None
    last_production_feed: str | None = None
    antimicrobial_end: int | None = None
    copper_pending: bool = False


@dataclass
class Replay:
    """What the log actually did, next to the verdict on every entry."""

    lines: dict[str, LineReplay]
    batches: list[dict[str, Any]]
    completions: dict[str, dict[str, Any]]
    statuses: list[str]
    finished: bool
    violations: list[str]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def verify(final_state: Mapping[str, Any], key: bytes | str) -> VerificationResult:
    """Score a final state 1 or 0, with the violation codes that cost it."""
    if not _chain_is_intact(final_state, key):
        # Nothing else can be trusted once the chain is broken, so the run
        # fails on that alone.
        return VerificationResult(score=0, violations=[D.V_TAMPERED_LOG])

    result = replay(final_state)
    return VerificationResult(
        score=0 if result.violations else 1,
        violations=list(result.violations),
    )


def replay(final_state: Mapping[str, Any]) -> Replay:
    """Walk the log and rebuild everything the state claims to have done.

    Also used by the Gantt charts and by the differential test that compares
    this reconstruction with the environment's own state.
    """
    task = D.Task.from_json(final_state["task"])
    log = list(final_state.get("log") or [])

    feeds = task.feed_map
    lines = task.line_map
    orders = task.order_map
    house = {rule.id: rule for rule in task.house_rules}

    state = {
        line_id: LineReplay(
            id=line_id,
            die_mm=line.initial_die_mm,
            concentrations=dict(line.initial_concentrations or D.zero_concentrations()),
        )
        for line_id, line in lines.items()
    }
    batches: list[dict[str, Any]] = []
    completions: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    violations: list[str] = []
    finished = False

    def flag(code: str) -> None:
        if code not in violations:
            violations.append(code)

    for index, entry in enumerate(log):
        action = entry.get("action")
        possible = _is_possible(action, task, state, completions, finished, index)
        statuses.append(D.STATUS_OK if possible else D.STATUS_INVALID)

        if entry.get("status") != D.STATUS_OK or not possible:
            # Either the environment rejected it, or it could never have been
            # accepted. Both mean the episode contains an invalid action.
            flag(D.V_INVALID_ACTION)
            if (
                entry.get("status") == D.STATUS_OK
                and isinstance(action, Mapping)
                and action.get("tool") == D.TOOL_PRODUCE
                and action.get("order") in completions
            ):
                # The log claims an order ran twice (SPEC section 9, rule 3).
                flag(D.V_DUPLICATE_ORDER)
            continue

        tool = action["tool"]
        if tool == D.TOOL_FINISH:
            finished = True
            continue

        line = lines[action["line"]]
        st = state[line.id]

        if tool == D.TOOL_WAIT:
            st.clock += int(action["minutes"])

        elif tool == D.TOOL_CHANGE_DIE:
            rule = house.get(D.H4_DIE_CHANGE_DAY_ONLY)
            if rule and st.clock > int(rule.params["latest_start"]):
                flag(D.V_H4_DIE_CHANGE_TIME)
            st.clock += _die_change_minutes(task, line.id)
            st.die_mm = int(action["die_mm"])

        elif tool == D.TOOL_FLUSH:
            start = st.clock
            st.concentrations = _carry_over(st.concentrations, (), task.carry_over_rate)
            st.clock += task.flush_minutes
            st.last_batch_kind = D.TOOL_FLUSH
            st.last_batch_feed = None
            st.copper_pending = False
            batches.append(_batch(line.id, D.TOOL_FLUSH, None, None, start, st.clock, st.concentrations))

        elif tool == D.TOOL_PRODUCE:
            order = orders[action["order"]]
            feed = feeds[order.feed]
            start = st.clock

            _check_line_rules(feed, line, flag)
            st.concentrations = _carry_over(st.concentrations, feed.substances, task.carry_over_rate)
            _check_limits(feed, st.concentrations, flag)
            _check_house_rules(house, task, order, feed, st, flag)

            # One produce delivers the whole order: there is no partial
            # production in the action space (SPEC section 6), so "produced
            # exactly once, in full" reduces to counting the produce entries.
            st.clock += _production_minutes(order.tonnes, line.rate_t_per_h)
            if st.clock > order.due:
                flag(D.V_LATE)

            st.last_batch_kind = D.TOOL_PRODUCE
            st.last_batch_feed = feed.id
            st.last_production_feed = feed.id
            if feed.contains(D.ANTIMICROBIAL):
                st.antimicrobial_end = st.clock
            if feed.id == "piglet_starter":
                st.copper_pending = True

            completions[order.id] = {"line": line.id, "start": start, "end": st.clock}
            batches.append(
                _batch(line.id, D.TOOL_PRODUCE, order.id, feed.id, start, st.clock, st.concentrations)
            )

    if set(completions) != set(orders):
        flag(D.V_MISSING_ORDER)
    if not finished:
        flag(D.V_NOT_FINISHED)

    return Replay(
        lines=state,
        batches=batches,
        completions=completions,
        statuses=statuses,
        finished=finished,
        violations=violations,
    )


# --------------------------------------------------------------------------
# The rules (SPEC sections 4 and 5)
# --------------------------------------------------------------------------


def _check_line_rules(feed: D.Feed, line: D.LineSpec, flag) -> None:
    """L3, L4 and L5: which line may run which feed at all."""
    if feed.ruminant and line.pap_type != D.PAP_NONE:
        flag(D.V_L3_RUMINANT_BAN)
    for substance in feed.substances:
        required = D.PAP_SUBSTANCE_LINE.get(substance)
        if required is not None and line.pap_type != required:
            flag(D.V_L4_PAP_LINE)
    group = D.SPECIES_PAP_GROUP.get(feed.species)
    if group is not None and group == line.pap_type:
        flag(D.V_L5_INTRA_SPECIES)


def _check_limits(feed: D.Feed, concentrations: Mapping[str, float], flag) -> None:
    """L1 and L2: how much of a substance a batch may carry over."""
    if not feed.contains(D.MONENSIN):
        limit = D.COCCIDIOSTAT_LIMIT.get(feed.carry_over_class)
        if limit is not None and concentrations[D.MONENSIN] > limit + D.LIMIT_TOL:
            flag(D.V_L1_COCCIDIOSTAT)
    if not feed.contains(D.ANTIMICROBIAL):
        if concentrations[D.ANTIMICROBIAL] > D.ANTIMICROBIAL_LIMIT + D.LIMIT_TOL:
            flag(D.V_L2_ANTIMICROBIAL)


def _check_house_rules(
    house: Mapping[str, D.HouseRule],
    task: D.Task,
    order: D.Order,
    feed: D.Feed,
    st: LineReplay,
    flag,
) -> None:
    """H1, H2, H5 and H6 at the moment a production starts.

    H3 (slow die) changes how long a die change takes and is applied when the
    clock advances, so it cannot be violated. H4 is checked at the die change.
    """
    rule = house.get(D.H1_CUSTOMER_FLUSH)
    if rule and order.customer == rule.params.get("customer"):
        if st.last_batch_kind != D.TOOL_FLUSH:
            flag(D.V_H1_CUSTOMER_FLUSH)

    rule = house.get(D.H2_SEQUENCE_BAN)
    if rule and st.last_production_feed == rule.params.get("feed_a"):
        if feed.id == rule.params.get("feed_b"):
            flag(D.V_H2_SEQUENCE_BAN)

    rule = house.get(D.H5_LAB_HOLD)
    if rule and feed.carry_over_class == D.CLASS_SENSITIVE and st.antimicrobial_end is not None:
        if st.clock < st.antimicrobial_end + int(rule.params["minutes"]):
            flag(D.V_H5_LAB_HOLD)

    rule = house.get(D.H6_COPPER_SHEEP_FLUSH)
    if rule and feed.id == str(rule.params.get("before_feed")) and st.copper_pending:
        flag(D.V_H6_COPPER_SHEEP_FLUSH)


# --------------------------------------------------------------------------
# Physical possibility (SPEC section 6), judged independently of the log
# --------------------------------------------------------------------------


def _is_possible(
    action: Any,
    task: D.Task,
    state: Mapping[str, LineReplay],
    completions: Mapping[str, Any],
    finished: bool,
    index: int,
) -> bool:
    """Could the environment have accepted this action at this point?

    The recorded status is not consulted: a log entry that says ``ok`` for
    something the press cannot do is still an invalid action.
    """
    if finished or index >= task.max_steps:
        return False
    if not isinstance(action, Mapping):
        return False
    tool = action.get("tool")
    if not isinstance(tool, str) or tool not in _ACTION_KEYS:
        return False
    if set(action) != _ACTION_KEYS[tool]:
        return False
    if tool == D.TOOL_FINISH:
        return True

    line_id = action.get("line")
    if line_id not in state:
        return False

    if tool == D.TOOL_WAIT:
        minutes = action["minutes"]
        return not isinstance(minutes, bool) and isinstance(minutes, int) and minutes > 0

    if tool == D.TOOL_CHANGE_DIE:
        die_mm = action["die_mm"]
        if isinstance(die_mm, bool) or not isinstance(die_mm, int):
            return False
        return die_mm in {feed.die_mm for feed in task.feeds}

    if tool == D.TOOL_PRODUCE:
        order_id = action["order"]
        if not isinstance(order_id, str) or order_id not in task.order_map:
            return False
        if order_id in completions:
            return False
        feed = task.feed_map[task.order_map[order_id].feed]
        return state[line_id].die_mm == feed.die_mm

    return True  # pragma: no cover - every tool is handled above


# --------------------------------------------------------------------------
# The log chain (SPEC section 8)
# --------------------------------------------------------------------------


def _chain_is_intact(final_state: Mapping[str, Any], key: bytes | str) -> bool:
    """Recompute the anchor and every HMAC, in order."""
    if not isinstance(final_state, Mapping):
        return False
    task_json = final_state.get("task")
    log = final_state.get("log")
    if not isinstance(task_json, Mapping) or not isinstance(log, list):
        return False

    secret = key.encode("utf-8") if isinstance(key, str) else key
    try:
        anchor = _sha256_hex(_canonical_json(task_json))
    except (TypeError, ValueError):
        return False

    # task_hash is a convenience field, so it is checked rather than trusted.
    stated = final_state.get("task_hash")
    if stated is not None and stated != anchor:
        return False

    previous = anchor
    for index, entry in enumerate(log):
        if not isinstance(entry, Mapping) or set(entry) != _ENTRY_KEYS:
            return False
        # Redundant with the HMAC below -- both fields are inside the signed
        # payload, so a reordered or renumbered log already fails there. Kept
        # because it states the intent and fails on the cheap check first.
        if entry["i"] != index or entry["prev_hash"] != previous:
            return False
        payload = {k: v for k, v in entry.items() if k != "hash"}
        try:
            message = previous + _canonical_json(payload)
        except (TypeError, ValueError):
            return False
        digest = hmac.new(secret, message.encode("utf-8"), hashlib.sha256).hexdigest()
        if not isinstance(entry["hash"], str) or not hmac.compare_digest(digest, entry["hash"]):
            return False
        previous = entry["hash"]
    return True


# --------------------------------------------------------------------------
# The formulas, implemented here a second time on purpose
# --------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _carry_over(
    previous: Mapping[str, float], substances: Iterable[str], rate: float
) -> dict[str, float]:
    """SPEC section 3: own content plus a share r of the previous batch."""
    contained = set(substances)
    return {
        s: (1.0 if s in contained else 0.0) + rate * float(previous.get(s, 0.0))
        for s in D.SUBSTANCES
    }


def _production_minutes(tonnes: float, rate_t_per_h: float) -> int:
    return int(math.ceil(tonnes * 60.0 / rate_t_per_h - 1e-9))


def _die_change_minutes(task: D.Task, line_id: str) -> int:
    """House rule H3 makes one line slower (SPEC section 5)."""
    for rule in task.house_rules:
        if rule.id == D.H3_SLOW_DIE and rule.params.get("line") == line_id:
            return int(rule.params.get("minutes", D.SLOW_DIE_CHANGE_MINUTES))
    return task.die_change_minutes


def _batch(
    line_id: str,
    kind: str,
    order_id: str | None,
    feed_id: str | None,
    start: int,
    end: int,
    concentrations: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "line": line_id,
        "kind": kind,
        "order": order_id,
        "feed": feed_id,
        "start": start,
        "end": end,
        "concentrations": dict(concentrations),
    }
