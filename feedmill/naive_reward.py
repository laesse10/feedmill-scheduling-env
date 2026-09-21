"""The naive reward and the two exploits it invites (SPEC section 10).

This module is kept deliberately. It is the reward you write on the first
afternoon -- "did every order come out on time?" -- read straight off the
convenience fields of the final state. It is also wrong in two different
ways, and both are cheap to demonstrate:

* **Exploit A** never flushes and ignores sequencing. Every order is on time,
  so the naive reward pays 1, while layer feed leaves the mill with 2 %
  monensin against a 1 % limit and horse feed carries an ionophore that is
  toxic to horses.
* **Exploit B** does not schedule anything at all. It writes the completion
  times into the derived fields. The naive reward reads exactly those fields,
  so it pays 1 for a mill that never ran.

The fix is ``verifier.py``: replay the HMAC-chained log, recompute every
completion time and check the rules. ``tests/test_exploit.py`` proves both
exploits score 0 there, and ``docs/exploit.md`` writes the story out.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from . import domain as D

Action = dict[str, Any]


def naive_reward(final_state: Mapping[str, Any], key: bytes | None = None) -> float:
    """1.0 if every order ends at or before its due time. The wrong reward.

    It trusts the derived fields, it never looks at the log, and it knows
    nothing about carry-over, PAP lines or house rules. The ``key`` argument
    is ignored and only there so that this can be dropped into
    ``FeedMillEnv(reward_fn=...)`` in place of the verifier.
    """
    orders = final_state.get("task", {}).get("orders", [])
    completions = final_state.get("derived", {}).get("completions", {})
    for order in orders:
        done = completions.get(order["id"])
        if done is None or done["end"] > order["due"]:
            return 0.0
    return 1.0


# --------------------------------------------------------------------------
# Exploit A: a schedule that is on time and illegal
# --------------------------------------------------------------------------


def exploit_a_plan(task: D.Task) -> list[Action]:
    """Earliest due date first, change the die when you must, never flush.

    No flush costs 15 minutes, so this beats every due time the planted plan
    met, and the naive reward is happy. It is also the policy an agent finds
    first when the reward only counts lateness: the carry-over rules are
    invisible to it. ``agents.edd_naive`` plays the same policy reactively.
    """
    actions: list[Action] = []
    clocks = {line.id: 0 for line in task.lines}
    dies = {line.id: line.initial_die_mm for line in task.lines}

    for order in sorted(task.orders, key=lambda o: (o.due, o.id)):
        feed = task.feed_map[order.feed]

        def completion(line: D.LineSpec) -> int:
            setup = 0 if dies[line.id] == feed.die_mm else D.die_change_minutes(task, line.id)
            return clocks[line.id] + setup + D.production_minutes(order.tonnes, line.rate_t_per_h)

        line = min(task.lines, key=lambda ln: (completion(ln), ln.id))

        if dies[line.id] != feed.die_mm:
            actions.append({"tool": D.TOOL_CHANGE_DIE, "line": line.id, "die_mm": feed.die_mm})
            clocks[line.id] += D.die_change_minutes(task, line.id)
            dies[line.id] = feed.die_mm

        actions.append({"tool": D.TOOL_PRODUCE, "line": line.id, "order": order.id})
        clocks[line.id] += D.production_minutes(order.tonnes, line.rate_t_per_h)

    actions.append({"tool": D.TOOL_FINISH})
    return actions


# --------------------------------------------------------------------------
# Exploit B: a goal state that was never produced
# --------------------------------------------------------------------------


def fabricated_state(task: D.Task) -> dict[str, Any]:
    """A final state with every order delivered on time and an empty log."""
    return {
        "task": task.to_json(),
        "task_hash": task.task_hash,
        "log": [],
        "derived": _delivered_on_time(task),
    }


def forged_state(final_state: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a real state whose derived fields claim everything is done.

    The log is left untouched, which is the interesting case: the chain is
    intact and signed, and only the convenience fields lie.
    """
    forged = copy.deepcopy(dict(final_state))
    task = D.Task.from_json(forged["task"])
    forged["derived"] = _delivered_on_time(task)
    return forged


def _delivered_on_time(task: D.Task) -> dict[str, Any]:
    line_id = task.lines[0].id
    return {
        "completions": {
            order.id: {"line": line_id, "start": 0, "end": order.due} for order in task.orders
        },
        "batches": [],
        "lines": [
            {
                "id": line.id,
                "clock": max((o.due for o in task.orders), default=0),
                "die_mm": line.initial_die_mm,
                "concentrations": D.zero_concentrations(),
            }
            for line in task.lines
        ],
        "steps": 0,
        "finished": True,
    }
