"""Rendering the environment for a language model. No API calls.

Three pieces, all pure functions:

* ``TOOL_SCHEMAS`` -- the five actions of SPEC section 6 as JSON Schema tool
  definitions, in the provider-neutral ``{name, description, input_schema}``
  shape that both the Anthropic and OpenAI tool formats are built from.
* ``observation_to_text`` -- the observation as something worth reading:
  clocks as times of day, residues as percentages, rules in their own words.
* ``parse_action`` -- the first JSON object in a model's reply, so a wrapper
  can hand it straight to ``env.step``.

Nothing here talks to a network, and nothing here decides anything. A driver
loop is a dozen lines on top, and is deliberately not part of this package.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from . import domain as D

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "name": D.TOOL_PRODUCE,
        "description": (
            "Run one order to completion on a line. The line must already carry "
            "the die that feed needs. Takes ceil(tonnes / rate * 60) minutes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "line": {"type": "string", "description": "line id, e.g. L1"},
                "order": {"type": "string", "description": "order id, e.g. O3"},
            },
            "required": ["line", "order"],
            "additionalProperties": False,
        },
    },
    {
        "name": D.TOOL_FLUSH,
        "description": (
            "Run a small batch of plain grain through the line and discard it. "
            "Takes 15 minutes and divides every residue by 50."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"line": {"type": "string"}},
            "required": ["line"],
            "additionalProperties": False,
        },
    },
    {
        "name": D.TOOL_CHANGE_DIE,
        "description": "Fit another die on a line. Takes 45 minutes, or 90 on an old press.",
        "input_schema": {
            "type": "object",
            "properties": {
                "line": {"type": "string"},
                "die_mm": {"type": "integer", "description": "die diameter in mm"},
            },
            "required": ["line", "die_mm"],
            "additionalProperties": False,
        },
    },
    {
        "name": D.TOOL_WAIT,
        "description": "Let a line stand idle for a whole number of minutes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "line": {"type": "string"},
                "minutes": {"type": "integer", "minimum": 1},
            },
            "required": ["line", "minutes"],
            "additionalProperties": False,
        },
    },
    {
        "name": D.TOOL_FINISH,
        "description": "End the day. Nothing can be produced afterwards.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
)


def system_prompt() -> str:
    """What the job is, once, before the first observation."""
    return (
        "You schedule one day of production in a compound feed mill.\n\n"
        "Every order must be finished before its due time, and every batch must "
        "obey EU feed law and the mill's own house rules. Both sets of rules are "
        "given to you in full in each observation.\n\n"
        "The trap is carry-over: every batch picks up a share of whatever the "
        "previous batch left in the line. A flush costs 15 minutes and clears it. "
        "Running a less sensitive feed in between works as a flush as well, and "
        "costs nothing. A schedule that finishes early but contaminates a feed "
        "scores zero.\n\n"
        "Answer with exactly one tool call per turn."
    )


def observation_to_text(observation: Mapping[str, Any]) -> str:
    """The observation as plain text, deterministic and compact."""
    out: list[str] = []
    out.append(
        f"FEED MILL, task {observation['task_id']} ({observation['difficulty']}), "
        f"step {observation['step']} of {observation['max_steps']}"
    )
    flush = observation["flush"]
    out.append(
        f"Minute 0 is 06:00. Carry-over rate {observation['carry_over_rate']:.0%} of the "
        f"previous batch. A flush is {flush['tonnes']:g} t and takes {flush['minutes']} min."
    )

    out.append("")
    out.append("LINES")
    for line in observation["lines"]:
        out.append(
            f"  {line['id']}  pap_type {line['pap_type']:<8} clock {_clock(line['clock'])}  "
            f"die {line['die_mm']} mm  die change {line['die_change_minutes']} min"
        )
        last = line["last_batch"]
        if last["kind"] is None:
            out.append("      last batch: none, the line is clean")
        elif last["kind"] == D.TOOL_FLUSH:
            out.append("      last batch: a flush")
        else:
            out.append(f"      last batch: {last['feed']}")
        residue = _residue(line["concentrations"])
        out.append(f"      residue: {residue}")

    out.append("")
    out.append(f"OPEN ORDERS ({len(observation['open_orders'])})")
    if observation["open_orders"]:
        out.append("  id     customer              feed                tonnes  due")
        for order in sorted(observation["open_orders"], key=lambda o: (o["due"], o["id"])):
            out.append(
                f"  {order['id']:<6} {order['customer']:<21} {order['feed']:<19} "
                f"{order['tonnes']:>6.0f}  {_clock(order['due'])}"
            )

    if observation["completed_orders"]:
        out.append("")
        out.append(f"COMPLETED ({len(observation['completed_orders'])})")
        for done in observation["completed_orders"]:
            verdict = "on time" if done["end"] <= done["due"] else "LATE"
            out.append(
                f"  {done['id']:<6} on {done['line']}, ended {_clock(done['end'])}, "
                f"due {_clock(done['due'])} -- {verdict}"
            )

    out.append("")
    out.append("FEED CATALOG")
    out.append("  id                   species      ruminant  contains         class           die")
    for feed in observation["feeds"]:
        contains = ", ".join(feed["substances"]) or "-"
        out.append(
            f"  {feed['id']:<20} {feed['species']:<12} {'yes' if feed['ruminant'] else 'no':<9} "
            f"{contains:<16} {feed['carry_over_class']:<15} {feed['die_mm']} mm"
        )

    out.append("")
    out.append("EU LAW")
    for rule in observation["legal_rules"]:
        out.append(f"  {rule['id']}  {rule['text']}")
        out.append(f"      basis: {rule['basis']}")

    out.append("")
    if observation["house_rules"]:
        out.append("HOUSE RULES (this mill only)")
        for rule in observation["house_rules"]:
            out.append(f"  {rule['id']}  {rule['text']}")
    else:
        out.append("HOUSE RULES: none for this mill.")

    if observation["last_action"] is not None:
        out.append("")
        status = observation["last_status"]
        line = f"LAST ACTION {json.dumps(observation['last_action'], sort_keys=True)} -> {status}"
        if observation["last_error"]:
            line += f" ({observation['last_error']})"
        out.append(line)

    return "\n".join(out)


def parse_action(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model's reply, or None.

    Tolerates prose and code fences around it, because models produce both.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                if isinstance(parsed, dict):
                    return parsed
                start = -1
    return None


def _clock(minute: int) -> str:
    return f"{(6 + minute // 60) % 24:02d}:{minute % 60:02d} (min {minute})"


def _residue(concentrations: Mapping[str, float]) -> str:
    present = [
        f"{substance} {value:.2%}"
        for substance, value in concentrations.items()
        if value > 0.0
    ]
    return ", ".join(present) if present else "clean"
