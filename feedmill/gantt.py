"""Gantt charts of a played episode.

The chart is drawn from the verifier's replay, not from the environment's
derived fields, so what you see is what was actually graded. Colour says what
kind of block it is; a red hatch marks a batch that carries a substance over
its limit, and a red outline marks an order that finished after its due time.

The limit arithmetic here is for drawing only. The authoritative check is
``verifier.verify``, whose violation list is printed under the chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # no display, no interactive backend

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from . import domain as D
from .verifier import VerificationResult, replay

COLOURS = {
    D.TOOL_PRODUCE: "#4c72b0",
    D.TOOL_FLUSH: "#55a868",
    D.TOOL_CHANGE_DIE: "#8c8c8c",
    D.TOOL_WAIT: "#d9d9d9",
}
LABELS = {
    D.TOOL_PRODUCE: "production",
    D.TOOL_FLUSH: "flush",
    D.TOOL_CHANGE_DIE: "die change",
    D.TOOL_WAIT: "idle",
}

#: PNG metadata that would otherwise make two identical runs differ.
_REPRODUCIBLE = {"Software": None, "Creation Time": None}


def clock_label(minute: int) -> str:
    """Minute 0 is 06:00 (SPEC section 2.6)."""
    return f"{(6 + minute // 60) % 24:02d}:{minute % 60:02d}"


def breaks_a_legal_rule(
    feed: D.Feed, line: D.LineSpec, concentrations: Mapping[str, float] | None
) -> bool:
    """L1-L5 for one batch, for drawing only.

    Repeated here so that the chart can mark the batch that went wrong; the
    verdict under the chart comes from the verifier.
    """
    if concentrations is not None:
        if not feed.contains(D.MONENSIN):
            limit = D.COCCIDIOSTAT_LIMIT.get(feed.carry_over_class)
            if limit is not None and concentrations[D.MONENSIN] > limit + D.LIMIT_TOL:
                return True
        if not feed.contains(D.ANTIMICROBIAL):
            if concentrations[D.ANTIMICROBIAL] > D.ANTIMICROBIAL_LIMIT + D.LIMIT_TOL:
                return True
    if feed.ruminant and line.pap_type != D.PAP_NONE:
        return True
    substance = feed.pap_substance
    if substance is not None and line.pap_type != D.PAP_SUBSTANCE_LINE[substance]:
        return True
    group = D.SPECIES_PAP_GROUP.get(feed.species)
    return group is not None and group == line.pap_type


def plot_schedule(
    final_state: Mapping[str, Any],
    path: str | Path,
    *,
    title: str,
    subtitle: str | None = None,
    result: VerificationResult | None = None,
) -> Path:
    """Draw one episode and write it to ``path``."""
    task = D.Task.from_json(final_state["task"])
    played = replay(final_state)
    feeds = task.feed_map
    orders = task.order_map
    line_ids = [line.id for line in task.lines]
    row = {line_id: index for index, line_id in enumerate(reversed(line_ids))}

    horizon = max((s["end"] for s in played.segments), default=60)
    horizon = max(horizon, max((o.due for o in task.orders), default=60))

    figure, axes = plt.subplots(figsize=(12, 1.6 + 1.1 * len(line_ids)))

    for segment in played.segments:
        y = row[segment["line"]]
        width = segment["end"] - segment["start"]
        feed = feeds.get(segment["feed"] or "")
        late = (
            segment["kind"] == D.TOOL_PRODUCE
            and segment["order"] in orders
            and segment["end"] > orders[segment["order"]].due
        )
        illegal = (
            segment["kind"] == D.TOOL_PRODUCE
            and feed is not None
            and breaks_a_legal_rule(
                feed, task.line_map[segment["line"]], segment["concentrations"]
            )
        )
        axes.barh(
            y,
            width,
            left=segment["start"],
            height=0.55,
            color=COLOURS[segment["kind"]],
            edgecolor="#c44e52" if (late or illegal) else "white",
            linewidth=2.0 if (late or illegal) else 0.5,
            hatch="//" if illegal else None,
        )
        if segment["kind"] == D.TOOL_PRODUCE and width >= horizon / 60:
            caption = segment["order"]
            if width >= horizon / 16:
                caption = f"{segment['order']}\n{segment['feed']}"
            axes.text(
                segment["start"] + width / 2,
                y,
                caption,
                ha="center",
                va="center",
                color="white",
                fontsize=6.5,
            )

    # due times, one tick per order on its line
    for order_id, done in played.completions.items():
        order = orders.get(order_id)
        if order is None:
            continue
        axes.plot(
            [order.due],
            [row[done["line"]] + 0.42],
            marker="v",
            markersize=5,
            color="#333333",
            linestyle="none",
        )

    axes.set_yticks(list(row.values()))
    axes.set_yticklabels(
        [f"{line_id}\n{task.line_map[line_id].pap_type}" for line_id in reversed(line_ids)]
    )
    step = 60 if horizon <= 720 else 120
    ticks = list(range(0, horizon + step, step))
    axes.set_xticks(ticks)
    axes.set_xticklabels([clock_label(t) for t in ticks], fontsize=8)
    axes.set_xlim(0, horizon * 1.02)
    axes.set_xlabel("time of day (minute 0 = 06:00)")
    axes.grid(axis="x", linestyle=":", alpha=0.4)
    axes.set_axisbelow(True)

    handles = [Patch(facecolor=COLOURS[kind], label=LABELS[kind]) for kind in COLOURS]
    handles.append(Patch(facecolor="white", edgecolor="#c44e52", hatch="//", label="breaks L1-L5"))
    handles.append(Patch(facecolor="white", edgecolor="#c44e52", label="late"))
    handles.append(
        plt.Line2D([], [], marker="v", color="#333333", linestyle="none", label="due time")
    )
    axes.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=7, fontsize=8)

    heading = title
    if result is not None:
        heading = f"{title}   |   verifier score {result.score}"
    axes.set_title(heading, loc="left", fontsize=11, fontweight="bold", pad=22)

    note = subtitle
    if note is None and result is not None:
        note = ", ".join(result.violations) if result.violations else "no violations"
    if note:
        axes.text(
            0.0,
            1.015,
            note,
            transform=axes.transAxes,
            fontsize=8.5,
            color="#c44e52" if result is not None and result.violations else "#3a7d44",
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140, metadata=_REPRODUCIBLE)
    plt.close(figure)
    return path
