"""Gantt charts of a played episode.

Drawn from the verifier's replay, not from the environment's derived fields,
so what you see is what was graded.

What the picture encodes, and why:

* **Colour says what a batch leaves behind**, because that is what decides
  whether the next batch is legal. A batch carrying a tracked substance is
  the source of every carry-over problem, a flush is the cure, and everything
  else is line time that makes no product. Three hues carry that, taken in
  order from the reference palette, with the substance also named in the bar
  label so identity never rests on colour alone.
* **A whisker runs from the end of each batch to its due time**, so slack is
  visible per order rather than as a row of unattached markers. A late order
  draws it backwards, in red.
* **A red outline and a 45-degree hatch mark a batch that breaks a rule**,
  with the reason written underneath.

The limit arithmetic here is for drawing only. The authoritative check is
``verifier.verify``, whose violation list is printed under the chart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # no display, no interactive backend

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import domain as D
from .verifier import VerificationResult, replay

# Reference palette: categorical slots 1-3, which are the three that validate
# on every pair rather than only on neighbours. Blocks on a line sit next to
# each other in any order, so every pair has to hold.
CLEAN = "#2a78d6"  # slot 1, blue
CARRIES = "#eb6834"  # slot 2, orange
FLUSH = "#1baf7a"  # slot 3, aqua

# Line time that produces nothing is chrome, not data.
DIE_CHANGE = "#b4b3ac"
IDLE = "#e1e0d9"

CRITICAL = "#d03b3b"  # status, reserved
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

#: PNG metadata that would otherwise make two identical runs differ.
_REPRODUCIBLE = {"Software": None, "Creation Time": None}

#: Substances the carry-over limits L1 and L2 apply to.
LIMITED = (D.MONENSIN, D.ANTIMICROBIAL)


def clock_label(minute: int) -> str:
    """Minute 0 is 06:00 (SPEC section 2.6)."""
    return f"{(6 + minute // 60) % 24:02d}:{minute % 60:02d}"


def rule_broken(
    feed: D.Feed, line: D.LineSpec, concentrations: Mapping[str, float] | None
) -> str | None:
    """Why this batch is illegal under L1-L5, in a few words, or None.

    Repeated here so the chart can say what went wrong on the batch itself;
    the verdict under the chart comes from the verifier.
    """
    if concentrations is not None:
        if not feed.contains(D.MONENSIN):
            limit = D.COCCIDIOSTAT_LIMIT.get(feed.carry_over_class)
            if limit is not None and concentrations[D.MONENSIN] > limit + D.LIMIT_TOL:
                return f"monensin {concentrations[D.MONENSIN]:.1%} > {limit:.0%}"
        if not feed.contains(D.ANTIMICROBIAL):
            level = concentrations[D.ANTIMICROBIAL]
            if level > D.ANTIMICROBIAL_LIMIT + D.LIMIT_TOL:
                return f"antimicrobial {level:.1%} > {D.ANTIMICROBIAL_LIMIT:.0%}"
    if feed.ruminant and line.pap_type != D.PAP_NONE:
        return "ruminant feed on a PAP line"
    substance = feed.pap_substance
    if substance is not None and line.pap_type != D.PAP_SUBSTANCE_LINE[substance]:
        return f"needs a {D.PAP_SUBSTANCE_LINE[substance]}-PAP line"
    group = D.SPECIES_PAP_GROUP.get(feed.species)
    if group is not None and group == line.pap_type:
        return f"{feed.species} feed on a {group}-PAP line"
    return None


def _production_colour(feed: D.Feed) -> str:
    return CARRIES if feed.substances else CLEAN


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

    figure, axes = plt.subplots(figsize=(13, 2.1 + 1.45 * len(line_ids)))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    for segment in played.segments:
        y = row[segment["line"]]
        start, end = segment["start"], segment["end"]
        width = end - start
        kind = segment["kind"]

        if kind == D.TOOL_PRODUCE:
            feed = feeds[segment["feed"]]
            line = task.line_map[segment["line"]]
            reason = rule_broken(feed, line, segment["concentrations"])
            colour = _production_colour(feed)
        else:
            feed = reason = None
            colour = {D.TOOL_FLUSH: FLUSH, D.TOOL_CHANGE_DIE: DIE_CHANGE, D.TOOL_WAIT: IDLE}[kind]

        axes.barh(
            y,
            width,
            left=start,
            height=0.52,
            color=colour,
            edgecolor=CRITICAL if reason else SURFACE,
            linewidth=1.8,
            hatch="//" if reason else None,
            zorder=3,
        )

        if kind == D.TOOL_PRODUCE:
            order = orders[segment["order"]]
            _label_batch(axes, y, start, width, horizon, feed.id, order)
            if reason:
                axes.text(
                    start + width / 2,
                    y - 0.40,
                    reason,
                    ha="center",
                    va="top",
                    fontsize=6,
                    color=CRITICAL,
                    zorder=5,
                )
            _due_whisker(axes, y, end, order.due)
        elif kind == D.TOOL_CHANGE_DIE and _fits(f"die {segment['feed']} mm", width, horizon):
            axes.text(
                start + width / 2,
                y,
                f"die {segment['feed']} mm",
                ha="center",
                va="center",
                fontsize=6,
                color=INK_SECONDARY,
                zorder=5,
            )

    axes.set_yticks(list(row.values()))
    axes.set_yticklabels(
        [
            f"{line_id}\n{_line_caption(task.line_map[line_id])}"
            for line_id in reversed(line_ids)
        ],
        fontsize=9,
        color=INK,
    )
    axes.set_ylim(-0.75, len(line_ids) - 0.25)

    step = 60 if horizon <= 720 else 120
    ticks = list(range(0, horizon + step, step))
    axes.set_xticks(ticks)
    axes.set_xticklabels([clock_label(t) for t in ticks], fontsize=8, color=INK_MUTED)
    axes.set_xlim(-horizon * 0.01, horizon * 1.03)
    axes.set_xlabel("time of day (minute 0 = 06:00)", fontsize=9, color=INK_SECONDARY)
    axes.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(length=0)

    handles = [
        Patch(facecolor=CLEAN, label="batch that leaves nothing behind"),
        Patch(facecolor=CARRIES, label="batch carrying a substance"),
        Patch(facecolor=FLUSH, label="flush"),
        Patch(facecolor=DIE_CHANGE, label="die change"),
        Patch(facecolor=IDLE, label="idle"),
        Patch(facecolor=SURFACE, edgecolor=CRITICAL, hatch="//", label="breaks a rule"),
        Line2D([], [], color=INK_MUTED, marker="|", markersize=7, label="slack to the due time"),
        Line2D([], [], color=CRITICAL, marker="|", markersize=7, label="late"),
    ]
    axes.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        fontsize=8,
        frameon=False,
        labelcolor=INK_SECONDARY,
    )

    heading = title
    if result is not None:
        heading = f"{title}   |   verifier score {result.score}"
    axes.set_title(heading, loc="left", fontsize=12, fontweight="bold", color=INK, pad=24)

    note = subtitle
    if note is None and result is not None:
        note = ", ".join(result.violations) if result.violations else "no violations"
    if note:
        axes.text(
            0.0,
            1.02,
            note,
            transform=axes.transAxes,
            fontsize=8.5,
            color=CRITICAL if result is not None and result.violations else "#006300",
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=140, facecolor=SURFACE, metadata=_REPRODUCIBLE)
    plt.close(figure)
    return path


def _line_caption(line: D.LineSpec) -> str:
    return "plain line" if line.pap_type == D.PAP_NONE else f"{line.pap_type} PAP"


#: Roughly how many characters of label fit per unit of bar width, as a share
#: of the horizon. Calibrated on the widest feed names at the sizes below;
#: a label that does not fit is dropped rather than allowed to overflow.
_CHARS_PER_WIDTH = 165.0


def _fits(text: str, width: int, horizon: int) -> bool:
    return len(text) <= _CHARS_PER_WIDTH * width / horizon


def _label_batch(axes, y: float, start: int, width: int, horizon: int, feed_id: str, order) -> None:
    """Feed first, because the feed is what the reader needs; id and tonnage under it."""
    centre = start + width / 2
    detail = f"{order.id} · {order.tonnes:g} t"

    if _fits(feed_id, width, horizon) and _fits(detail, width, horizon):
        axes.text(centre, y + 0.09, feed_id, ha="center", va="center", fontsize=7,
                  color="white", zorder=5)
        axes.text(centre, y - 0.12, detail, ha="center", va="center",
                  fontsize=6, color="white", alpha=0.85, zorder=5)
    elif _fits(feed_id, width, horizon):
        axes.text(centre, y, feed_id, ha="center", va="center", fontsize=6.5,
                  color="white", zorder=5)
    else:
        # A narrow bar still has to say what the feed was -- that is usually
        # the reason it or its neighbour is in trouble -- so the name is cut
        # rather than replaced by the order id.
        room = int(_CHARS_PER_WIDTH * width / horizon)
        if room >= 6:
            axes.text(centre, y, feed_id[: room - 1] + "\u2026", ha="center", va="center",
                      fontsize=6.5, color="white", zorder=5)
        elif _fits(order.id, width, horizon):
            axes.text(centre, y, order.id, ha="center", va="center", fontsize=6,
                      color="white", zorder=5)


def _due_whisker(axes, y: float, end: int, due: int) -> None:
    """From the end of the batch to its due time: the slack, or the overrun."""
    late = due < end
    axes.plot(
        [due, end] if late else [end, due],
        [y + 0.36, y + 0.36],
        color=CRITICAL if late else INK_MUTED,
        linewidth=1.6 if late else 1.0,
        solid_capstyle="butt",
        zorder=4,
    )
    axes.plot(
        [due], [y + 0.36], marker="|", markersize=7,
        color=CRITICAL if late else INK_MUTED, zorder=4,
    )
