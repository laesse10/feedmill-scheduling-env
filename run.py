#!/usr/bin/env python3
"""Evaluate the baselines and write the results.

    python run.py                                     the full evaluation
    python run.py --seeds 20                          a quicker one
    python run.py --task-file examples/example_task.json    an external mill

The default run plays the three baselines of SPEC section 12 on the held-out
seeds, prints the results table, and writes results/results.md next to two
Gantt charts: one schedule the verifier accepts and one that collects the
naive reward while breaking the law.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from feedmill import domain as D
from feedmill import generator as G
from feedmill.agents import AGENTS, EpisodeResult, run_episode
from feedmill.gantt import plot_schedule
from feedmill.naive_reward import naive_reward
from feedmill.verifier import VerificationResult

DIFFICULTIES = ("easy", "medium", "hard")
HELD_OUT_START = 10000
DEFAULT_SEEDS = 100


@dataclass(frozen=True)
class Evaluation:
    """Every episode of one run, indexed by agent and difficulty."""

    episodes: dict[tuple[str, str], list[EpisodeResult]]
    seeds: range
    difficulties: tuple[str, ...]

    def rate(self, agent: str, difficulty: str) -> float:
        runs = self.episodes[(agent, difficulty)]
        return sum(e.score for e in runs) / len(runs)

    def overall(self, agent: str) -> float:
        return sum(self.rate(agent, d) for d in self.difficulties) / len(self.difficulties)

    def violations(self, agent: str) -> Counter:
        counts: Counter = Counter()
        for difficulty in self.difficulties:
            for episode in self.episodes[(agent, difficulty)]:
                counts.update(episode.violations)
        return counts


def evaluate(seeds: range, difficulties: Sequence[str]) -> Evaluation:
    episodes: dict[tuple[str, str], list[EpisodeResult]] = {}
    for difficulty in difficulties:
        tasks = [G.generate(seed, difficulty).task for seed in seeds]
        for name, make in AGENTS.items():
            episodes[(name, difficulty)] = [run_episode(make(), task) for task in tasks]
    return Evaluation(episodes=episodes, seeds=seeds, difficulties=tuple(difficulties))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def results_table(evaluation: Evaluation) -> list[str]:
    header = (
        f"{'agent':<12}"
        + "".join(f"{d:>8} " for d in evaluation.difficulties)
        + f"{'all':>8} "
    )
    rows = [header, "-" * len(header)]
    for agent in AGENTS:
        line = f"{agent:<12}"
        for difficulty in evaluation.difficulties:
            line += f"{evaluation.rate(agent, difficulty):>8.0%} "
        rows.append(line + f"{evaluation.overall(agent):>8.1%} ")
    return rows


def knowledge_gap(evaluation: Evaluation) -> list[tuple[str, float]]:
    return [
        (d, evaluation.rate("full_aware", d) - evaluation.rate("law_aware", d))
        for d in evaluation.difficulties
    ]


def naive_reward_summary(evaluation: Evaluation) -> list[tuple[str, float, float]]:
    """How often the naive reward pays, and how often it pays for an illegal plan.

    The second number is conditional: the share of the *paid* runs that the
    verifier rejects, not the share of all tasks.
    """
    rows = []
    for difficulty in evaluation.difficulties:
        episodes = evaluation.episodes[("edd_naive", difficulty)]
        paid = [e for e in episodes if naive_reward(e.final_state) == 1.0]
        exploited = [e for e in paid if e.score == 0]
        rows.append(
            (difficulty, len(paid) / len(episodes), len(exploited) / len(paid) if paid else 0.0)
        )
    return rows


def print_report(evaluation: Evaluation) -> None:
    seeds = evaluation.seeds
    print()
    print("Feed mill scheduling environment")
    print(
        f"Held-out seeds {seeds.start}-{seeds.stop - 1}, "
        f"{len(seeds)} tasks per difficulty, verifier score"
    )
    print()
    for row in results_table(evaluation):
        print(row)
    print()
    print("Value of company knowledge (full_aware - law_aware)")
    for difficulty, gap in knowledge_gap(evaluation):
        print(f"  {difficulty:<8} {gap:+.0%}")
    print()
    print("law_aware violations (it knows the law, not the house rules)")
    for code, count in evaluation.violations("law_aware").most_common():
        print(f"  {code:<24} {count:>4}")
    print()
    print("The naive reward pays for schedules the verifier rejects")
    for difficulty, paid, exploited in naive_reward_summary(evaluation):
        print(
            f"  {difficulty:<8} pays on {paid:>4.0%} of tasks,  "
            f"and {exploited:>4.0%} of those schedules are illegal"
        )


def write_report(path: Path, evaluation: Evaluation, charts: dict[str, Path]) -> Path:
    seeds = evaluation.seeds
    lines = [
        "# Results",
        "",
        f"Baselines of SPEC section 12 on held-out seeds {seeds.start}-{seeds.stop - 1}, "
        f"{len(seeds)} tasks per difficulty. Reproduce with `python run.py`.",
        "",
        "## Success rate (verifier score 1)",
        "",
        "| agent | " + " | ".join(evaluation.difficulties) + " | all variants |",
        "|---" * (len(evaluation.difficulties) + 2) + "|",
    ]
    for agent in AGENTS:
        cells = " | ".join(f"{evaluation.rate(agent, d):.0%}" for d in evaluation.difficulties)
        lines.append(f"| `{agent}` | {cells} | **{evaluation.overall(agent):.1%}** |")

    lines += [
        "",
        "`edd_naive` earliest due date first, never flushes. `law_aware` adds L1-L5. "
        "`full_aware` adds the house rules. All three share one policy and differ only "
        "in what they are allowed to know.",
        "",
        "## Value of company knowledge",
        "",
        "| difficulty | gap (full_aware - law_aware) |",
        "|---|---|",
    ]
    for difficulty, gap in knowledge_gap(evaluation):
        lines.append(f"| {difficulty} | {gap:+.0%} |")

    lines += [
        "",
        "Control (SPEC section 12): with the house rules removed the two agents take "
        "identical actions on every task, so the gap is exactly 0. "
        "See `tests/test_agents.py::test_the_control_holds_on_tasks_with_no_house_rules`.",
        "",
        "## What law_aware gets wrong",
        "",
        "| violation | count |",
        "|---|---|",
    ]
    for code, count in evaluation.violations("law_aware").most_common():
        lines.append(f"| `{code}` | {count} |")
    lines += [
        "",
        "No `L1`-`L6` anywhere: the gap above is house rules and lateness, nothing else.",
        "",
        "## The naive reward against the verifier",
        "",
        "| difficulty | naive reward = 1 | of those, illegal |",
        "|---|---|---|",
    ]
    for difficulty, paid, exploited in naive_reward_summary(evaluation):
        lines.append(f"| {difficulty} | {paid:.0%} | {exploited:.0%} |")
    lines += [
        "",
        "The naive reward counts lateness only, so it pays for `edd_naive`'s flush-free "
        "schedules. See `docs/exploit.md`.",
        "",
        "## Charts",
        "",
    ]
    for caption, chart in charts.items():
        lines.append(f"![{caption}]({chart.name})")
        lines.append("")
        lines.append(f"*{caption}*")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def pick_demo_task(seeds: Iterable[int], difficulties: Sequence[str]) -> D.Task | None:
    """The first task where full_aware succeeds and edd_naive cheats.

    "Cheats" means the naive reward pays and the verifier does not, which is
    what makes the pair of charts worth looking at.
    """
    for difficulty in difficulties:
        for seed in seeds:
            task = G.generate(seed, difficulty).task
            good = run_episode(AGENTS["full_aware"](), task)
            bad = run_episode(AGENTS["edd_naive"](), task)
            if good.score == 1 and bad.score == 0 and naive_reward(bad.final_state) == 1.0:
                return task
    return None


def write_charts(out_dir: Path, task: D.Task) -> dict[str, Path]:
    charts: dict[str, Path] = {}
    for agent, name, caption in (
        ("full_aware", "gantt_valid.png", "A schedule the verifier accepts"),
        ("edd_naive", "gantt_exploit.png", "On time, and illegal: the naive reward pays 1"),
    ):
        episode = run_episode(AGENTS[agent](), task)
        charts[f"{caption} ({task.task_id}, {agent})"] = plot_schedule(
            episode.final_state,
            out_dir / name,
            title=f"{task.task_id} - {agent}",
            result=VerificationResult(episode.score, episode.violations),
        )
    return charts


# --------------------------------------------------------------------------
# External tasks (SPEC section 13)
# --------------------------------------------------------------------------


def run_task_file(path: Path, out_dir: Path) -> int:
    task = D.Task.from_json(json.loads(path.read_text()))
    print()
    print(f"Task file {path}")
    print(
        f"  {task.task_id}: {len(task.orders)} orders, {len(task.lines)} line(s), "
        f"{len(task.house_rules)} house rule(s), {len(task.feeds)} feeds in the catalog"
    )
    print(f"  task_hash {task.task_hash}")
    print()
    print(f"{'agent':<12}{'score':>7}  violations")
    print("-" * 60)
    best: EpisodeResult | None = None
    for name, make in AGENTS.items():
        episode = run_episode(make(), task)
        codes = ", ".join(episode.violations) if episode.violations else "-"
        print(f"{name:<12}{episode.score:>7}  {codes}")
        if best is None or episode.score > best.score:
            best = episode

    assert best is not None
    chart = plot_schedule(
        best.final_state,
        out_dir / f"gantt_{task.task_id}.png",
        title=f"{task.task_id} - {best.agent}",
        result=VerificationResult(best.score, best.violations),
    )
    print()
    print(f"Wrote {chart}")
    return 0


# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-file", type=Path, help="evaluate one external task instead")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="tasks per difficulty")
    parser.add_argument("--seed-start", type=int, default=HELD_OUT_START, help="first seed")
    parser.add_argument(
        "--difficulties", nargs="+", default=list(DIFFICULTIES), choices=list(DIFFICULTIES)
    )
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument("--no-charts", action="store_true", help="skip the Gantt charts")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.task_file:
        return run_task_file(args.task_file, args.out)

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    evaluation = evaluate(seeds, args.difficulties)
    print_report(evaluation)

    charts: dict[str, Path] = {}
    if not args.no_charts:
        task = pick_demo_task(seeds, tuple(reversed(args.difficulties)))
        if task is not None:
            charts = write_charts(args.out, task)

    report = write_report(args.out / "results.md", evaluation, charts)
    print()
    print(f"Wrote {report}" + ("".join(f", {c}" for c in charts.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
