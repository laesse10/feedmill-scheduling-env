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
import time
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

    def failed(self, agent: str) -> int:
        """How many tasks this agent did not solve, across every difficulty."""
        return sum(
            1
            for difficulty in self.difficulties
            for episode in self.episodes[(agent, difficulty)]
            if episode.score == 0
        )

    @property
    def tasks(self) -> int:
        return len(self.seeds) * len(self.difficulties)

    @property
    def episodes_played(self) -> int:
        return sum(len(runs) for runs in self.episodes.values())

    @property
    def actions_taken(self) -> int:
        return sum(e.steps for runs in self.episodes.values() for e in runs)

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


def reward_comparison(evaluation: Evaluation) -> list[tuple[str, float, float, float, float]]:
    """Each agent under both rewards: (agent, naive all, verifier all, naive hard, verifier hard).

    The naive reward and the verifier do not merely disagree about single
    episodes; they rank the agents in opposite orders.
    """
    rows = []
    for agent in AGENTS:
        naive_all = verifier_all = 0.0
        naive_hard = verifier_hard = float("nan")
        for difficulty in evaluation.difficulties:
            episodes = evaluation.episodes[(agent, difficulty)]
            naive = sum(naive_reward(e.final_state) for e in episodes) / len(episodes)
            naive_all += naive
            verifier_all += evaluation.rate(agent, difficulty)
            if difficulty == "hard":
                naive_hard, verifier_hard = naive, evaluation.rate(agent, difficulty)
        n = len(evaluation.difficulties)
        rows.append((agent, naive_all / n, verifier_all / n, naive_hard, verifier_hard))
    return rows


def print_report(evaluation: Evaluation) -> None:
    """One screen: the table, what the loser gets wrong, and the broken reward."""
    seeds = evaluation.seeds
    difficulties = evaluation.difficulties
    rule = "  " + "-" * 62
    knows = {
        "edd_naive": "knows nothing",
        "law_aware": "+ EU law",
        "full_aware": "+ house rules",
    }

    def columns(values: Sequence[str]) -> str:
        return "".join(f"{v:>9}" for v in values)

    print()
    print("  Feed mill scheduling environment")
    print(
        f"  held-out seeds {seeds.start}-{seeds.stop - 1} · "
        f"{len(seeds)} tasks per difficulty · verifier score"
    )
    print()
    print(f"  {'SUCCESS RATE':<28}{columns([*difficulties, 'all'])}")
    print(rule)
    for agent in AGENTS:
        cells = [f"{evaluation.rate(agent, d):.0%}" for d in difficulties]
        cells.append(f"{evaluation.overall(agent):.1%}")
        print(f"  {agent:<14}{knows.get(agent, ''):<14}{columns(cells)}")
    print()
    gap = [f"{100 * g:+.0f}" for _, g in knowledge_gap(evaluation)]
    print(f"  {'value of company knowledge':<28}{columns(gap)}  points")
    print("  (full_aware - law_aware)")

    print()
    counts = evaluation.violations("law_aware").most_common()
    print(f"  {'WHAT LAW_AWARE GETS WRONG':<28}{'tasks':>9}")
    print(rule)
    for code, count in counts:
        print(f"  {code:<28}{count:>9}")
    print(
        f"  it fails {evaluation.failed('law_aware')} of {evaluation.tasks} tasks, "
        f"every one on a house rule or a due time —"
    )
    print("  it breaks no legal rule anywhere")

    print()
    print(f"  {'THE NAIVE REWARD':<28}{columns(['naive', 'verifier'])}")
    print(rule)
    for agent, naive_all, verifier_all, _, _ in reward_comparison(evaluation):
        print(f"  {agent:<28}{columns([f'{naive_all:.1%}', f'{verifier_all:.1%}'])}")
    illegal = ", ".join(
        f"{share:.0%} ({difficulty})" for difficulty, _, share in naive_reward_summary(evaluation)
    )
    print("  it ranks them backwards; of the edd_naive schedules it pays for,")
    print(f"  {illegal} are illegal")
    print()


def print_performance(evaluation: Evaluation, evaluating: float, total: float) -> None:
    """What it cost to produce the numbers above."""
    per_episode = 1000 * evaluating / max(evaluation.episodes_played, 1)
    print("  PERFORMANCE")
    print("  " + "-" * 62)
    print(
        f"  {evaluation.tasks:,} tasks generated · {evaluation.episodes_played:,} episodes"
        f" · {evaluation.actions_taken:,} actions"
    )
    print(
        f"  {evaluating:.1f} s to generate, play and verify "
        f"({per_episode:.1f} ms per episode) · {total:.1f} s total"
    )
    print()


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
        "## The naive reward ranks the agents backwards",
        "",
        "| agent | naive reward | verifier | naive (hard) | verifier (hard) |",
        "|---|---:|---:|---:|---:|",
    ]
    for agent, naive_all, verifier_all, naive_hard, verifier_hard in reward_comparison(evaluation):
        lines.append(
            f"| `{agent}` | {naive_all:.0%} | {verifier_all:.1%} | "
            f"{naive_hard:.0%} | {verifier_hard:.0%} |"
        )
    lines += [
        "",
        "Under the naive reward the three agents are within three points of each "
        "other and the *worst* one leads, because flushes and lab holds cost time "
        "and lateness is all that reward can see. Under the verifier they separate "
        "cleanly. Training on the naive reward does not merely tolerate the illegal "
        "policy, it selects for it.",
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
    """The first task where full_aware succeeds, edd_naive cheats, and a flush
    is what separates them.

    "Cheats" means the naive reward pays and the verifier does not. The flush
    matters for the picture: on a task whose feeds never contaminate each
    other, the valid schedule contains no flush at all, and the two charts
    then show the line rules but not the carry-over that motivates them.
    Falls back to the looser criterion if no task satisfies the strict one.
    """
    fallback: D.Task | None = None
    for difficulty in difficulties:
        for seed in seeds:
            task = G.generate(seed, difficulty).task
            good = run_episode(AGENTS["full_aware"](), task)
            bad = run_episode(AGENTS["edd_naive"](), task)
            if not (good.score == 1 and bad.score == 0 and naive_reward(bad.final_state) == 1.0):
                continue
            if any(action["tool"] == D.TOOL_FLUSH for action in good.actions):
                return task
            fallback = fallback or task
    return fallback


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
    print(f"  {task.task_id}   ({path})")
    print(
        f"  {len(task.orders)} orders · {len(task.lines)} line(s) · "
        f"{len(task.house_rules)} house rule(s) · {len(task.feeds)} feeds"
    )
    print(f"  task_hash {task.task_hash[:32]}...")
    print()
    print(f"  {'AGENT':<14}{'SCORE':>7}   violations")
    print("  " + "-" * 62)
    best: EpisodeResult | None = None
    for name, make in AGENTS.items():
        episode = run_episode(make(), task)
        codes = ", ".join(episode.violations) if episode.violations else "none"
        print(f"  {name:<14}{episode.score:>7}   {codes}")
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
    print(f"  wrote {chart}")
    print()
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

    started = time.perf_counter()
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    evaluation = evaluate(seeds, args.difficulties)
    evaluating = time.perf_counter() - started
    print_report(evaluation)

    charts: dict[str, Path] = {}
    if not args.no_charts:
        task = pick_demo_task(seeds, tuple(reversed(args.difficulties)))
        if task is not None:
            charts = write_charts(args.out, task)

    report = write_report(args.out / "results.md", evaluation, charts)
    print_performance(evaluation, evaluating, time.perf_counter() - started)
    for written in (report, *charts.values()):
        print(f"  wrote {written}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
