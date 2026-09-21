"""Baseline tests and the experiment of SPEC section 12.

Everything here runs on the held-out seeds 10000-10099, which the generator
never sees during development.
"""

from __future__ import annotations

import dataclasses

import pytest

from feedmill import domain as D
from feedmill import generator as G
from feedmill.agents import AGENTS, EpisodeResult, full_aware, law_aware, run_episode
from feedmill.naive_reward import exploit_a_plan

DIFFICULTIES = ("easy", "medium", "hard")
HELD_OUT = range(10000, 10100)

LEGAL_CODES = frozenset(
    {
        D.V_L1_COCCIDIOSTAT,
        D.V_L2_ANTIMICROBIAL,
        D.V_L3_RUMINANT_BAN,
        D.V_L4_PAP_LINE,
        D.V_L5_INTRA_SPECIES,
        D.V_L6_WRONG_DIE,
    }
)
HOUSE_CODES = frozenset(
    {
        D.V_H1_CUSTOMER_FLUSH,
        D.V_H2_SEQUENCE_BAN,
        D.V_H4_DIE_CHANGE_TIME,
        D.V_H5_LAB_HOLD,
        D.V_H6_COPPER_SHEEP_FLUSH,
    }
)


@pytest.fixture(scope="module")
def runs() -> dict[tuple[str, str], list[EpisodeResult]]:
    out: dict[tuple[str, str], list[EpisodeResult]] = {}
    for difficulty in DIFFICULTIES:
        tasks = [G.generate(seed, difficulty).task for seed in HELD_OUT]
        for name, make in AGENTS.items():
            out[(name, difficulty)] = [run_episode(make(), task) for task in tasks]
    return out


def _rate(runs, agent: str, difficulty: str) -> float:
    episodes = runs[(agent, difficulty)]
    return sum(e.score for e in episodes) / len(episodes)


def _overall(runs, agent: str) -> float:
    return sum(_rate(runs, agent, d) for d in DIFFICULTIES) / len(DIFFICULTIES)


# -- the agents behave like agents -----------------------------------------


def test_agents_only_ever_emit_valid_actions(runs) -> None:
    for (agent, difficulty), episodes in runs.items():
        for episode in episodes:
            statuses = {entry["status"] for entry in episode.final_state["log"]}
            assert statuses <= {D.STATUS_OK}, f"{agent} on {episode.task_id}: {statuses}"


def test_agents_always_finish_within_the_step_budget(runs) -> None:
    for (agent, _), episodes in runs.items():
        for episode in episodes:
            assert episode.actions[-1] == {"tool": D.TOOL_FINISH}, agent
            assert episode.steps <= D.DEFAULT_MAX_STEPS
            assert D.V_NOT_FINISHED not in episode.violations


def test_every_agent_produces_every_order(runs) -> None:
    for (agent, _), episodes in runs.items():
        for episode in episodes:
            assert D.V_MISSING_ORDER not in episode.violations, agent
            assert D.V_DUPLICATE_ORDER not in episode.violations, agent


# -- what each level of knowledge buys -------------------------------------


def test_edd_naive_never_flushes(runs) -> None:
    for difficulty in DIFFICULTIES:
        for episode in runs[("edd_naive", difficulty)]:
            assert not any(a["tool"] == D.TOOL_FLUSH for a in episode.actions)


def test_edd_naive_is_exploit_a_played_as_a_policy(runs) -> None:
    """The reactive baseline and the planned exploit are the same schedule."""
    for difficulty in DIFFICULTIES:
        for seed, episode in zip(HELD_OUT, runs[("edd_naive", difficulty)]):
            task = G.generate(seed, difficulty).task
            assert episode.actions == exploit_a_plan(task)


def test_law_aware_never_breaks_a_legal_rule(runs) -> None:
    """300 held-out tasks, no L1-L6 anywhere: the law is what it does know."""
    for difficulty in DIFFICULTIES:
        for episode in runs[("law_aware", difficulty)]:
            assert not set(episode.violations) & LEGAL_CODES, (
                f"{episode.task_id}: {episode.violations}"
            )


def test_law_aware_fails_only_on_house_rules_and_lateness(runs) -> None:
    for difficulty in DIFFICULTIES:
        for episode in runs[("law_aware", difficulty)]:
            assert set(episode.violations) <= HOUSE_CODES | {D.V_LATE}


def test_full_aware_fails_only_on_lateness_or_a_stranded_sequence_ban(runs) -> None:
    """Greedy earliest-due-date has no lookahead.

    It can leave the two feeds of a sequence ban (H2) as the last two orders
    on a line, with nothing left to put between them. The planted solution
    proves the task was solvable, so this is the baseline being simple, not
    the task being impossible.
    """
    for difficulty in DIFFICULTIES:
        for episode in runs[("full_aware", difficulty)]:
            assert set(episode.violations) <= {D.V_LATE, D.V_H2_SEQUENCE_BAN}, (
                f"{episode.task_id}: {episode.violations}"
            )


# -- success rates (SPEC section 12) ---------------------------------------


def test_more_knowledge_never_scores_worse(runs) -> None:
    for difficulty in DIFFICULTIES:
        edd = _rate(runs, "edd_naive", difficulty)
        law = _rate(runs, "law_aware", difficulty)
        full = _rate(runs, "full_aware", difficulty)
        assert edd <= law <= full, f"{difficulty}: {edd:.2f} {law:.2f} {full:.2f}"


def test_the_measured_success_rates_stay_in_range(runs) -> None:
    """Loose bounds around the numbers reported in the README."""
    assert _overall(runs, "edd_naive") <= 0.40
    assert 0.30 <= _overall(runs, "law_aware") <= 0.60
    assert _overall(runs, "full_aware") >= 0.85
    assert _rate(runs, "full_aware", "hard") >= 0.60
    assert _rate(runs, "edd_naive", "hard") <= 0.10


# -- the experiment: what company knowledge is worth ------------------------


def test_the_gap_is_positive_where_house_rules_exist(runs) -> None:
    """medium and hard always carry house rules, so the gap must open up."""
    for difficulty in ("medium", "hard"):
        gap = _rate(runs, "full_aware", difficulty) - _rate(runs, "law_aware", difficulty)
        assert gap > 0.25, f"{difficulty}: gap {gap:.2f}"


def test_the_control_holds_on_tasks_with_no_house_rules(runs) -> None:
    """SPEC section 12: without house rules the gap must be about zero.

    Here it is exactly zero, and for a reason stronger than a measurement:
    the two agents share one policy and differ only in a flag, so with no
    house rules to read they take the same actions step for step. If this
    ever fails, the gap is measuring something other than house rules.
    """
    for difficulty in DIFFICULTIES:
        for seed in range(10000, 10030):
            task = dataclasses.replace(G.generate(seed, difficulty).task, house_rules=())
            without = run_episode(law_aware(), task)
            withall = run_episode(full_aware(), task)
            assert without.actions == withall.actions
            assert without.score == withall.score


def test_the_natural_control_on_easy_tasks_without_house_rules(runs) -> None:
    """The same control without touching the generated task at all."""
    seeds = [s for s in HELD_OUT if not G.generate(s, "easy").task.house_rules]
    assert len(seeds) > 20, "not enough easy tasks without house rules to compare"
    gap = 0
    for seed in seeds:
        task = G.generate(seed, "easy").task
        gap += run_episode(full_aware(), task).score - run_episode(law_aware(), task).score
    assert gap == 0


def test_the_law_aware_violation_histogram_is_dominated_by_house_rules(runs) -> None:
    counts: dict[str, int] = {}
    for difficulty in DIFFICULTIES:
        for episode in runs[("law_aware", difficulty)]:
            for code in episode.violations:
                counts[code] = counts.get(code, 0) + 1
    assert counts, "law_aware should fail on something"
    house = sum(n for code, n in counts.items() if code in HOUSE_CODES)
    assert house > sum(counts.values()) / 2
    assert max(counts, key=counts.get) == D.V_H1_CUSTOMER_FLUSH
