"""The one command of the case study: python run.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import run
from feedmill import domain as D
from feedmill import generator as G
from feedmill.agents import AGENTS, run_episode
from feedmill.naive_reward import naive_reward

ROOT = Path(__file__).resolve().parent.parent


def test_the_default_run_prints_the_table_and_writes_the_outputs(tmp_path, capsys) -> None:
    assert run.main(["--seeds", "3", "--out", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    for agent in AGENTS:
        assert agent in printed
    assert "SUCCESS RATE" in printed
    assert "value of company knowledge" in printed
    assert "WHAT LAW_AWARE GETS WRONG" in printed
    assert "THE NAIVE REWARD" in printed
    assert "PERFORMANCE" in printed
    assert "episodes" in printed and "ms per episode" in printed
    assert not any(line != line.rstrip() for line in printed.splitlines()), (
        "no trailing whitespace in the report"
    )

    report = tmp_path / "results.md"
    assert report.exists()
    text = report.read_text()
    assert "# Results" in text
    for agent in AGENTS:
        assert f"`{agent}`" in text
    assert "gantt_valid.png" in text and "gantt_exploit.png" in text

    for name in ("gantt_valid.png", "gantt_exploit.png"):
        chart = tmp_path / name
        assert chart.exists() and chart.stat().st_size > 5_000
        assert chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_run_is_reproducible(tmp_path) -> None:
    """Same command, same bytes: no timestamps in the report or the charts."""
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        run.main(["--seeds", "3", "--out", str(out)])
    for name in ("results.md", "gantt_valid.png", "gantt_exploit.png"):
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == hashlib.sha256(
            (second / name).read_bytes()
        ).hexdigest(), name


def test_no_charts_skips_the_charts(tmp_path) -> None:
    assert run.main(["--seeds", "2", "--out", str(tmp_path), "--no-charts"]) == 0
    assert (tmp_path / "results.md").exists()
    assert not list(tmp_path.glob("*.png"))


def test_a_subset_of_difficulties(tmp_path, capsys) -> None:
    assert run.main(["--seeds", "2", "--difficulties", "easy", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "results.md").read_text()
    assert "| easy |" in text or "| easy " in text
    assert "medium" not in text.split("## Value")[0]


# -- the external task (SPEC section 13) -----------------------------------


def test_the_example_task_file_runs(tmp_path, capsys) -> None:
    example = ROOT / "examples" / "example_task.json"
    assert run.main(["--task-file", str(example), "--out", str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert "muehle-thurtal" in printed
    assert "task_hash" in printed
    for agent in AGENTS:
        assert agent in printed

    charts = list(tmp_path.glob("gantt_*.png"))
    assert len(charts) == 1 and charts[0].stat().st_size > 5_000


def test_a_task_file_written_by_hand_needs_no_seed(tmp_path) -> None:
    task = G.generate(0, "easy").task
    document = task.to_json()
    del document["seed"], document["difficulty"]
    path = tmp_path / "hand_written.json"
    path.write_text(json.dumps(document))
    assert run.main(["--task-file", str(path), "--out", str(tmp_path)]) == 0


# -- the demo task behind the two charts -----------------------------------


def test_the_demo_task_shows_a_real_exploit() -> None:
    task = run.pick_demo_task(range(10000, 10010), ("hard", "medium", "easy"))
    assert task is not None

    good = run_episode(AGENTS["full_aware"](), task)
    bad = run_episode(AGENTS["edd_naive"](), task)
    assert good.score == 1, "the valid chart must show a schedule that passes"
    assert bad.score == 0, "the exploit chart must show one that does not"
    assert naive_reward(bad.final_state) == 1.0, "and the naive reward must pay for it"
    assert any(a["tool"] == D.TOOL_FLUSH for a in good.actions), (
        "the valid chart must contain a flush, or the pair does not show carry-over"
    )


def test_the_evaluation_matches_running_the_agents_directly() -> None:
    evaluation = run.evaluate(range(10000, 10005), ["medium"])
    task = G.generate(10000, "medium").task
    assert evaluation.episodes[("law_aware", "medium")][0].score == run_episode(
        AGENTS["law_aware"](), task
    ).score
    assert 0.0 <= evaluation.overall("full_aware") <= 1.0
