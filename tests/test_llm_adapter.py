"""The language model adapter: schemas, rendering, parsing. No API calls."""

from __future__ import annotations

import json

import pytest

from feedmill import domain as D
from feedmill import generator as G
from feedmill.env import ACTION_KEYS, FeedMillEnv
from feedmill.llm_adapter import (
    TOOL_SCHEMAS,
    observation_to_text,
    parse_action,
    system_prompt,
)


@pytest.fixture(scope="module")
def observation() -> dict:
    env = FeedMillEnv()
    obs = env.reset(task=G.generate(10000, "hard").task)
    obs, *_ = env.step({"tool": "produce", "line": "L1", "order": "O3"})
    obs, *_ = env.step({"tool": "produce", "line": "L1", "order": "O404"})
    return obs


# -- tool schemas ----------------------------------------------------------


def test_the_schemas_describe_exactly_the_five_tools() -> None:
    assert {s["name"] for s in TOOL_SCHEMAS} == set(D.TOOLS)


def test_every_schema_matches_what_the_environment_accepts() -> None:
    """A model that follows the schema cannot emit a malformed action."""
    for schema in TOOL_SCHEMAS:
        properties = schema["input_schema"]["properties"]
        assert set(properties) | {"tool"} == ACTION_KEYS[schema["name"]], schema["name"]
        required = set(schema["input_schema"].get("required", []))
        assert required == set(properties), schema["name"]
        assert schema["input_schema"]["additionalProperties"] is False
        assert schema["description"].strip()


def test_an_action_built_from_a_schema_is_accepted(observation) -> None:
    env = FeedMillEnv()
    env.reset(task=G.generate(10000, "medium").task)
    schema = next(s for s in TOOL_SCHEMAS if s["name"] == D.TOOL_FLUSH)
    action = {"tool": schema["name"], **{key: "L1" for key in schema["input_schema"]["properties"]}}
    _, _, _, _, info = env.step(action)
    assert info["status"] == D.STATUS_OK


def test_the_schemas_are_json_serialisable() -> None:
    json.dumps(TOOL_SCHEMAS)


# -- observation to text ---------------------------------------------------


def test_the_text_carries_what_an_agent_needs(observation) -> None:
    text = observation_to_text(observation)
    assert observation["task_id"] in text
    for line in observation["lines"]:
        assert line["id"] in text and line["pap_type"] in text
    for order in observation["open_orders"]:
        assert order["id"] in text and order["feed"] in text and order["customer"] in text
    for rule in observation["legal_rules"] + observation["house_rules"]:
        assert rule["text"] in text
    assert "O3" in text and "on time" in text, "the completed order and its verdict"
    assert "06:00" in text, "clocks are shown as times of day"


def test_the_text_reports_the_rejected_action(observation) -> None:
    text = observation_to_text(observation)
    assert "invalid" in text
    assert observation["last_error"] in text


def test_the_text_never_contains_the_key_or_the_solution() -> None:
    env = FeedMillEnv()
    obs = env.reset(task=G.generate(3, "hard").task)
    text = observation_to_text(obs)
    assert env.secret_key.hex() not in text
    assert "planted" not in text.lower()


def test_the_rendering_is_deterministic(observation) -> None:
    assert observation_to_text(observation) == observation_to_text(observation)


def test_the_system_prompt_names_the_trap() -> None:
    prompt = system_prompt()
    assert "carry-over" in prompt and "flush" in prompt
    assert "one tool call" in prompt


# -- parsing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"tool": "produce", "line": "L1", "order": "O3"}',
        'I will run O3 next.\n{"tool": "produce", "line": "L1", "order": "O3"}',
        '```json\n{"tool": "produce", "line": "L1", "order": "O3"}\n```',
        'Thinking about {braces} first.\n{"tool": "produce", "line": "L1", "order": "O3"}',
    ],
)
def test_parse_action_finds_the_call(reply: str) -> None:
    assert parse_action(reply) == {"tool": "produce", "line": "L1", "order": "O3"}


def test_parse_action_handles_braces_inside_strings() -> None:
    reply = '{"tool": "wait", "line": "L1", "minutes": 30}'
    assert parse_action(reply)["minutes"] == 30
    assert parse_action('{"tool": "flush", "line": "L{1}"}') == {"tool": "flush", "line": "L{1}"}


@pytest.mark.parametrize("reply", ["no json here", "", "[1, 2, 3]", "{not json}"])
def test_parse_action_returns_none_when_there_is_nothing_to_parse(reply: str) -> None:
    assert parse_action(reply) is None


def test_a_parsed_action_can_be_stepped() -> None:
    env = FeedMillEnv()
    env.reset(task=G.generate(0, "easy").task)
    action = parse_action('Let me flush first:\n{"tool": "flush", "line": "L1"}')
    _, _, _, _, info = env.step(action)
    assert info["status"] == D.STATUS_OK
