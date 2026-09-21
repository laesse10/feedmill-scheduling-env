"""The task JSON schema and the example task (SPEC section 13).

`jsonschema` is not a dependency, so the subset of JSON Schema the file
actually uses is checked by the small validator below. It supports exactly
the keywords `schemas/task.schema.json` relies on, and the last test proves
it rejects what it should.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from feedmill import domain as D
from feedmill import generator as G
from feedmill.agents import full_aware, run_episode

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schemas" / "task.schema.json"
EXAMPLE_PATH = ROOT / "examples" / "example_task.json"

TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
}


def _is_type(value: Any, expected: str) -> bool:
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return isinstance(value, TYPES[expected]) and not isinstance(value, bool)


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """A validator for the JSON Schema subset this project uses."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected and not _is_type(instance, expected):
        return [f"{path}: expected {expected}, got {type(instance).__name__}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")

    if expected == "object":
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties")
        if extra is False:
            for key in set(instance) - set(properties):
                errors.append(f"{path}: unexpected property {key!r}")
        elif isinstance(extra, dict):
            for key, value in instance.items():
                if key not in properties:
                    errors += validate(value, extra, f"{path}.{key}")
        for key, sub in properties.items():
            if key in instance:
                errors += validate(instance[key], sub, f"{path}.{key}")

    elif expected == "array":
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: needs at least {schema['minItems']} items")
        if schema.get("uniqueItems") and len({json.dumps(i, sort_keys=True) for i in instance}) != len(instance):
            errors.append(f"{path}: items must be unique")
        if "items" in schema:
            for index, item in enumerate(instance):
                errors += validate(item, schema["items"], f"{path}[{index}]")

    elif expected in ("number", "integer"):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")

    return errors


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


@pytest.fixture(scope="module")
def example() -> dict[str, Any]:
    return json.loads(EXAMPLE_PATH.read_text())


# -- the schema ------------------------------------------------------------


def test_the_schema_is_well_formed(schema) -> None:
    assert schema["$schema"].startswith("https://json-schema.org/draft/")
    assert set(schema["required"]) == {"task_id", "feeds", "lines", "orders"}
    assert schema["additionalProperties"] is False
    for name, prop in schema["properties"].items():
        assert "type" in prop, name
        assert prop.get("description"), f"{name} has no description"


def test_the_schema_covers_every_field_the_task_writes(schema) -> None:
    task = G.generate(0, "hard").task
    assert set(task.to_json()) <= set(schema["properties"])


# -- the example task ------------------------------------------------------


def test_the_example_task_validates(example, schema) -> None:
    assert validate(example, schema) == []


def test_the_example_task_round_trips(example) -> None:
    task = D.Task.from_json(example)
    assert task.task_id == "muehle-thurtal-2024-03-11"
    assert D.Task.from_json(task.to_json()).task_hash == task.task_hash


def test_the_example_task_is_a_mill_not_a_generated_task(example) -> None:
    """It carries no seed or difficulty, and its own shorter catalog."""
    assert "seed" not in example and "difficulty" not in example
    task = D.Task.from_json(example)
    assert task.difficulty == "external"
    assert len(task.feeds) < len(D.FEED_CATALOG)
    assert {line.rate_t_per_h for line in task.lines} == {22.5, 18.0}
    assert any(order.tonnes % 1 for order in task.orders), "real tonnages are not round"


def test_the_example_task_is_solvable(example) -> None:
    task = D.Task.from_json(example)
    assert run_episode(full_aware(), task).score == 1


# -- generated tasks conform too -------------------------------------------


@pytest.mark.parametrize("difficulty", ("easy", "medium", "hard"))
def test_every_generated_task_validates_against_the_schema(difficulty: str, schema) -> None:
    for seed in range(20):
        task = G.generate(seed, difficulty).task
        assert validate(task.to_json(), schema) == [], task.task_id


# -- the validator has teeth -----------------------------------------------


@pytest.mark.parametrize(
    "break_it,expected",
    [
        (lambda t: t.pop("orders"), "missing required property 'orders'"),
        (lambda t: t.update(surprise=1), "unexpected property 'surprise'"),
        (lambda t: t["orders"][0].update(tonnes=-5), "<= exclusiveMinimum"),
        (lambda t: t["orders"][0].update(due="soon"), "expected integer"),
        (lambda t: t["lines"][0].update(pap_type="fish"), "is not one of"),
        (lambda t: t["feeds"][0].update(carry_over_class="very"), "is not one of"),
        (lambda t: t["feeds"][0]["substances"].append("glitter"), "is not one of"),
        (lambda t: t.update(feeds=[]), "needs at least 1 items"),
        (lambda t: t.update(carry_over_rate=2.0), "> maximum"),
    ],
)
def test_the_validator_rejects_a_broken_task(example, schema, break_it, expected: str) -> None:
    broken = copy.deepcopy(example)
    break_it(broken)
    errors = validate(broken, schema)
    assert errors, "the validator accepted a broken task"
    assert any(expected in error for error in errors), errors
