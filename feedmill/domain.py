"""Domain model of the feed mill: substances, feeds, lines, orders, rules.

Shared by the environment, the generator and the verifier.

Per SPEC section 9 the verifier imports *only* constants and data classes from
this module. All rule logic therefore lives in the module that needs it: the
generator checks its planted plan with its own code, the verifier re-derives
everything from the log with its own code. The duplication is deliberate --
it keeps the verifier independent of the code that produced the state it
checks, and makes the "planted solution scores 1" test a real cross-check
instead of a tautology.

The helper functions at the bottom (concentration update, production time,
canonical JSON) are used by ``env.py``, ``generator.py`` and ``agents.py``,
never by ``verifier.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# --------------------------------------------------------------------------
# Process constants (SPEC sections 2, 3, 4, 5, 7)
# --------------------------------------------------------------------------

#: Minute 0 = 06:00, one day of production. [DECISION] SPEC 2.6
HORIZON_MIN: int = 1440

#: Throughput of a line. [ASSUMPTION] SPEC 2.4
DEFAULT_RATE_T_PER_H: float = 20.0

#: Share of the previous batch that the next batch picks up. [ASSUMPTION] SPEC 3
CARRY_OVER_RATE: float = 0.02

#: A flush is a small batch of plain grain that is run through and discarded.
FLUSH_TONNES: float = 1.0
FLUSH_MINUTES: int = 15

#: Die change duration, and the slow variant of house rule H3. [ASSUMPTION]
DIE_CHANGE_MINUTES: int = 45
SLOW_DIE_CHANGE_MINUTES: int = 90

#: House rule H4: die changes may only *start* in [0, 960] (06:00-22:00).
DIE_CHANGE_LATEST_START_MIN: int = 960

#: House rule H5: lab release hold after a medicated batch.
LAB_HOLD_MINUTES: int = 120

#: Episode truncation. SPEC 7
DEFAULT_MAX_STEPS: int = 200

#: Tolerance for concentration limit comparisons (limits are "<=").
LIMIT_TOL: float = 1e-9

# --------------------------------------------------------------------------
# Substances and carry-over classes (SPEC 2.1, 2.2)
# --------------------------------------------------------------------------

MONENSIN = "monensin"
ANTIMICROBIAL = "antimicrobial"
COPPER_HIGH = "copper_high"
PAP_PIG_SUBSTANCE = "pap_pig"
PAP_POULTRY_SUBSTANCE = "pap_poultry"

SUBSTANCES: tuple[str, ...] = (
    MONENSIN,
    ANTIMICROBIAL,
    COPPER_HIGH,
    PAP_PIG_SUBSTANCE,
    PAP_POULTRY_SUBSTANCE,
)

CLASS_TARGET = "target"
CLASS_SENSITIVE = "sensitive"
CLASS_LESS_SENSITIVE = "less_sensitive"

#: L1: maximum monensin carry-over into a feed that does not contain it,
#: as a fraction of the authorised level in the target feed. SPEC 4 (L1)
COCCIDIOSTAT_LIMIT: dict[str, float] = {
    CLASS_SENSITIVE: 0.01,
    CLASS_LESS_SENSITIVE: 0.03,
}

#: L2: maximum antimicrobial carry-over into any non-target feed.
#: [DECISION] one limit instead of substance-specific levels. SPEC 4 (L2)
ANTIMICROBIAL_LIMIT: float = 0.01

# --------------------------------------------------------------------------
# Species and PAP lines (SPEC 2.3, 2.4, 4)
# --------------------------------------------------------------------------

PAP_NONE = "none"
PAP_PIG = "pig"
PAP_POULTRY = "poultry"
PAP_TYPES: tuple[str, ...] = (PAP_NONE, PAP_PIG, PAP_POULTRY)

#: The PAP type a species belongs to, for the intra-species ban L5.
#: ``None`` means the species is not a PAP source in this model.
SPECIES_PAP_GROUP: dict[str, str | None] = {
    "broiler": PAP_POULTRY,
    "laying_hen": PAP_POULTRY,
    "pig": PAP_PIG,
    "cattle": None,
    "sheep": None,
    "horse": None,
}

#: The line type a PAP substance requires under the derogation L4.
PAP_SUBSTANCE_LINE: dict[str, str] = {
    PAP_PIG_SUBSTANCE: PAP_PIG,
    PAP_POULTRY_SUBSTANCE: PAP_POULTRY,
}

# --------------------------------------------------------------------------
# Violation codes (SPEC 9)
# --------------------------------------------------------------------------

V_L1_COCCIDIOSTAT = "L1_COCCIDIOSTAT"
V_L2_ANTIMICROBIAL = "L2_ANTIMICROBIAL"
V_L3_RUMINANT_BAN = "L3_RUMINANT_BAN"
V_L4_PAP_LINE = "L4_PAP_LINE"
V_L5_INTRA_SPECIES = "L5_INTRA_SPECIES"
V_L6_WRONG_DIE = "L6_WRONG_DIE"
V_H1_CUSTOMER_FLUSH = "H1_CUSTOMER_FLUSH"
V_H2_SEQUENCE_BAN = "H2_SEQUENCE_BAN"
V_H4_DIE_CHANGE_TIME = "H4_DIE_CHANGE_TIME"
V_H5_LAB_HOLD = "H5_LAB_HOLD"
V_H6_COPPER_SHEEP_FLUSH = "H6_COPPER_SHEEP_FLUSH"
V_LATE = "LATE"
V_MISSING_ORDER = "MISSING_ORDER"
V_DUPLICATE_ORDER = "DUPLICATE_ORDER"
V_INVALID_ACTION = "INVALID_ACTION"
V_NOT_FINISHED = "NOT_FINISHED"
V_TAMPERED_LOG = "TAMPERED_LOG"

#: H3 (slow die) never produces a violation code: it changes how long a die
#: change physically takes, so the environment and the verifier both apply it
#: when they advance the clock.

# --------------------------------------------------------------------------
# House rule templates (SPEC 5)
# --------------------------------------------------------------------------

H1_CUSTOMER_FLUSH = "H1"
H2_SEQUENCE_BAN = "H2"
H3_SLOW_DIE = "H3"
H4_DIE_CHANGE_DAY_ONLY = "H4"
H5_LAB_HOLD = "H5"
H6_COPPER_SHEEP_FLUSH = "H6"

HOUSE_RULE_TEMPLATES: tuple[str, ...] = (
    H1_CUSTOMER_FLUSH,
    H2_SEQUENCE_BAN,
    H3_SLOW_DIE,
    H4_DIE_CHANGE_DAY_ONLY,
    H5_LAB_HOLD,
    H6_COPPER_SHEEP_FLUSH,
)

# --------------------------------------------------------------------------
# Tools (SPEC 6)
# --------------------------------------------------------------------------

TOOL_PRODUCE = "produce"
TOOL_FLUSH = "flush"
TOOL_CHANGE_DIE = "change_die"
TOOL_WAIT = "wait"
TOOL_FINISH = "finish"

TOOLS: tuple[str, ...] = (
    TOOL_PRODUCE,
    TOOL_FLUSH,
    TOOL_CHANGE_DIE,
    TOOL_WAIT,
    TOOL_FINISH,
)

STATUS_OK = "ok"
STATUS_INVALID = "invalid"


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LegalRule:
    """One EU rule, structured for code and in one sentence for a reader."""

    id: str
    kind: str
    params: dict[str, Any]
    text: str
    basis: str

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": dict(self.params),
            "text": self.text,
            "basis": self.basis,
        }


@dataclass(frozen=True)
class Feed:
    """One product in the feed catalog (SPEC 2.3)."""

    id: str
    species: str
    ruminant: bool
    substances: tuple[str, ...]
    carry_over_class: str
    die_mm: int

    def contains(self, substance: str) -> bool:
        return substance in self.substances

    @property
    def pap_substance(self) -> str | None:
        for s in self.substances:
            if s in PAP_SUBSTANCE_LINE:
                return s
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "species": self.species,
            "ruminant": self.ruminant,
            "substances": list(self.substances),
            "carry_over_class": self.carry_over_class,
            "die_mm": self.die_mm,
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "Feed":
        return Feed(
            id=str(d["id"]),
            species=str(d["species"]),
            ruminant=bool(d["ruminant"]),
            substances=tuple(str(s) for s in d["substances"]),
            carry_over_class=str(d["carry_over_class"]),
            die_mm=int(d["die_mm"]),
        )


@dataclass(frozen=True)
class LineSpec:
    """A production line (SPEC 2.4)."""

    id: str
    rate_t_per_h: float
    pap_type: str
    initial_die_mm: int
    initial_concentrations: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rate_t_per_h": self.rate_t_per_h,
            "pap_type": self.pap_type,
            "initial_die_mm": self.initial_die_mm,
            "initial_concentrations": zero_concentrations() | dict(self.initial_concentrations),
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "LineSpec":
        return LineSpec(
            id=str(d["id"]),
            rate_t_per_h=float(d["rate_t_per_h"]),
            pap_type=str(d["pap_type"]),
            initial_die_mm=int(d["initial_die_mm"]),
            initial_concentrations=zero_concentrations()
            | {str(k): float(v) for k, v in dict(d.get("initial_concentrations", {})).items()},
        )


@dataclass(frozen=True)
class Order:
    """A customer order for one feed (SPEC 2.5)."""

    id: str
    customer: str
    feed: str
    tonnes: float
    due: int

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "customer": self.customer,
            "feed": self.feed,
            "tonnes": self.tonnes,
            "due": self.due,
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "Order":
        return Order(
            id=str(d["id"]),
            customer=str(d["customer"]),
            feed=str(d["feed"]),
            tonnes=float(d["tonnes"]),
            due=int(d["due"]),
        )


@dataclass(frozen=True)
class HouseRule:
    """A company-specific rule (SPEC 5): template id, parameters, text."""

    id: str
    params: dict[str, Any]
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "params": dict(self.params), "text": self.text}

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "HouseRule":
        return HouseRule(
            id=str(d["id"]),
            params=dict(d.get("params", {})),
            text=str(d["text"]),
        )


@dataclass(frozen=True)
class Task:
    """A complete scheduling task. Hashed into the log chain (SPEC 8)."""

    task_id: str
    seed: int
    difficulty: str
    horizon_min: int
    carry_over_rate: float
    flush_tonnes: float
    flush_minutes: int
    die_change_minutes: int
    max_steps: int
    feeds: tuple[Feed, ...]
    lines: tuple[LineSpec, ...]
    orders: tuple[Order, ...]
    house_rules: tuple[HouseRule, ...]

    # -- lookups -----------------------------------------------------------

    @property
    def feed_map(self) -> dict[str, Feed]:
        return {f.id: f for f in self.feeds}

    @property
    def line_map(self) -> dict[str, LineSpec]:
        return {ln.id: ln for ln in self.lines}

    @property
    def order_map(self) -> dict[str, Order]:
        return {o.id: o for o in self.orders}

    def house_rule(self, template: str) -> HouseRule | None:
        """First house rule of the given template, or None."""
        for rule in self.house_rules:
            if rule.id == template:
                return rule
        return None

    def house_rules_of(self, template: str) -> tuple[HouseRule, ...]:
        return tuple(r for r in self.house_rules if r.id == template)

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "horizon_min": self.horizon_min,
            "carry_over_rate": self.carry_over_rate,
            "flush_tonnes": self.flush_tonnes,
            "flush_minutes": self.flush_minutes,
            "die_change_minutes": self.die_change_minutes,
            "max_steps": self.max_steps,
            "feeds": [f.to_json() for f in self.feeds],
            "lines": [ln.to_json() for ln in self.lines],
            "orders": [o.to_json() for o in self.orders],
            "house_rules": [r.to_json() for r in self.house_rules],
        }

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "Task":
        return Task(
            task_id=str(d["task_id"]),
            seed=int(d["seed"]),
            difficulty=str(d["difficulty"]),
            horizon_min=int(d.get("horizon_min", HORIZON_MIN)),
            carry_over_rate=float(d.get("carry_over_rate", CARRY_OVER_RATE)),
            flush_tonnes=float(d.get("flush_tonnes", FLUSH_TONNES)),
            flush_minutes=int(d.get("flush_minutes", FLUSH_MINUTES)),
            die_change_minutes=int(d.get("die_change_minutes", DIE_CHANGE_MINUTES)),
            max_steps=int(d.get("max_steps", DEFAULT_MAX_STEPS)),
            feeds=tuple(Feed.from_json(f) for f in d["feeds"]),
            lines=tuple(LineSpec.from_json(ln) for ln in d["lines"]),
            orders=tuple(Order.from_json(o) for o in d["orders"]),
            house_rules=tuple(HouseRule.from_json(r) for r in d.get("house_rules", [])),
        )

    @property
    def task_hash(self) -> str:
        return sha256_hex(canonical_json(self.to_json()))


# --------------------------------------------------------------------------
# Feed catalog (SPEC 2.3)
# --------------------------------------------------------------------------

FEED_CATALOG: tuple[Feed, ...] = (
    Feed("broiler_mon", "broiler", False, (MONENSIN,), CLASS_TARGET, 3),
    Feed("broiler_withdrawal", "broiler", False, (), CLASS_SENSITIVE, 3),
    Feed("broiler_pigpap", "broiler", False, (PAP_PIG_SUBSTANCE,), CLASS_SENSITIVE, 3),
    Feed("layer", "laying_hen", False, (), CLASS_SENSITIVE, 3),
    Feed("piglet_starter", "pig", False, (COPPER_HIGH,), CLASS_LESS_SENSITIVE, 3),
    Feed("pig_grower", "pig", False, (), CLASS_LESS_SENSITIVE, 4),
    Feed("pig_medicated", "pig", False, (ANTIMICROBIAL,), CLASS_LESS_SENSITIVE, 4),
    Feed("pig_poultrypap", "pig", False, (PAP_POULTRY_SUBSTANCE,), CLASS_LESS_SENSITIVE, 4),
    Feed("dairy", "cattle", True, (), CLASS_SENSITIVE, 6),
    Feed("sheep", "sheep", True, (), CLASS_LESS_SENSITIVE, 4),
    Feed("horse", "horse", False, (), CLASS_SENSITIVE, 6),
)

FEED_BY_ID: dict[str, Feed] = {f.id: f for f in FEED_CATALOG}

#: Feeds that carry processed animal protein and therefore need a dedicated
#: line under the derogation L4.
PAP_FEED_IDS: tuple[str, ...] = tuple(f.id for f in FEED_CATALOG if f.pap_substance)

DIE_SIZES: tuple[int, ...] = tuple(sorted({f.die_mm for f in FEED_CATALOG}))

#: The EU rules of SPEC section 4, structured and in one sentence each.
#: They are part of every observation (SPEC section 7).
LEGAL_RULES: tuple[LegalRule, ...] = (
    LegalRule(
        id="L1",
        kind="carry_over_limit",
        params={"substance": MONENSIN, "limits": dict(COCCIDIOSTAT_LIMIT)},
        text=(
            "Monensin carried over into a feed that does not contain it must stay "
            "at or below 1 % of the authorised level for sensitive feeds and 3 % "
            "for less sensitive feeds."
        ),
        basis="Directive 2009/8/EC amending Annex I of Directive 2002/32/EC",
    ),
    LegalRule(
        id="L2",
        kind="carry_over_limit",
        params={"substance": ANTIMICROBIAL, "limit": ANTIMICROBIAL_LIMIT},
        text=(
            "An antimicrobial carried over into any feed that does not contain it "
            "must stay at or below 1 % of the authorised level."
        ),
        basis="Regulation (EU) 2019/4 Article 7; Delegated Regulation (EU) 2024/1229",
    ),
    LegalRule(
        id="L3",
        kind="line_restriction",
        params={"ruminant_feeds_allowed_pap_types": [PAP_NONE]},
        text=(
            "Ruminant feeds may never be produced on a line that handles processed "
            "animal protein, whatever is flushed in between."
        ),
        basis="Regulation (EC) No 999/2001 Article 7(1) and Annex IV",
    ),
    LegalRule(
        id="L4",
        kind="line_restriction",
        params={"required_line_pap_type": dict(PAP_SUBSTANCE_LINE)},
        text=(
            "Porcine PAP may be used in poultry feed and poultry PAP in pig feed, "
            "but only on a line dedicated to that PAP type."
        ),
        basis="Commission Regulation (EU) 2021/1372 amending Annex IV of Regulation (EC) No 999/2001",
    ),
    LegalRule(
        id="L5",
        kind="line_restriction",
        params={
            "species_pap_group": {k: v for k, v in SPECIES_PAP_GROUP.items() if v is not None}
        },
        text=(
            "No feed for a species may be produced on a line whose PAP type is that "
            "same species: pig feeds never on a pig line, poultry feeds never on a "
            "poultry line."
        ),
        basis="Regulation (EC) No 1069/2009 Article 11(1)(a)",
    ),
    LegalRule(
        id="L6",
        kind="physical",
        params={"die_change_minutes": DIE_CHANGE_MINUTES},
        text=(
            "A feed can only be pelleted with its own die; changing the die takes "
            "45 minutes. Producing with the wrong die is impossible, not illegal."
        ),
        basis="Physical constraint of the press, not law",
    ),
)


# --------------------------------------------------------------------------
# Helpers (not used by verifier.py -- see module docstring)
# --------------------------------------------------------------------------


def zero_concentrations() -> dict[str, float]:
    """A clean line: no residue of any tracked substance."""
    return {s: 0.0 for s in SUBSTANCES}


def next_concentrations(
    previous: Mapping[str, float],
    substances: Iterable[str],
    carry_over_rate: float = CARRY_OVER_RATE,
) -> dict[str, float]:
    """Concentration vector of a batch (SPEC 3).

    ``own content (1.0 if the batch contains s) + r * previous batch``.
    A flush is simply a batch that contains nothing.
    """
    contained = set(substances)
    return {
        s: (1.0 if s in contained else 0.0) + carry_over_rate * float(previous.get(s, 0.0))
        for s in SUBSTANCES
    }


def production_minutes(tonnes: float, rate_t_per_h: float) -> int:
    """ceil(tonnes / rate * 60), guarded against float noise (SPEC 2.5)."""
    exact = tonnes * 60.0 / rate_t_per_h
    return int(math.ceil(exact - 1e-9))


def die_change_minutes(task: Task, line_id: str) -> int:
    """Die change duration on a line, taking house rule H3 into account."""
    for rule in task.house_rules_of(H3_SLOW_DIE):
        if rule.params.get("line") == line_id:
            return int(rule.params.get("minutes", SLOW_DIE_CHANGE_MINUTES))
    return task.die_change_minutes


def canonical_json(obj: Any) -> str:
    """Canonical JSON used for every hash in this project."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
