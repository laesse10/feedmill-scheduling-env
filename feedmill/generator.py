"""Deterministic task generator with a planted solution (SPEC section 11).

``generate(seed, difficulty)`` draws orders and house rules, builds one valid
schedule with a constructive heuristic, and only then sets the due times from
that schedule::

    due = planted completion time * slack + jitter,   jitter >= 0

so every generated task is solvable by construction. The planted plan is
returned next to the task but is never part of the task itself, and therefore
never reaches the observation.

``plan_violations`` re-simulates an action list from scratch and reports every
rule of SPEC sections 4, 5 and 9 that it breaks. It is used to validate the
planted plan at generation time. It is *not* the verifier: it knows nothing
about the tamper-evident log. The verifier checks the same plans a second
time, independently (see ``tests/test_verifier_required.py``).
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import domain as D

Action = dict[str, Any]

#: Upper bound of the uniform jitter added to every due time.
MAX_JITTER_MIN: int = 30

#: How often the generator redraws a task before giving up. A draw is rejected
#: when the constructive heuristic cannot place it inside the day (for example
#: a die change that house rule H4 would push past 22:00).
MAX_ATTEMPTS: int = 200

CUSTOMERS: tuple[str, ...] = (
    "Bosshard AG",
    "Hof Meier",
    "Landi Thurtal",
    "Vogt und Soehne",
    "Zuber Farms",
    "Keller Mast",
)


@dataclass(frozen=True)
class DifficultyConfig:
    """One column of the difficulty table in SPEC section 11."""

    name: str
    n_lines: int
    n_orders: tuple[int, int]
    tonnes: tuple[int, int]
    slack: float
    n_house_rules: tuple[int, int]
    pap_feeds: bool


DIFFICULTIES: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig("easy", 1, (5, 6), (10, 30), 2.0, (0, 1), False),
    "medium": DifficultyConfig("medium", 1, (8, 10), (10, 40), 1.4, (2, 2), False),
    "hard": DifficultyConfig("hard", 2, (12, 16), (10, 40), 1.15, (3, 4), True),
}


@dataclass(frozen=True)
class Segment:
    """One block of occupied line time in the planted plan (for Gantt charts)."""

    line: str
    kind: str  # produce | flush | die_change | wait
    start: int
    end: int
    order: str | None = None
    feed: str | None = None
    concentrations: dict[str, float] | None = None


@dataclass(frozen=True)
class PlantedPlan:
    actions: tuple[Action, ...]
    segments: tuple[Segment, ...]
    completion: dict[str, int]
    makespan: int

    @property
    def batches(self) -> tuple[Segment, ...]:
        return tuple(s for s in self.segments if s.kind in (D.TOOL_PRODUCE, D.TOOL_FLUSH))


@dataclass(frozen=True)
class GeneratedTask:
    task: D.Task
    plan: PlantedPlan


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def generate(seed: int, difficulty: str) -> GeneratedTask:
    """Deterministically build a task and its planted solution."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty!r}, expected one of {sorted(DIFFICULTIES)}")
    cfg = DIFFICULTIES[difficulty]
    for attempt in range(MAX_ATTEMPTS):
        result = _attempt(seed, cfg, attempt)
        if result is not None:
            return result
    raise RuntimeError(f"no solvable {difficulty} task found for seed {seed}")


def generate_task(seed: int, difficulty: str) -> D.Task:
    """The task alone, without the planted solution."""
    return generate(seed, difficulty).task


# --------------------------------------------------------------------------
# Drawing a task
# --------------------------------------------------------------------------


def _rng(seed: int, difficulty: str, attempt: int) -> random.Random:
    return random.Random(f"feedmill|{difficulty}|{seed}|{attempt}")


def _attempt(seed: int, cfg: DifficultyConfig, attempt: int) -> GeneratedTask | None:
    rng = _rng(seed, cfg.name, attempt)

    lines = _draw_lines(rng, cfg)
    orders = _draw_orders(rng, cfg, lines)
    if orders is None:
        return None

    assignment = _assign_lines(orders, lines)
    if assignment is None:
        return None

    house_rules = _draw_house_rules(rng, cfg, orders, lines, assignment)

    # Provisional task: the due times are placeholders until the planted plan
    # exists, because they are derived from it.
    task = D.Task(
        task_id=f"{cfg.name}-{seed}",
        seed=seed,
        difficulty=cfg.name,
        horizon_min=D.HORIZON_MIN,
        carry_over_rate=D.CARRY_OVER_RATE,
        flush_tonnes=D.FLUSH_TONNES,
        flush_minutes=D.FLUSH_MINUTES,
        die_change_minutes=D.DIE_CHANGE_MINUTES,
        max_steps=D.DEFAULT_MAX_STEPS,
        feeds=D.FEED_CATALOG,
        lines=lines,
        orders=orders,
        house_rules=house_rules,
    )

    plan = _plant(task, assignment, rng)
    if plan is None:
        return None

    task = dataclasses.replace(task, orders=_set_due_times(rng, cfg, orders, plan.completion))
    if any(o.due > task.horizon_min for o in task.orders):
        return None  # the day is full; redraw a lighter task

    if plan_violations(task, plan.actions):
        return None  # defensive: never hand out a task whose plan breaks a rule
    return GeneratedTask(task=task, plan=plan)


def _draw_lines(rng: random.Random, cfg: DifficultyConfig) -> tuple[D.LineSpec, ...]:
    lines = [
        D.LineSpec(
            id="L1",
            rate_t_per_h=D.DEFAULT_RATE_T_PER_H,
            pap_type=D.PAP_NONE,
            initial_die_mm=rng.choice(D.DIE_SIZES),
            initial_concentrations=D.zero_concentrations(),
        )
    ]
    if cfg.n_lines > 1:
        pap_type = rng.choice([D.PAP_PIG, D.PAP_POULTRY])
        lines.append(
            D.LineSpec(
                id="L2",
                rate_t_per_h=D.DEFAULT_RATE_T_PER_H,
                pap_type=pap_type,
                initial_die_mm=rng.choice(D.DIE_SIZES),
                initial_concentrations=D.zero_concentrations(),
            )
        )
    return tuple(lines)


def _feed_pool(cfg: DifficultyConfig, lines: Sequence[D.LineSpec]) -> tuple[D.Feed, ...]:
    """Feeds that can actually be produced on these lines.

    [DECISION] A PAP feed is only drawn when the task has a line of its type,
    so every drawn order is producible. On a hard task this means exactly one
    of the two PAP feeds is available, depending on the drawn line type.
    """
    pap_types = {ln.pap_type for ln in lines}
    pool = []
    for feed in D.FEED_CATALOG:
        substance = feed.pap_substance
        if substance is None:
            pool.append(feed)
        elif cfg.pap_feeds and D.PAP_SUBSTANCE_LINE[substance] in pap_types:
            pool.append(feed)
    return tuple(pool)


def _draw_orders(
    rng: random.Random, cfg: DifficultyConfig, lines: Sequence[D.LineSpec]
) -> tuple[D.Order, ...] | None:
    pool = _feed_pool(cfg, lines)
    n_orders = rng.randint(*cfg.n_orders)
    feeds = [rng.choice(pool) for _ in range(n_orders)]

    if cfg.pap_feeds:
        # [DECISION] a hard task always exercises the PAP rules.
        pap_feeds = [f for f in pool if f.pap_substance]
        if pap_feeds and not any(f.pap_substance for f in feeds):
            feeds[rng.randrange(n_orders)] = rng.choice(pap_feeds)

    # A task is only interesting when something can contaminate something else.
    if not any(f.substances for f in feeds):
        return None
    if not any(f.carry_over_class != D.CLASS_TARGET for f in feeds):
        return None

    n_customers = max(2, min(len(CUSTOMERS), n_orders // 2))
    customers = rng.sample(CUSTOMERS, n_customers)
    return tuple(
        D.Order(
            id=f"O{i + 1}",
            customer=rng.choice(customers),
            feed=feed.id,
            tonnes=float(rng.randint(*cfg.tonnes)),
            due=D.HORIZON_MIN,  # placeholder, replaced by _set_due_times
        )
        for i, feed in enumerate(feeds)
    )


def _line_allows(feed: D.Feed, line: D.LineSpec) -> bool:
    """L3, L4 and L5 as a line restriction (SPEC section 4)."""
    if feed.ruminant and line.pap_type != D.PAP_NONE:
        return False  # L3
    substance = feed.pap_substance
    if substance is not None and line.pap_type != D.PAP_SUBSTANCE_LINE[substance]:
        return False  # L4
    group = D.SPECIES_PAP_GROUP.get(feed.species)
    if group is not None and group == line.pap_type:
        return False  # L5
    return True


def _assign_lines(
    orders: Sequence[D.Order], lines: Sequence[D.LineSpec]
) -> dict[str, list[D.Order]] | None:
    """Longest-processing-time-first assignment to the least loaded legal line."""
    assignment: dict[str, list[D.Order]] = {ln.id: [] for ln in lines}
    load: dict[str, int] = {ln.id: 0 for ln in lines}
    ordered = sorted(
        orders,
        key=lambda o: (-D.production_minutes(o.tonnes, D.DEFAULT_RATE_T_PER_H), o.id),
    )
    for order in ordered:
        feed = D.FEED_BY_ID[order.feed]
        legal = [ln for ln in lines if _line_allows(feed, ln)]
        if not legal:
            return None
        best = min(legal, key=lambda ln: (load[ln.id], ln.id))
        assignment[best.id].append(order)
        load[best.id] += D.production_minutes(order.tonnes, best.rate_t_per_h)
    for line_id in assignment:
        assignment[line_id].sort(key=lambda o: int(o.id[1:]))
    return assignment


# --------------------------------------------------------------------------
# House rules (SPEC section 5)
# --------------------------------------------------------------------------


def _draw_house_rules(
    rng: random.Random,
    cfg: DifficultyConfig,
    orders: Sequence[D.Order],
    lines: Sequence[D.LineSpec],
    assignment: Mapping[str, Sequence[D.Order]],
) -> tuple[D.HouseRule, ...]:
    """Draw 0-4 house rules, preferring templates that actually bite.

    A template is "applicable" when the drawn orders can trigger it at all;
    a vacuous rule would make the law-only vs full-knowledge experiment of
    SPEC section 12 look weaker than it is.
    """
    n = rng.randint(*cfg.n_house_rules)
    if n == 0:
        return ()

    builders = {
        D.H1_CUSTOMER_FLUSH: _rule_customer_flush,
        D.H2_SEQUENCE_BAN: _rule_sequence_ban,
        D.H3_SLOW_DIE: _rule_slow_die,
        D.H4_DIE_CHANGE_DAY_ONLY: _rule_die_change_day_only,
        D.H5_LAB_HOLD: _rule_lab_hold,
        D.H6_COPPER_SHEEP_FLUSH: _rule_copper_sheep_flush,
    }
    applicable = [
        tid
        for tid in D.HOUSE_RULE_TEMPLATES
        if _template_applies(tid, orders, lines, assignment)
    ]
    chosen = list(rng.sample(applicable, min(n, len(applicable))))
    if len(chosen) < n:  # fall back to vacuous templates to reach the count
        rest = [t for t in D.HOUSE_RULE_TEMPLATES if t not in chosen]
        chosen += rng.sample(rest, n - len(chosen))

    rules = []
    for tid in sorted(chosen):
        rule = builders[tid](rng, orders, lines, assignment)
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


def _feeds_on_line(
    assignment: Mapping[str, Sequence[D.Order]], line_id: str
) -> list[D.Feed]:
    return [D.FEED_BY_ID[o.feed] for o in assignment[line_id]]


def _template_applies(
    template: str,
    orders: Sequence[D.Order],
    lines: Sequence[D.LineSpec],
    assignment: Mapping[str, Sequence[D.Order]],
) -> bool:
    feed_ids = {o.feed for o in orders}
    if template == D.H1_CUSTOMER_FLUSH:
        return bool(orders)
    if template == D.H2_SEQUENCE_BAN:
        return any(len({o.feed for o in assignment[ln.id]}) >= 2 for ln in lines)
    if template == D.H3_SLOW_DIE:
        return any(len({f.die_mm for f in _feeds_on_line(assignment, ln.id)}) >= 2 for ln in lines)
    if template == D.H4_DIE_CHANGE_DAY_ONLY:
        return any(len({f.die_mm for f in _feeds_on_line(assignment, ln.id)}) >= 2 for ln in lines)
    if template == D.H5_LAB_HOLD:
        return "pig_medicated" in feed_ids and any(
            D.FEED_BY_ID[o.feed].carry_over_class == D.CLASS_SENSITIVE for o in orders
        )
    if template == D.H6_COPPER_SHEEP_FLUSH:
        return "piglet_starter" in feed_ids and "sheep" in feed_ids
    return False


def _rule_customer_flush(rng, orders, lines, assignment) -> D.HouseRule | None:
    customer = rng.choice(sorted({o.customer for o in orders}))
    return D.HouseRule(
        id=D.H1_CUSTOMER_FLUSH,
        params={"customer": customer},
        text=(
            f"Every order of {customer} must be produced directly after a flush "
            f"of the line."
        ),
    )


def _rule_sequence_ban(rng, orders, lines, assignment) -> D.HouseRule | None:
    candidates: list[tuple[str, str]] = []
    for line in lines:
        feeds = sorted({o.feed for o in assignment[line.id]})
        candidates += [(a, b) for a in feeds for b in feeds if a != b]
    if not candidates:
        feeds = sorted({o.feed for o in orders})
        candidates = [(a, b) for a in feeds for b in feeds if a != b]
    if not candidates:
        return None
    feed_a, feed_b = rng.choice(sorted(set(candidates)))
    return D.HouseRule(
        id=D.H2_SEQUENCE_BAN,
        params={"feed_a": feed_a, "feed_b": feed_b},
        text=(
            f"{feed_b} must never be the next production after {feed_a} on the "
            f"same line, even with a flush in between."
        ),
    )


def _rule_slow_die(rng, orders, lines, assignment) -> D.HouseRule | None:
    with_changes = [
        ln for ln in lines if len({f.die_mm for f in _feeds_on_line(assignment, ln.id)}) >= 2
    ]
    line = rng.choice(with_changes or list(lines))
    return D.HouseRule(
        id=D.H3_SLOW_DIE,
        params={"line": line.id, "minutes": D.SLOW_DIE_CHANGE_MINUTES},
        text=(
            f"Line {line.id} has an old press: a die change there takes "
            f"{D.SLOW_DIE_CHANGE_MINUTES} minutes instead of {D.DIE_CHANGE_MINUTES}."
        ),
    )


def _rule_die_change_day_only(rng, orders, lines, assignment) -> D.HouseRule | None:
    return D.HouseRule(
        id=D.H4_DIE_CHANGE_DAY_ONLY,
        params={"latest_start": D.DIE_CHANGE_LATEST_START_MIN},
        text=(
            f"A die change may only be started between minute 0 and minute "
            f"{D.DIE_CHANGE_LATEST_START_MIN} (06:00 to 22:00)."
        ),
    )


def _rule_lab_hold(rng, orders, lines, assignment) -> D.HouseRule | None:
    return D.HouseRule(
        id=D.H5_LAB_HOLD,
        params={"minutes": D.LAB_HOLD_MINUTES},
        text=(
            f"After a batch containing an antimicrobial, the next production of "
            f"a sensitive feed on that line may only start {D.LAB_HOLD_MINUTES} "
            f"minutes after that batch ended (lab release)."
        ),
    )


def _rule_copper_sheep_flush(rng, orders, lines, assignment) -> D.HouseRule | None:
    return D.HouseRule(
        id=D.H6_COPPER_SHEEP_FLUSH,
        params={"after_feed": "piglet_starter", "before_feed": "sheep"},
        text=(
            "After piglet_starter a flush is required before any sheep batch on "
            "that line."
        ),
    )


# --------------------------------------------------------------------------
# The planted solution
# --------------------------------------------------------------------------


def _dirtiness(feed: D.Feed) -> int:
    """How much trouble this feed leaves behind for the next batch."""
    if feed.contains(D.MONENSIN):
        return 4
    if feed.contains(D.ANTIMICROBIAL):
        return 3
    if feed.contains(D.COPPER_HIGH):
        return 2
    if feed.pap_substance:
        return 1
    return 0


def _plant(
    task: D.Task, assignment: Mapping[str, Sequence[D.Order]], rng: random.Random
) -> PlantedPlan | None:
    """Sequence every line cleanly, then walk the clock and emit the actions."""
    actions: list[Action] = []
    segments: list[Segment] = []
    completion: dict[str, int] = {}
    makespan = 0

    for line in task.lines:
        orders = list(assignment[line.id])
        sequence = _sequence_line(task, line, orders, rng)
        if sequence is None:
            return None
        scheduled = _schedule_line(task, line, sequence)
        if scheduled is None:
            return None
        line_actions, line_segments, line_completion, clock = scheduled
        actions += line_actions
        segments += line_segments
        completion.update(line_completion)
        makespan = max(makespan, clock)

    actions.append({"tool": D.TOOL_FINISH})
    return PlantedPlan(
        actions=tuple(actions),
        segments=tuple(segments),
        completion=completion,
        makespan=makespan,
    )


def _sequence_line(
    task: D.Task, line: D.LineSpec, orders: Sequence[D.Order], rng: random.Random
) -> list[D.Order] | None:
    """Group the orders by die and run the clean ones before the dirty ones.

    Depth-first with that heuristic as the candidate order, so that house rule
    H2 (sequence ban) can be respected by backtracking instead of by luck.
    """
    if not orders:
        return []
    feeds = task.feed_map
    ban = task.house_rule(D.H2_SEQUENCE_BAN)
    banned: tuple[str, str] | None = (
        (str(ban.params["feed_a"]), str(ban.params["feed_b"])) if ban else None
    )
    tiebreak = {o.id: rng.random() for o in orders}

    def key(current_die: int, order: D.Order) -> tuple[Any, ...]:
        feed = feeds[order.feed]
        return (
            0 if feed.die_mm == current_die else 1,
            feed.die_mm,
            _dirtiness(feed),
            tiebreak[order.id],
        )

    budget = [20_000]

    def dfs(remaining: list[D.Order], previous_feed: str | None, die: int) -> list[D.Order] | None:
        if not remaining:
            return []
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        for order in sorted(remaining, key=lambda o: key(die, o)):
            feed = feeds[order.feed]
            if banned and previous_feed == banned[0] and feed.id == banned[1]:
                continue
            rest = [o for o in remaining if o.id != order.id]
            tail = dfs(rest, feed.id, feed.die_mm)
            if tail is not None:
                return [order] + tail
        return None

    return dfs(list(orders), None, line.initial_die_mm)


def _schedule_line(
    task: D.Task, line: D.LineSpec, sequence: Sequence[D.Order]
) -> tuple[list[Action], list[Segment], dict[str, int], int] | None:
    """Turn a production sequence into timed actions, inserting what is needed.

    Order inside one batch block: ``change_die`` -> ``wait`` -> ``flush`` ->
    ``produce``. The flush is emitted last so that it is not only the previous
    batch but literally the previous action before the production, which
    satisfies house rule H1 under either reading of "directly preceded".
    """
    feeds = task.feed_map
    h1 = task.house_rule(D.H1_CUSTOMER_FLUSH)
    h4 = task.house_rule(D.H4_DIE_CHANGE_DAY_ONLY)
    h5 = task.house_rule(D.H5_LAB_HOLD)
    h6 = task.house_rule(D.H6_COPPER_SHEEP_FLUSH)

    actions: list[Action] = []
    segments: list[Segment] = []
    completion: dict[str, int] = {}

    clock = 0
    die = line.initial_die_mm
    conc = dict(line.initial_concentrations or D.zero_concentrations())
    last_antimicrobial_end: int | None = None
    copper_pending = False

    for order in sequence:
        feed = feeds[order.feed]

        if die != feed.die_mm:
            duration = D.die_change_minutes(task, line.id)
            if h4 and clock > int(h4.params["latest_start"]):
                return None  # this draw cannot be planted inside the day
            actions.append({"tool": D.TOOL_CHANGE_DIE, "line": line.id, "die_mm": feed.die_mm})
            segments.append(Segment(line.id, D.TOOL_CHANGE_DIE, clock, clock + duration))
            clock += duration
            die = feed.die_mm

        need_flush = _needs_flush(task, conc, feed)
        if h1 and order.customer == h1.params["customer"]:
            need_flush = True
        if h6 and feed.id == str(h6.params["before_feed"]) and copper_pending:
            need_flush = True

        flush_minutes = task.flush_minutes if need_flush else 0
        if (
            h5
            and feed.carry_over_class == D.CLASS_SENSITIVE
            and last_antimicrobial_end is not None
        ):
            release = last_antimicrobial_end + int(h5.params["minutes"])
            wait = release - (clock + flush_minutes)
            if wait > 0:
                actions.append({"tool": D.TOOL_WAIT, "line": line.id, "minutes": wait})
                segments.append(Segment(line.id, D.TOOL_WAIT, clock, clock + wait))
                clock += wait

        if need_flush:
            conc = D.next_concentrations(conc, (), task.carry_over_rate)
            actions.append({"tool": D.TOOL_FLUSH, "line": line.id})
            segments.append(
                Segment(
                    line.id,
                    D.TOOL_FLUSH,
                    clock,
                    clock + task.flush_minutes,
                    concentrations=dict(conc),
                )
            )
            clock += task.flush_minutes
            copper_pending = False

        duration = D.production_minutes(order.tonnes, line.rate_t_per_h)
        conc = D.next_concentrations(conc, feed.substances, task.carry_over_rate)
        actions.append({"tool": D.TOOL_PRODUCE, "line": line.id, "order": order.id})
        segments.append(
            Segment(
                line.id,
                D.TOOL_PRODUCE,
                clock,
                clock + duration,
                order=order.id,
                feed=feed.id,
                concentrations=dict(conc),
            )
        )
        clock += duration
        completion[order.id] = clock

        if feed.contains(D.ANTIMICROBIAL):
            last_antimicrobial_end = clock
        if h6 and feed.id == str(h6.params["after_feed"]):
            copper_pending = True

    return actions, segments, completion, clock


def _needs_flush(task: D.Task, conc: Mapping[str, float], feed: D.Feed) -> bool:
    """True when producing ``feed`` straight away would break L1 or L2."""
    after = D.next_concentrations(conc, feed.substances, task.carry_over_rate)
    return bool(_limit_violations(feed, after))


def _limit_violations(feed: D.Feed, conc: Mapping[str, float]) -> list[str]:
    """L1 and L2 for one batch (SPEC section 4)."""
    codes: list[str] = []
    if not feed.contains(D.MONENSIN):
        limit = D.COCCIDIOSTAT_LIMIT.get(feed.carry_over_class)
        if limit is not None and conc[D.MONENSIN] > limit + D.LIMIT_TOL:
            codes.append(D.V_L1_COCCIDIOSTAT)
    if not feed.contains(D.ANTIMICROBIAL):
        if conc[D.ANTIMICROBIAL] > D.ANTIMICROBIAL_LIMIT + D.LIMIT_TOL:
            codes.append(D.V_L2_ANTIMICROBIAL)
    return codes


def _set_due_times(
    rng: random.Random,
    cfg: DifficultyConfig,
    orders: Sequence[D.Order],
    completion: Mapping[str, int],
) -> tuple[D.Order, ...]:
    """due = planted completion * slack + jitter (SPEC section 11)."""
    return tuple(
        dataclasses.replace(
            order,
            due=int(round(completion[order.id] * cfg.slack)) + rng.randint(0, MAX_JITTER_MIN),
        )
        for order in orders
    )


# --------------------------------------------------------------------------
# Independent check of an action list
# --------------------------------------------------------------------------


def plan_violations(task: D.Task, actions: Sequence[Action]) -> list[str]:
    """Every rule of SPEC sections 4, 5 and 9 that this action list breaks.

    Re-derives clocks, dies and concentrations from the actions alone, so a
    planted plan that passes here was not simply declared correct by the code
    that built it. Returns violation codes in the order they were found,
    without duplicates.
    """
    feeds = task.feed_map
    lines = task.line_map
    orders = task.order_map
    h1 = task.house_rule(D.H1_CUSTOMER_FLUSH)
    h2 = task.house_rule(D.H2_SEQUENCE_BAN)
    h4 = task.house_rule(D.H4_DIE_CHANGE_DAY_ONLY)
    h5 = task.house_rule(D.H5_LAB_HOLD)
    h6 = task.house_rule(D.H6_COPPER_SHEEP_FLUSH)

    codes: list[str] = []

    def add(code: str) -> None:
        if code not in codes:
            codes.append(code)

    state = {
        line_id: {
            "clock": 0,
            "die": line.initial_die_mm,
            "conc": dict(line.initial_concentrations or D.zero_concentrations()),
            "last_batch": None,  # "produce" | "flush"
            "last_feed": None,
            "antimicrobial_end": None,
            "copper_pending": False,
        }
        for line_id, line in lines.items()
    }
    produced: list[str] = []
    finished = False

    for action in actions:
        if finished:
            add(D.V_INVALID_ACTION)
            break
        tool = action.get("tool")
        if tool == D.TOOL_FINISH:
            finished = True
            continue
        line_id = action.get("line")
        if tool not in D.TOOLS or line_id not in state:
            add(D.V_INVALID_ACTION)
            continue
        line = lines[line_id]
        st = state[line_id]

        if tool == D.TOOL_WAIT:
            minutes = action.get("minutes", 0)
            if not isinstance(minutes, int) or minutes <= 0:
                add(D.V_INVALID_ACTION)
                continue
            st["clock"] += minutes

        elif tool == D.TOOL_CHANGE_DIE:
            die_mm = action.get("die_mm")
            if die_mm not in D.DIE_SIZES:
                add(D.V_INVALID_ACTION)
                continue
            if h4 and st["clock"] > int(h4.params["latest_start"]):
                add(D.V_H4_DIE_CHANGE_TIME)
            st["clock"] += D.die_change_minutes(task, line_id)
            st["die"] = die_mm

        elif tool == D.TOOL_FLUSH:
            st["conc"] = D.next_concentrations(st["conc"], (), task.carry_over_rate)
            st["clock"] += task.flush_minutes
            st["last_batch"] = D.TOOL_FLUSH
            st["copper_pending"] = False

        elif tool == D.TOOL_PRODUCE:
            order_id = action.get("order")
            if order_id not in orders:
                add(D.V_INVALID_ACTION)
                continue
            if order_id in produced:
                add(D.V_DUPLICATE_ORDER)
                continue
            order = orders[order_id]
            feed = feeds[order.feed]
            produced.append(order_id)

            if st["die"] != feed.die_mm:
                add(D.V_L6_WRONG_DIE)
            if feed.ruminant and line.pap_type != D.PAP_NONE:
                add(D.V_L3_RUMINANT_BAN)
            substance = feed.pap_substance
            if substance is not None and line.pap_type != D.PAP_SUBSTANCE_LINE[substance]:
                add(D.V_L4_PAP_LINE)
            group = D.SPECIES_PAP_GROUP.get(feed.species)
            if group is not None and group == line.pap_type:
                add(D.V_L5_INTRA_SPECIES)

            st["conc"] = D.next_concentrations(st["conc"], feed.substances, task.carry_over_rate)
            for code in _limit_violations(feed, st["conc"]):
                add(code)

            if h1 and order.customer == h1.params["customer"] and st["last_batch"] != D.TOOL_FLUSH:
                add(D.V_H1_CUSTOMER_FLUSH)
            if h2 and st["last_feed"] == h2.params["feed_a"] and feed.id == h2.params["feed_b"]:
                add(D.V_H2_SEQUENCE_BAN)
            if (
                h5
                and feed.carry_over_class == D.CLASS_SENSITIVE
                and st["antimicrobial_end"] is not None
                and st["clock"] < st["antimicrobial_end"] + int(h5.params["minutes"])
            ):
                add(D.V_H5_LAB_HOLD)
            if h6 and feed.id == str(h6.params["before_feed"]) and st["copper_pending"]:
                add(D.V_H6_COPPER_SHEEP_FLUSH)

            st["clock"] += D.production_minutes(order.tonnes, line.rate_t_per_h)
            if st["clock"] > order.due:
                add(D.V_LATE)

            st["last_batch"] = D.TOOL_PRODUCE
            st["last_feed"] = feed.id
            if feed.contains(D.ANTIMICROBIAL):
                st["antimicrobial_end"] = st["clock"]
            if h6 and feed.id == str(h6.params["after_feed"]):
                st["copper_pending"] = True

    if set(produced) != set(orders):
        add(D.V_MISSING_ORDER)
    if not finished:
        add(D.V_NOT_FINISHED)
    return codes
