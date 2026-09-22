# CLAUDE.md

## What we are building

A small, locally runnable RL environment for scheduling a compound feed mill,
with a separate verifier, a task generator, baseline agents, a documented
reward exploit and its fix. It is a take-home case study, so clarity,
correctness and honest reporting matter more than size.

**SPEC.md is the single source of truth.** Never deviate from it silently.
If something in SPEC.md is ambiguous or seems wrong, stop and ask instead of
guessing. Do not change SPEC.md yourself.

## Architecture invariants (never break these)

- `verifier.py` reads only the final state. It must not import from `env.py`.
  It may import constants and data classes from `domain.py` only.
- The verifier recomputes everything by replaying the log. It never trusts
  derived fields in the state.
- Everything is deterministic: same seed → same task → same `task_hash`.
- The HMAC secret key never appears in any observation, log entry or file
  the agent can see.
- The environment enforces only physical possibility. Legal and house rule
  violations must be possible to commit and must be caught by the verifier.
- Actions are JSON tool calls exactly as defined in SPEC.md section 6.

## Repository layout

```
feedmill-env/
├── README.md
├── SPEC.md                 # given, do not edit
├── ENV_CARD.md             # datasheet for a buyer
├── run.py                  # one command: evaluate and print results
├── feedmill/
│   ├── domain.py           # substances, feed catalog, rule constants, data classes
│   ├── generator.py        # generate(seed, difficulty) with planted solution
│   ├── env.py              # FeedMillEnv: reset, step, HMAC log
│   ├── verifier.py         # verify(final_state, key)
│   ├── naive_reward.py     # the naive reward, kept for the exploit demo
│   ├── agents.py           # edd_naive, law_aware, full_aware
│   ├── llm_adapter.py      # tool schemas + observation-to-text, no API calls
│   └── gantt.py            # matplotlib Gantt charts
├── schemas/task.schema.json
├── examples/example_task.json
├── tests/
│   ├── test_verifier_required.py   # the four required tests
│   ├── test_generator.py
│   ├── test_env_verifier_agreement.py
│   ├── test_exploit.py
│   └── redteam/                     # one file per cheating strategy
├── docs/exploit.md
└── results/                # written by run.py
```

## Milestones

Work through these in order. After each milestone: run `pytest`, commit,
then stop and give me a short summary (what was built, what was decided,
open questions). Wait for my go before starting the next milestone.

**Core** (covers every requirement of the case): milestones 1–4, the baselines
in milestone 5, `python run.py` with the results table from milestone 7, and the
README. Everything else is **stretch**. If time runs short,
a polished core beats a half-finished stretch.

1. **Domain and generator.** `domain.py` (feed catalog, substances, carry-over
   model and limits from SPEC.md sections 2–4), `generator.py`, planted solution,
   determinism tests, a test that every planted solution obeys all rules.
   Add unit tests reproducing the three carry-over examples in SPEC.md section 3.
2. **Environment.** `env.py` with the HMAC-chained log and invalid-action handling.
3. **Verifier and required tests.** Write the tests first:
   initial state → 0, planted solution → 1 (100 seeds × 3 difficulties),
   invalid action → 0, directly written goal state → 0.
   Then implement `verifier.py`. Add a differential test: for random valid and
   invalid action sequences, the verifier's replay matches the env's state.
4. **Exploit.** `naive_reward.py`, exploits A and B from SPEC.md section 10,
   tests showing naive reward = 1 and verifier = 0, plus `docs/exploit.md`
   (naive reward, the wrong solution, the fix, the test).
5. **Baselines (core) and experiment (stretch).** `agents.py`, success rate per
   difficulty and over all variants on held-out seeds 10000–10999 (100 per
   difficulty). Then the law-only vs full-knowledge gap, the control on tasks
   with zero house rules, and the violation-code histogram for `law_aware`.
6. **Red team (stretch).** One test file per cheating strategy in `tests/redteam/`:
   skipped flush, flush too early (production in between), ruminant feed on a
   PAP line, PAP feed on the wrong line, intra-species violation, each house
   rule violated, late order, partial or duplicate production, reordered log
   entries, truncated log, wrong HMAC key, edited derived fields. All must score 0.
7. **run.py and outputs.** `python run.py` prints the results table and writes
   `results/results.md` plus two Gantt PNGs (one valid plan, one exploit plan).
   `python run.py --task-file examples/example_task.json` evaluates an external task.
   Add `schemas/task.schema.json` and `llm_adapter.py`.
8. **Docs.** `README.md` (what, one command, results table, the requirements
   trace from SPEC.md section 14, the legal basis table, how to plug in real
   data, the in-process key limit from SPEC.md section 8) and `ENV_CARD.md` (task, actions, difficulty distribution,
   baseline success rates, known limits).

## Working rules

- Tests first, then code.
- **Never weaken, delete, skip or xfail a test to make it pass.** If a test
  seems wrong, stop and tell me why.
- Never special-case seeds or test inputs in production code.
- If baseline numbers look surprising (e.g. `edd_naive` succeeds often on
  hard tasks), report it; do not tune the generator to make numbers look good
  without telling me.
- Small, focused commits with clear messages in English.
- **No `Co-Authored-By` lines, no mention of AI assistance in commits, code,
  comments or docs.**

## Code style

- Python 3.11+, type hints, `dataclasses`, small pure functions.
- Canonical JSON for hashing: `json.dumps(obj, sort_keys=True, separators=(",", ":"))`.
- Docstrings where the reason for a design choice is not obvious.
- All code, comments and docs in English.

## Dependencies

- Only the Python standard library, `pytest` and `matplotlib`.
- No network access at runtime, no API calls, no downloads, nothing paid.
- A `requirements.txt` with pinned versions.

## Definition of done

- Fresh clone → `pip install -r requirements.txt` → `python run.py` works
  without any other setup and prints the results table.
- `pytest` is fully green, including `tests/redteam/`.
- `git log` contains no co-author lines.
