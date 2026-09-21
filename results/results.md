# Results

Baselines of SPEC section 12 on held-out seeds 10000-10099, 100 tasks per difficulty. Reproduce with `python run.py`.

## Success rate (verifier score 1)

| agent | easy | medium | hard | all variants |
|---|---|---|---|---|
| `edd_naive` | 58% | 22% | 0% | **26.7%** |
| `law_aware` | 83% | 39% | 14% | **45.3%** |
| `full_aware` | 100% | 100% | 74% | **91.3%** |

`edd_naive` earliest due date first, never flushes. `law_aware` adds L1-L5. `full_aware` adds the house rules. All three share one policy and differ only in what they are allowed to know.

## Value of company knowledge

| difficulty | gap (full_aware - law_aware) |
|---|---|
| easy | +17% |
| medium | +61% |
| hard | +60% |

Control (SPEC section 12): with the house rules removed the two agents take identical actions on every task, so the gap is exactly 0. See `tests/test_agents.py::test_the_control_holds_on_tasks_with_no_house_rules`.

## What law_aware gets wrong

| violation | count |
|---|---|
| `H1_CUSTOMER_FLUSH` | 115 |
| `H6_COPPER_SHEEP_FLUSH` | 37 |
| `H5_LAB_HOLD` | 31 |
| `LATE` | 23 |
| `H2_SEQUENCE_BAN` | 2 |

No `L1`-`L6` anywhere: the gap above is house rules and lateness, nothing else.

## The naive reward against the verifier

| difficulty | naive reward = 1 | and illegal |
|---|---|---|
| easy | 100% | 42% |
| medium | 100% | 78% |
| hard | 84% | 84% |

The naive reward counts lateness only, so it pays for `edd_naive`'s flush-free schedules. See `docs/exploit.md`.

## Charts

![A schedule the verifier accepts (hard-10000, full_aware)](gantt_valid.png)

*A schedule the verifier accepts (hard-10000, full_aware)*

![On time, and illegal: the naive reward pays 1 (hard-10000, edd_naive)](gantt_exploit.png)

*On time, and illegal: the naive reward pays 1 (hard-10000, edd_naive)*

