# Results

Baselines of SPEC section 12 on held-out seeds 10000-10099, 100 tasks per difficulty. Reproduce with `python run.py`.

## Success rate (verifier score 1)

| agent | easy | medium | hard | all variants |
|---|---|---|---|---|
| `edd_naive` | 48% | 16% | 0% | **21.3%** |
| `law_aware` | 84% | 43% | 13% | **46.7%** |
| `full_aware` | 100% | 100% | 72% | **90.7%** |

`edd_naive` earliest due date first, never flushes. `law_aware` adds L1-L5. `full_aware` adds the house rules. All three share one policy and differ only in what they are allowed to know.

## Value of company knowledge

| difficulty | gap (full_aware - law_aware) |
|---|---|
| easy | +16% |
| medium | +57% |
| hard | +59% |

Control (SPEC section 12): with the house rules removed the two agents take identical actions on every task, so the gap is exactly 0. See `tests/test_agents.py::test_the_control_holds_on_tasks_with_no_house_rules`.

## What law_aware gets wrong

| violation | count |
|---|---|
| `H1_CUSTOMER_FLUSH` | 113 |
| `H5_LAB_HOLD` | 34 |
| `H6_COPPER_SHEEP_FLUSH` | 27 |
| `LATE` | 23 |
| `H2_SEQUENCE_BAN` | 2 |

No `L1`-`L6` anywhere: the gap above is house rules and lateness, nothing else.

## The naive reward ranks the agents backwards

| agent | naive reward | verifier | naive (hard) | verifier (hard) |
|---|---:|---:|---:|---:|
| `edd_naive` | 95% | 21.3% | 84% | 0% |
| `law_aware` | 92% | 46.7% | 77% | 13% |
| `full_aware` | 91% | 90.7% | 73% | 72% |

Under the naive reward the three agents are within three points of each other and the *worst* one leads, because flushes and lab holds cost time and lateness is all that reward can see. Under the verifier they separate cleanly. Training on the naive reward does not merely tolerate the illegal policy, it selects for it.

## The naive reward against the verifier

| difficulty | naive reward = 1 | of those, illegal |
|---|---|---|
| easy | 100% | 52% |
| medium | 100% | 84% |
| hard | 84% | 100% |

The naive reward counts lateness only, so it pays for `edd_naive`'s flush-free schedules. See `docs/exploit.md`.

## Charts

![A schedule the verifier accepts (hard-10002, full_aware)](gantt_valid.png)

*A schedule the verifier accepts (hard-10002, full_aware)*

![On time, and illegal: the naive reward pays 1 (hard-10002, edd_naive)](gantt_exploit.png)

*On time, and illegal: the naive reward pays 1 (hard-10002, edd_naive)*

