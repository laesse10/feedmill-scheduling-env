# Environment card: feed mill scheduling

A datasheet for deciding whether this environment is useful to you. Numbers
are measured on held-out seeds 10000-10099 and reproduced by `python run.py`.

## At a glance

| | |
|---|---|
| **Domain** | Compound feed mill production scheduling, one day |
| **Task type** | Single-agent, deterministic, discrete actions, terminal binary reward |
| **Variants** | 3 difficulties x unlimited seeds, all solvable by construction |
| **Episode length** | 6 to 27 actions for a competent policy; budget 200 |
| **Verification** | Independent replay of a tamper-evident log, score 1 or 0 |
| **Dependencies** | Python 3.11+, `matplotlib`, `pytest`. No network, no API keys |
| **Speed** | 0.2 ms to generate a hard task; 2.2 ms to generate, play and verify one |
| **Determinism** | Same seed, same task hash, in any process; identical report bytes |
| **Licence of the domain** | EU feed law, public; see the legal basis table in the README |

## The task

One or two production lines must produce every order of the day before its due
time. Each line has its own clock, its own die and its own residue. Every batch
picks up 2 % of whatever the previous batch left behind, so the *order* in
which feeds are produced decides whether a batch is legal.

A schedule is correct only if all of this holds at once: every order produced
exactly once and in full, every order on time, the carry-over limits respected
for every batch, the animal-protein bans respected on every line, and every
house rule of that mill followed.

Finding a good order is combinatorial. Checking a finished one is a replay.

## Action space

| action | effect | cost |
|---|---|---|
| `{"tool":"produce","line":L,"order":O}` | runs the order; residue updates | ceil(t / rate x 60) min |
| `{"tool":"flush","line":L}` | divides the residue by 50 | 15 min |
| `{"tool":"change_die","line":L,"die_mm":D}` | fits another die | 45 min, 90 on an old press |
| `{"tool":"wait","line":L,"minutes":M}` | idles the line | M min |
| `{"tool":"finish"}` | ends the episode | — |

Rejected as physically impossible, logged and ignored: unknown tool, line or
order, an order already produced, the wrong die, a non-positive or fractional
wait, extra keys, anything after `finish` or after truncation.

**Not rejected:** any schedule that is merely illegal. That is deliberate. The
environment models the press, the verifier models the law.

## Observation

Per line: clock, die, PAP dedication, die change duration, the residue of every
tracked substance, and what the last batch was. Globally: open orders with
customer, feed, tonnage and due time; completed orders with their times; the
feed catalog; the carry-over rate; the six legal rules and the mill's house
rules, each as structured parameters **and** one sentence of English; the step
count; and the result of the last action.

The baselines read the limits and the line rules out of the observation rather
than importing them, which is how the observation is kept sufficient to act on.
`llm_adapter.py` renders it as ~4 KB of text and exposes the actions as tool
schemas.

## Reward

Step reward 0. Terminal reward is the verifier's score, 1 or 0, with a list of
violation codes attached for diagnosis: `L1_COCCIDIOSTAT`, `L2_ANTIMICROBIAL`,
`L3_RUMINANT_BAN`, `L4_PAP_LINE`, `L5_INTRA_SPECIES`, `L6_WRONG_DIE`,
`H1_CUSTOMER_FLUSH`, `H2_SEQUENCE_BAN`, `H4_DIE_CHANGE_TIME`, `H5_LAB_HOLD`,
`H6_COPPER_SHEEP_FLUSH`, `LATE`, `MISSING_ORDER`, `DUPLICATE_ORDER`,
`INVALID_ACTION`, `NOT_FINISHED`, `TAMPERED_LOG`.

There is no partial credit. A schedule one minute late scores what a schedule
that contaminates layer feed scores.

## Difficulty distribution

| | easy | medium | hard |
|---|---|---|---|
| lines | 1 (`pap_type none`) | 1 (`none`) | 2 (`none` + one PAP line) |
| orders | 5-6 | 8-10 | 12-16 |
| tonnes per task (mean) | 110 t | 221 t | 348 t |
| due-time slack | x 2.0 | x 1.4 | x 1.15 |
| house rules | 0 or 1 (46 % / 54 %) | exactly 2 | 3 or 4 (45 % / 55 %) |
| feeds | no PAP feeds | no PAP feeds | all, incl. the PAP feed matching the line |
| planted plan | 8.9 actions, 426 min | 13.5 actions, 804 min | 21.1 actions, 790 min |
| flushes / die changes in it | 0.7 / 1.7 | 1.6 / 1.9 | 2.5 / 3.3 |

Every task is **solvable by construction**: the generator builds a valid
schedule first and derives the due times from it. The planted schedule is never
in the observation. Development seeds are 0-999, evaluation seeds 10000-10999.

## Baseline success rates

Held-out seeds 10000-10099, 100 tasks per difficulty.

| agent | knows | easy | medium | hard | all |
|---|---|---:|---:|---:|---:|
| `edd_naive` | nothing | 58 % | 22 % | 0 % | 26.7 % |
| `law_aware` | EU law L1-L5 | 83 % | 39 % | 14 % | 45.3 % |
| `full_aware` | law + house rules | 100 % | 100 % | 74 % | 91.3 % |

The three are one policy behind three knowledge flags, so the differences are
knowledge and not engineering. `law_aware` commits no legal violation on any of
the 300 tasks. `full_aware`'s remaining failures are 25 late tasks and one
stranded sequence ban, all caused by earliest-due-date having no lookahead, not
by the tasks being unsolvable.

Headroom for a learned policy: the whole gap between 91.3 % and 100 %, plus
everything below it.

## Reward hacking

An agent rewarded only for punctuality converges on "never flush". Measured on
`edd_naive` over the held-out set, that policy earns the naive reward on 100 %
of easy, 100 % of medium and 84 % of hard tasks, and **42 % / 78 % / 100 %** of
those schedules are illegal. On hard tasks the most common failures are not
carry-over but the line rules: ruminant feed on a PAP line and feed for a
species on that species' own PAP line.

The verifier defeats it by replaying the log rather than reading the summary.
`docs/exploit.md` has the worked example; `tests/redteam/` has one file per way
of cheating, thirteen in all.

## Intended use

Good for: benchmarking verifiable-reward methods where the checker must be
independent of the environment; studying reward hacking with a concrete,
legally-grounded failure mode; testing whether an agent can use rules given to
it in natural language; LLM tool-use evaluation on a short horizon.

Not good for: real production planning; throughput, cost or energy
optimisation; anything needing partial credit or a continuous objective;
compliance advice.

## Known limits

1. **In-process key.** The HMAC key is created per environment instance and
   never appears in an observation, log or state, but an agent in the same
   Python process can reach it by introspection. Run the environment in its own
   process or container for anything adversarial.
2. **Due times leak the planted solution** to within the jitter, under 30
   minutes, because they are derived from it.
3. **House rules are generated**, so the knowledge experiment demonstrates a
   mechanism rather than the value of a real mill's private rules.
4. **Modelling simplifications**: one carry-over rate for every substance and
   line; a single 1 % limit for all antimicrobials instead of substance-specific
   levels; copper handled as a house rule rather than a legal maximum; no
   partial deliveries; no changeover cost beyond time. `SPEC.md` section 15
   lists every assumed number.
5. **Legal citations are unverified** from inside this repository. `SPEC.md`
   marks them `[VERIFY]`; they should be checked against EUR-Lex before anyone
   relies on them.
6. **Pass/fail only**, so the reward carries no information about *how close* a
   failed schedule was.

## Provenance

Domain and design are specified in `SPEC.md`, which marks every deliberate
design choice `[DECISION]` and every plausible-but-unmeasured number
`[ASSUMPTION]`. The rules L1-L5 come from EU feed law; L6 is physics; H1-H6 are
invented company rules of the kind a mill actually keeps.
