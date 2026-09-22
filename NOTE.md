# Note

**Feed mill scheduling:** a day of orders on one or two lines, under EU feed law
and a mill's own house rules. Hard to solve, cheap to check. `python run.py`
reproduces everything below.

## Decisions

**The task.** Sequence-dependent carry-over makes good schedules combinatorial; a
finished one is just a replay. The constraints are real EU law, so "correct" has
a definition outside my own code — which I could check, and did.

**The verifier is independent by construction.** It never imports the
environment; canonical JSON, the HMAC chain, the carry-over update and production
time are written a second time from the specification, so a mistake in one
arithmetic cannot hide in both.

**It never reads the derived fields.** Completion times are recomputed by
replaying an HMAC-chained log anchored at the task hash, which makes a forged
summary worthless rather than merely detectable.

**The environment enforces only physical possibility.** A wrong die is rejected;
a contaminating sequence is not — an agent has to be able to make real mistakes.
Tasks are solvable by construction: a valid schedule is planted first and the due
times derived from it.

## Results

Held-out seeds 10000–10099, 100 tasks per variant, scored by the verifier:

| agent | easy | medium | hard | all |
|---|---:|---:|---:|---:|
| `edd_naive` | 48 % | 16 % | 0 % | 21.3 % |
| `law_aware` | 84 % | 43 % | 13 % | 46.7 % |
| `full_aware` | 100 % | 100 % | 72 % | 90.7 % |

One policy behind three knowledge flags, so the difference measures knowledge and
not engineering: **+16 / +57 / +59 points**. `law_aware` breaks no legal rule on
any of the 300 tasks. Control: remove the house rules and the two agents emit
identical actions, so the gap is exactly 0.

The exploit **ranks the agents backwards** — `edd_naive` takes 95 % of the naive
reward against `full_aware`'s 91 %, and 21.3 % against 90.7 % on the verifier. A
flush costs 15 minutes and lateness is all that reward sees, so removing one can
never lower it: training on it selects for the illegal policy.

Of eleven bugs injected into the verifier, ten fail the 251 tests. Checking the
citations found an error in my own specification — sheep feed sat at 3 % where
Annex I of Directive 2002/32/EC puts small ruminants at 1 % — which the verifier
had propagated deterministically and no test could have caught.

## Limits

- **The instances are synthetic.** The generator rejects draws it finds
  uninteresting and guarantees solvability, which is exactly the irregularity a
  real order book would have supplied.
- The HMAC key lives in the agent's process; isolation is a deployment property.
- Due times derive from the planted schedule, so they leak it to within the jitter.
- House rules are generated: a mechanism, not the value of real ones.
- Pass/fail scoring says nothing about how close a failure came.

## Next steps

Rules read out of documents rather than handed over in the observation. A real
order book through the task schema with the solvability guarantee dropped — an
infeasible day is real, and "which order do you sacrifice" cannot be trained
here. Carry-over measured per substance and plant, a shaped reward beside the
binary one, and the environment in its own process.
