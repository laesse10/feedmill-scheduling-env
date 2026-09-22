# Note

## What this is

A feed mill produces feed in batches on a line, and every batch leaves a residue
that contaminates the next. The **order** you produce in therefore decides
whether the feed is legal to sell: run broiler feed containing a coccidiostat
straight before laying-hen feed, and the hen feed carries 2 % of that drug
against an EU limit of 1 %. A 15-minute flush clears it.

An agent schedules a day of orders on one or two lines — each order before its
due time, each batch inside the EU carry-over limits and animal-protein bans, the
mill's house rules followed. Finding a good order is combinatorial; checking a
finished one is a replay. The rules are real EU law, so "correct" is defined
outside my own code. `python run.py` reproduces everything below.

## Decisions

- **The verifier is written twice.** It never imports the environment, and
  re-derives canonical JSON, the log signature, the contamination update and
  batch durations from the specification; shared arithmetic would hide a mistake
  from both at once.
- **It ignores what the environment reports.** The final state carries a summary
  of completion times; the verifier never reads it, replaying a signed log
  instead. An agent may rewrite that summary freely and nothing changes.
- **The environment permits illegal schedules**, refusing only what a press
  physically cannot do. An agent has to be able to make the real mistake.

## Results

Three baselines, all earliest-due-date, differing only in what they may consider:
`edd_naive` knows nothing, `law_aware` knows EU law, `full_aware` also knows the
house rules. 100 held-out tasks per difficulty:

| agent | easy | medium | hard | all |
|---|---:|---:|---:|---:|
| `edd_naive` | 48 % | 16 % | 0 % | 21.3 % |
| `law_aware` | 84 % | 43 % | 13 % | 46.7 % |
| `full_aware` | 100 % | 100 % | 72 % | 90.7 % |

One policy behind three flags, so the gap measures knowledge and not code
quality: **+16 / +57 / +59 points**. Control: remove the house rules and the last
two produce identical schedules, putting the gap at exactly 0.

**The obvious reward is worse than useless.** "Did every order finish on time?"
scores the three at 95 %, 92 % and 91 % — the *worst* agent wins, because a flush
costs 15 minutes and lateness is all that reward sees. Removing a flush can never
lower it, so training on it selects for the illegal policy. The verifier scores
the same three 21 %, 47 % and 91 %.

**Checking the law found an error in my own specification:** sheep feed sat at
3 % where Annex I of Directive 2002/32/EC puts small ruminants at 1 %. No test
could have caught it — every test agreed with the same wrong constant.

## Limits

- **The tasks are generated, not real.** The generator discards draws it finds
  uninteresting and guarantees solvability; a real order book is lumpier, and
  sometimes impossible.
- The signing key sits in the agent's process; isolation needs a separate one.
- Due times derive from the hidden solution, leaking it to within half an hour.
- House rules are invented: the experiment shows a mechanism, not their worth.
- Scoring is pass/fail — one minute late scores what poisoning a herd scores.

## Next steps

Give the agent the rules as documents instead of structured fields, and infer the
unwritten ones from past schedules. Run a real order book through the task format
without the solvability guarantee. Measure contamination per substance, and add a
graded reward beside the binary one.
