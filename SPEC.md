# Feed Mill Scheduling Environment — Specification

Single source of truth for domain and design.
**[DECISION]** = deliberate design choice. **[ASSUMPTION]** = plausible value, not measured.

---

## 1. The task in one sentence

A compound feed mill must schedule a day's feed orders on one or two production
lines so that every order is finished before its due time, no batch exceeds the
EU limits for carried-over substances, EU animal-protein bans are respected,
and all company-specific house rules are followed.

Why this task: sequence-dependent carry-over and die changes make good schedules
hard to find, but any finished schedule is cheap to check exactly.
"Hard to solve, easy to check" is the property verifiable RL needs.

---

## 2. Entities

### 2.1 Substances

Concentrations are expressed as a **fraction of the authorised level in the
target feed** (target feed = 1.0). This is how EU carry-over limits are defined.

| id | what | appears in |
|---|---|---|
| `monensin` | coccidiostat (ionophore) for broilers | `broiler_mon` |
| `antimicrobial` | antimicrobial in medicated feed (on veterinary prescription) | `pig_medicated` |
| `copper_high` | copper at piglet level (150 mg/kg) | `piglet_starter` |
| `pap_pig` | processed animal protein of porcine origin | `broiler_pigpap` |
| `pap_poultry` | processed animal protein of poultry origin | `pig_poultrypap` |

### 2.2 Carry-over sensitivity classes

| class | meaning | coccidiostat limit |
|---|---|---|
| `sensitive` | sensitive non-target species, withdrawal feed before slaughter, continuous food-producing animals (laying hens, dairy cows), and feed for the target species without added coccidiostat | 1 % |
| `less_sensitive` | other non-target species | 3 % |

### 2.3 Feed catalog

| feed id | species | ruminant | contains | class | die (mm) |
|---|---|---|---|---|---|
| `broiler_mon` | broiler | no | monensin | target | 3 |
| `broiler_withdrawal` | broiler | no | – | sensitive | 3 |
| `broiler_pigpap` | broiler | no | pap_pig | sensitive | 3 |
| `layer` | laying hen | no | – | sensitive | 3 |
| `piglet_starter` | pig | no | copper_high | less_sensitive | 3 |
| `pig_grower` | pig | no | – | less_sensitive | 4 |
| `pig_medicated` | pig | no | antimicrobial | less_sensitive | 4 |
| `pig_poultrypap` | pig | no | pap_poultry | less_sensitive | 4 |
| `dairy` | cattle | yes | – | sensitive | 6 |
| `sheep` | sheep | yes | – | sensitive | 4 |
| `horse` | horse | no | – | sensitive | 6 |

Classes taken from Annex I of Directive 2002/32/EC as amended by Regulation
(EU) No 574/2011, which places equine species, small ruminants (sheep and
goat), bovine, dairy cattle, laying birds and withdrawal feed in the strict
1.25 mg/kg tier for monensin sodium, and all other species in the 3.75 mg/kg
tier. `horse`, `sheep`, `dairy`, `layer` and `broiler_withdrawal` are therefore
`sensitive`; the pig feeds are `less_sensitive`.

### 2.4 Lines

`id`, `rate_t_per_h` (default 20 **[ASSUMPTION]**), `pap_type` ∈ {`none`, `pig`, `poultry`},
current die, concentration vector of the last batch, own clock in minutes.

### 2.5 Orders

`id`, `customer`, `feed`, `tonnes`, `due` (minute).
Production time in minutes = ceil(tonnes / rate_t_per_h × 60).

### 2.6 Time **[DECISION]**

Minute 0 = 06:00, horizon 1440 (one day). Each line has its own clock; actions on
a line start at that clock and advance it.

---

## 3. Carry-over model **[DECISION]**

Every batch (production or flush) leaves a residue in the line. The next batch
picks up a fixed share `r` of the previous batch's concentrations.

    concentration of substance s in batch B
        = (1.0 if B contains s, else 0)  +  r × concentration of s in the previous batch

- `r = 0.02` (2 % carry-over without cleaning) **[ASSUMPTION]**
- A **flush** is a small batch (1 t, 15 min) of plain grain run through the line
  and discarded. Its concentrations are `r × previous`, so the batch after a
  flush picks up only `r × r` (0.04 %).

Examples with r = 2 %:

- `broiler_mon` → `layer`: monensin in layer = 2 % > 1 % limit → **violation**.
- `broiler_mon` → flush → `layer`: 0.04 % → ok.
- `broiler_mon` → `pig_grower` → `layer`: pig_grower gets 2 % (≤ 3 %, ok),
  layer gets 0.04 % (ok). A less-sensitive batch acts as a natural flush.
  Real mills use this "sequencing as flushing" trick, and a good agent should
  discover it.

---

## 4. Legal rules (public, EU law)

| id | rule in the environment | legal basis | source |
|---|---|---|---|
| **L1** | Coccidiostat carry-over into non-target feed: monensin concentration ≤ 1 % for `sensitive`, ≤ 3 % for `less_sensitive` feeds. | Directive 2009/8/EC amending Annex I of Directive 2002/32/EC (maximum levels of unavoidable carry-over of coccidiostats in non-target feed). The 1 % / 3 % rationale is set out in the recitals of Regulation (EU) No 574/2011. Food-side limits: Regulation (EC) No 124/2009. | https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%3A32009L0008 · https://www.legislation.gov.uk/eur/2011/574/data.html |
| **L2** | Antimicrobial carry-over into any non-target feed ≤ 1 %. **[DECISION]** simplification; the real law sets substance-specific maximum levels. | Regulation (EU) 2019/4 on medicated feed, Article 7 (avoid cross-contamination; delegated acts set maximum levels); Delegated Regulation (EU) 2024/1229 (specific maximum levels for 24 antimicrobial active substances in non-target feed). | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32019R0004 · https://eur-lex.europa.eu/eli/reg_del/2024/1229/oj |
| **L3** | **Ruminant feed ban:** ruminant feeds (`dairy`, `sheep`) may never run on a line with `pap_type ≠ none`, regardless of flushing. | Regulation (EC) No 999/2001 (TSE Regulation), Article 7(1) prohibits feeding ruminants protein derived from animals; Annex IV sets out the extended ban and derogations. | https://eur-lex.europa.eu/eli/reg/2021/1372/oj/eng (amending act, cites Art. 7) |
| **L4** | **Non-ruminant PAP derogation:** porcine PAP may be used in poultry feed and poultry PAP in pig feed, but only on lines dedicated to that PAP type: `broiler_pigpap` only on `pap_type = pig` lines, `pig_poultrypap` only on `pap_type = poultry` lines. | Commission Regulation (EU) 2021/1372 amending Annex IV of Regulation (EC) No 999/2001 (allows pig PAP in poultry feed, poultry PAP in pig feed, under dedicated-line and channelling conditions). | https://eur-lex.europa.eu/eli/reg/2021/1372/oj/eng |
| **L5** | **Intra-species recycling ban ("anti-cannibalism"):** no feed for species X may run on a line whose `pap_type` is X. Pig feeds never on a `pig` line; poultry feeds never on a `poultry` line. **[DECISION]** zero tolerance, so it is modelled as a line restriction, not a carry-over limit. | Regulation (EC) No 1069/2009 (Animal By-Products Regulation), Article 11(1)(a): prohibits feeding terrestrial animals of a species with PAP derived from bodies or parts of bodies of animals of the same species. | https://eur-lex.europa.eu/eli/reg/2009/1069/oj |
| **L6** | **Die size:** a feed can only be pelleted with its die. `change_die` takes 45 min **[ASSUMPTION]**. Wrong die = physically impossible = invalid action. | Physical constraint, not law. | – |

### Related law, deliberately not modelled in v1

| topic | legal basis | why not modelled |
|---|---|---|
| Copper maximum content per species (e.g. piglets up to 4 weeks after weaning 150 mg/kg, sheep 15 mg/kg) | Commission Implementing Regulation (EU) 2018/1039 | At 2 % carry-over, 150 mg/kg becomes 3 mg/kg, far below the sheep maximum. Copper is therefore handled as a stricter **house rule** (H6), which shows the difference between law and company practice. |
| Ban on feeding catering waste to farmed animals | Regulation (EC) No 1069/2009, Article 11(1)(b) | Concerns ingredients, not sequencing. Candidate for the document-checking next step. |
| Fishmeal in feed for non-ruminants, milk replacers for unweaned ruminants | Regulation (EC) No 999/2001, Annex IV, Chapter II and IV | Adds feeds without adding a new rule type. |
| Veterinary prescription required for medicated feed | Regulation (EU) 2019/4, Article 16 ("Prescription") | Document-level check, natural next step. |
| General feed hygiene, HACCP, traceability | Regulation (EC) No 183/2005 | Process requirement, not a schedule property. |
| Switzerland | Swiss feed ordinances (Futtermittel-Verordnung) | Rules differ in detail; EU law is used as reference. |

**Verification record.** Checked against the consolidated texts on
legislation.gov.uk; EUR-Lex blocks automated retrieval.

- **L1** — Regulation (EU) No 574/2011 amending Annex I of Directive
  2002/32/EC, section VII entry 6 (monensin sodium): 1.25 mg/kg for "equine
  species, dogs, small ruminants (sheep and goat), ducks, bovine, dairy cattle,
  laying birds" and for withdrawal feed, 3.75 mg/kg for "other animal species".
  The two tiers stand in exactly the 1:3 ratio used here and correspond to 1 %
  and 3 % of a 125 mg/kg authorisation in target feed.
- **L2** — Regulation (EU) 2019/4 Article 7 ("Cross-contamination"): 7(1)
  requires operators to avoid it; 7(3) directs the Commission to set "specific
  maximum levels of cross-contamination for active substances in non-target
  feed" for the antimicrobials in Annex II. Delegated Regulation (EU) 2024/1229
  did so for 24 substances, applying from 20 May 2025. The single 1 % limit
  here stays a **[DECISION]** simplification of those levels.
- **L3** — Regulation (EC) No 999/2001 Article 7 ("Prohibitions concerning
  animal feeding"), 7(1): "The feeding to ruminants of protein derived from
  animals shall be prohibited." 7(2) extends the prohibition to non-ruminants
  in accordance with Annex IV.
- **L5** — Regulation (EC) No 1069/2009 Article 11 ("Restrictions on use"),
  11(1)(a) prohibits "the feeding of terrestrial animals of a given species
  other than fur animals with processed animal protein derived from the bodies
  or parts of bodies of animals of the same species".
- **Copper** (not modelled, see the table above) — Commission Implementing
  Regulation (EU) 2018/1039: 150 mg/kg for piglets up to 4 weeks after weaning,
  15 mg/kg for ovine. At 2 % carry-over that is 3 mg/kg, which is why copper is
  a house rule here and not a legal limit.

**[DECISION]** L4 is modelled as a one-directional constraint: a PAP feed may
only run on a line of its PAP type, while other feeds may run there subject to
L3 and L5. Annex IV of Regulation (EC) No 999/2001 is stricter. For the
aquaculture regime that Regulation (EU) 2021/1372 extends to poultry and pigs,
the compound feed "shall be produced in establishments ... dedicated exclusively
to the production of feed for aquaculture animals", with a derogation where
facilities are physically separated. Read into this model, a PAP line stands for
a plant dedicated to one target species. The two readings agree for every feed
in the catalog except `horse`, which this model permits on a PAP line and strict
dedication would not. The post-2021 text of Annex IV could not be retrieved, so
the simplification is recorded here rather than tightened on an inference.

---

## 5. House rules (private company knowledge)

Each task has a company profile with 0–4 house rules from these templates.
House rules only ever tighten the law. Each has structured fields and a
one-sentence natural-language `text`.

| id | template | meaning |
|---|---|---|
| H1 | `customer_flush(customer)` | Every order of this customer must be directly preceded by a flush on its line. Stricter customer contract. |
| H2 | `sequence_ban(feed_a, feed_b)` | feed_b may never be the next production after feed_a on the same line, even with a flush between. Introduced after a past incident. |
| H3 | `slow_die(line)` | Die changes on this line take 90 min instead of 45 (old press). |
| H4 | `die_change_day_only` | Die changes may only start between minute 0 and 960 (06:00–22:00). |
| H5 | `lab_hold(minutes=120)` | After a batch containing `antimicrobial`, the next production of a `sensitive` feed on that line may start only 120 min after that batch ended (lab release). |
| H6 | `copper_sheep_flush` | After `piglet_starter`, a flush is required before any `sheep` batch on that line. Stricter than law (see section 4). |

---

## 6. Actions (JSON tool calls)

    {"tool": "produce",    "line": "L1", "order": "O3"}
    {"tool": "flush",      "line": "L1"}
    {"tool": "change_die", "line": "L1", "die_mm": 4}
    {"tool": "wait",       "line": "L1", "minutes": 30}
    {"tool": "finish"}

**Invalid actions** (rejected, logged with status `invalid`, state unchanged):
unknown tool, unknown line or order, order already produced, wrong die,
non-positive wait, any action after `finish`, malformed input.

**[DECISION]** The environment enforces only physical possibility. Legal and
house rule violations can be committed and are caught by the verifier, so an
agent can make real mistakes, which is what training needs.

---

## 7. Episode

- `reset(seed, difficulty)` → first observation.
- `step(action)` → `(observation, reward, terminated, truncated, info)`
  (Gymnasium-style signature, no gymnasium dependency).
- Ends on `finish`; truncates after `max_steps` (default 200).
- Step reward 0; terminal reward = verifier score.

**Observation:** per-line clock, die, pap_type, concentrations of the last batch;
open and completed orders; feed catalog; carry-over rate `r`; legal rules
(structured + text); house rules (structured + text); step count.

**[DECISION]** v1 gives all rules directly in the observation. Extracting them
from documents or inferring implicit house rules from historical schedules is
the documented next step.

---

## 8. State and tamper-evident log

Final state contains:

- the full task and `task_hash` = SHA-256 of its canonical JSON,
- the action log `[{i, action, status, prev_hash, hash}]` with
  `hash = HMAC-SHA256(secret_key, prev_hash + canonical_json(entry))`,
  first `prev_hash = task_hash`,
- derived convenience fields (completion times, batch concentrations).

The secret key is created per environment instance and passed to the verifier
out of band; it never appears in an observation.
**The verifier never trusts derived fields**; it recomputes by replay.

Known limit: an agent running in the same Python process could read the key by
introspection. In a real deployment the environment runs in a separate process
or container. Document this in the README.

---

## 9. Verifier

`verify(final_state, key) -> VerificationResult(score: 0|1, violations: list)`

Reads only the final state. Independent of `env.py` (shares only constants and
data classes from `domain.py`).

Score 1 only if all hold:

1. Log chain intact (hashes and HMACs valid, starts at `task_hash`).
2. No `invalid` entry; episode ended with `finish`.
3. Every order produced exactly once, in full.
4. Every order ends at or before its due time.
5. L1–L6 hold for every batch.
6. All house rules of the task hold.

Each failure adds a violation code (`L1_COCCIDIOSTAT`, `L3_RUMINANT_BAN`,
`L5_INTRA_SPECIES`, `H2_SEQUENCE_BAN`, `LATE`, `TAMPERED_LOG`, …).

---

## 10. Rewards: naive vs. fixed (the documented exploit)

**Naive reward:** 1 if every order in the derived fields ends at or before its due time.

**Exploit A (main):** never flush and ignore sequencing. All orders are on time,
naive reward = 1, but layer feed contains 2 % monensin (limit 1 %), and horse
feed contains monensin, which is toxic to horses.

**Exploit B:** write completion times directly into the derived fields without
a valid log. Naive reward = 1.

**Fix:** the verifier in section 9. Tests prove both exploits score 0.

---

## 11. Task generator and difficulty

`generate(seed, difficulty)` is deterministic.

**Planted solution [DECISION]:** draw orders and house rules first, build one
valid schedule with a constructive heuristic that obeys every rule, then set
due times from it:

    due = planted completion time × slack + jitter,   with jitter ≥ 0

This guarantees solvability. The planted plan is never in the observation.

| | easy | medium | hard |
|---|---|---|---|
| lines | 1 (`pap_type none`) | 1 (`none`) | 2 (`none` + one PAP line, type drawn per seed) |
| orders | 5–6 | 8–10 | 12–16 |
| tonnes per order | 10–30 | 10–40 | 10–40 |
| slack | 2.0 | 1.4 | 1.15 |
| house rules | 0–1 | 2 | 3–4 |
| feeds | no PAP feeds | no PAP feeds | all |

Seeds: development 0–999, held-out evaluation 10000–10999.

---

## 12. Baselines and experiment

- `edd_naive`: earliest due date first, changes dies as needed, never flushes.
- `law_aware`: earliest due date first, flushes whenever the next batch would
  break L1/L2, assigns lines respecting L3–L5, ignores house rules.
- `full_aware`: `law_aware` plus all house rules.

Report success rate per difficulty and over all variants (100 held-out seeds per difficulty).

**Experiment: value of company knowledge**

    gap = success rate of full_aware − success rate of law_aware

per difficulty, plus a histogram of `law_aware` violation codes.

**Control (how the test could fail):** on tasks with zero house rules the gap
must be about 0. If it is not, the gap measures something other than house
rules and the result is invalid.

Honest limit: house rules are generated, so this shows the mechanism, not the
value of real house rules.

---

## 13. Real data interface

Tasks, feed catalog and house rules are JSON files with a documented schema
(`schemas/task.schema.json`). `run.py --task-file path.json` evaluates all
baselines on an external task. A real mill's orders, catalog, carry-over rate
and house rules could be converted into this format without code changes.

---

## 14. Requirements trace (for README)

| case requirement | where |
|---|---|
| reset(), step(action), observation, reward | `env.py`, section 7 |
| separate verifier reading only the final state | `verifier.py`, sections 8–9 |
| ≥ 3 task variants generated by code | `generator.py`, section 11 |
| baseline with success rate over all variants | `agents.py`, section 12 |
| verifier tests: initial 0, correct 1, invalid 0, written goal state 0 | `tests/test_verifier_required.py` |
| one documented exploit, fix and proving test | `docs/exploit.md`, `tests/test_exploit.py`, section 10 |
| runs with one command | `python run.py` |
| note of at most one page | `NOTE.md` |

---

## 15. Assumptions and their status

Plant values, assumed and still unmeasured:

- Carry-over rate 2 %, flush 1 t / 15 min, die change 45 min, line rate 20 t/h.
- One carry-over rate for all substances; real rates depend on substance and plant.

Deliberate simplifications, recorded in section 4:

- L2 uses a single 1 % limit instead of the substance-specific levels of
  Delegated Regulation (EU) 2024/1229.
- L4 is modelled as a one-directional line constraint.

Checked against the legal text (see the verification record in section 4):

- The species classes of section 2.3.
- The legal basis of L1, L2, L3 and L5, and the copper figures behind H6.

Cited but not checked: the post-2021 text of Annex IV of Regulation (EC)
No 999/2001, and the rows of the "related law" table other than Regulation
(EU) 2019/4.
