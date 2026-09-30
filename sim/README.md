# Draft-model backtest: evolutionary tournament on anonymized history

Answers "is draftmodel.py actually good?" by evolving weight-variants of it in
simulated historical leagues and grading the champion on held-out seasons it
never trained on.

## No-leakage guarantees

- **Point-in-time knowledge only.** For season Y a drafter sees: that year's
  preseason ADP (+ ADP stdev, bye week) and positional rank→points curves fit
  exclusively on seasons **< Y**. Season-Y weekly points live only inside the
  environment and are used solely to score completed seasons.
- **Anonymized players.** `fetch_history.py` replaces every name at ingestion
  with `POS_<sha1-hash>` (e.g. `RB_1a2b3c4d`) and drops team abbreviations.
  No script prints or stores a real name; policies see only
  `{id, pos, adp, adp_sd, bye, proj}`.
- **Train/holdout split.** Evolution touches 2010–2020 only. 2021–2025 are the
  final exam (`validate.py`), never seen during evolution or champion selection.
- **In-season knowledge is causal.** Weekly lineups use preseason expectation
  blended with points-per-game *from earlier weeks of that same season* only.

## Pipeline

```
python fetch_history.py    # download + anonymize 2010-2025 (once, ~2 min)
python evolve.py           # fixed run: 40 generations on 2010-2020 (~1 min)
python evolve.py --resume  # continue a previous run from its checkpoint
python evolve.py --loop --resume   # run continuously until the guard stalls
python evolve.py --loop --resume --patience 0   # run forever until Ctrl-C
python validate.py         # champion + hand-tuned default vs 9 ESPN-autopick
                           # bots on held-out 2021-2025
```

Continuous mode (`--loop`): carves 2019-2020 out of training as a GUARD slice,
checks the current best genomes on it every `--check-every` generations, and
auto-stops after `--patience` checks without a new guard best (i.e., when more
generations = memorizing the training years, not getting better). State is
checkpointed every generation; Ctrl-C is safe and `--resume` continues.

**Measured warning about continuous mode:** in practice the fixed run
outperformed it. The loop's guard correctly stopped training (best at gen 50
of 125), but champions evolved on 2010-2018 generalized badly to 2021+
(title% at or below the 10% luck line) regardless of how they were selected —
while the fixed run trained through 2020 hit 23.5%. Era drift dominates
weight-tuning: recent training years are worth more than extra generations.
Use `--loop` to explore; trust the fixed run for the deployable champion.
Also: every extra peek at the holdout weakens it as an exam - don't iterate
against validate.py.

## How it works

- **League** = 10 drafters, snake, 16 rounds, my league's lineup
  (QB/2RB/2WR/TE/FLEX/K/DST), PPR, no keepers. Draft slots shuffled per league.
- **Season** = 13-week round-robin regular season, semis week 14 (1v4, 2v3),
  final = weeks 15+16 cumulative. K/DST score flat constants — no draft skill
  there, but the roster slots and timing cost stay real.
- **Managed weeks (env v2, 2026-08-10)**: every week each team may make one
  waiver move (drop worst benched skill player for a clearly better free
  agent; worst record claims first) and one 1-for-1 cross-position trade
  (executes only if BOTH sides' typical-week lineups improve past their
  thresholds; bye-neutral valuation; deadline week 10). Lineups are
  matchup-aware: best expected lineup, tilted toward volatile players when
  projecting to lose the week and toward floor when projecting to win
  (`var_tilt`). Free-agent stats are public (observed weekly points tracked
  for all players, past weeks only).
- **Opponents (env v2)**: validation/guard bots are `NoisyAutopickPolicy` —
  ESPN autopick over a personally jittered board (each player's ADP perturbed
  by N(0, adp_sd), the measured variance in where real humans drafted them).
  Bots also manage rosters with default thresholds. `--clean-bots` restores
  the deterministic v1 benchmark.
- **Genome** = the weight vector of draftmodel.py (flex demand shares, bench
  discount β, baseline depth, lookahead weight, per-position value biases,
  reach penalty, K/DST timing) **+ 3 management genes** (`var_tilt`,
  `waiver_thr`, `trade_min`). `genome.py` lists ranges; defaults reproduce
  the hand-tuned model.
- **Evolution** (`evolve.py`): each generation the population is shuffled into
  leagues of 10; fitness = wins + 3×championship + points/1000. **Each league's
  playoff champion survives unchanged** (variance-preserving, per design); the
  rest of the next generation is tournament-selected crossover + Gaussian
  mutation. A hall of fame keeps the best-ever league results.
- **Champion selection**: hall-of-famers + recent champs are re-run against
  9 ESPN-autopick bots across all training years; best mean fitness →
  `results/champion.json`.
- **Validation** (`validate.py`): champion (and the hand-tuned default) vs
  9 autopick bots on the holdout years. Luck baselines: 10% titles, 40%
  playoffs, 6.5 wins. A real edge = title% above 10% by multiple SEs.

## Trade lab (open-ended trade learning — `trade_lab.py`, 2026-08-10)

The Monopoly-style experiment: every team drafts identically (deterministic
autopick, shuffled slots) so ONLY trading differs. After every week (playoffs
included) each team may propose one 1-for-1 trade of any player for any player
(K/DST excluded, rosters stay legal); the counterparty accepts or rejects.
Both decisions come from each team's own evolved value net (14 causal features
→ tanh(8) → value, 129 weights) — no hand-coded strategy or fairness rule;
trades happen only where two nets disagree about two players. League champions
survive unchanged, the rest are bred (crossover + mutation). Runs until
Ctrl-C; checkpoints every generation; `--resume` continues.

```
python trade_lab.py             # fresh run, forever (Ctrl-C safe)
python trade_lab.py --resume    # continue
```

**AlphaZero discipline (added same day):** population mean is pinned ~8.1 by
zero-sum arithmetic (65 wins + 1 title per league, always) — it can NEVER
show progress. Progress lives in the ratchet: an archived INCUMBENT net
(`results_tradelab/incumbent.json`, persists across runs, starts as the
zero-net that never trades). Every `--gate-every` (50) gens the best screened
candidate plays the incumbent head-to-head in the SAME leagues (+8 rational
owners, `--gate-leagues` 150) and dethrones it only if the mean paired
fitness margin beats `--gate-margin` (0.35). Each click is appended to the
incumbent's lineage — that list is the Elo curve. After each gate, `--inject`
(5) incumbent mutants re-anchor the population. `trade_champion.json` is
always the incumbent, so the deliverable can never regress.
`--reset-incumbent` wipes the archive back to the no-trade baseline.

Watch two numbers in `results_tradelab/history.jsonl` (both printed each
gate cycle): **ratchet clicks** (is the lineage growing?) and the incumbent's
**bench vs rational owners** (absolute skill vs the luck line: 10% titles,
6.5 wins). First click = trading beats not-trading at all. Throughput
≈ 8-9k generations (≈ 85k league-seasons) per day plus ~20% gate overhead.
Holdout 2021+ untouched. The eventual combined model = best draft genome +
this trade net, wired together via `run_season(trade_fn=...)`.

**v2: biased-owner ecosystem (2026-08-11).** Nets now hunt exploitable
counterparties: RECENCY owners (w=0.85 on observed form — sell-low/buy-high)
and ANCHOR owners (w=0.15 — preseason-name loyalists) judge trades by raw
player value under their distorted lens and propose their own (bad) offers;
RATIONAL owners (w=0.50, strict lineup rule) remain the unexploitable
control. Training leagues = 6 nets + 2 recency + 1 anchor + 1 rational.
**Environment certified before handover:** an oracle trader using true values
beats never-trading by +0.63 fitness (+2.7 SE) over 300 paired leagues,
making ~7 trades/season — real, learnable edge ≈ 1.8× the gate margin.
Two modeling lessons found by oracle-testing (trades were ~zero until both
were fixed): (1) proposers must NEGOTIATE DOWN — the max-own-gain offer is
always a star-grab that gets vetoed; acceptable trades rank deep in the offer
list; (2) biased owners must value trades by PLAYER value, not lineup
improvement — the lineup rule is the sophisticated behavior and recreates the
no-trade theorem even for biased lenses. Old incumbents are incompatible
(archived in `results_tradelab_v1_notrade_2026-08-10/`); v2 starts the
ratchet fresh from the zero net. Throughput ≈ 20 s/generation.

**v2 result (gen 11,504; graded 2026-08-12, verdict corrected 2026-08-13):**
the incumbent (10 ratchet clicks, last at gen 10,000) beats never-trading —
paired incumbent-vs-zero-net at n=5,000 leagues per split: **train
+0.306±0.054 (+5.6 SE), holdout 2021-25 +0.196±0.054 (+3.6 SE)** — real and
it generalizes, ≈⅓–½ of the certified oracle ceiling (+0.63), ~11.7 trades/
season, bench vs mixed owners 13.5%±1.1% titles / 6.80 wins (n=960). The
initial 2026-08-12 grading at n=300 (+0.21±0.22 train, +0.09±0.23 holdout)
had read as "no skill" — a +0.2 effect needs ~2,000+ paired leagues to
resolve (SE≈0.05 at 5k, ~0.13 s/league); don't grade trade nets at n=300.
Caveat: the edge comes from exploiting RECENCY/ANCHOR owners; vs purely
rational owners the v1 no-trade theorem still applies. For real-league
offers keep using `draftmodel.py --trades` with noisy paired sims.

**v1 result (2026-08-10, 3,738 gens / ~100k league-seasons, 4.5 h):** the
experiment converged to the NO-TRADE THEOREM. Incumbent vs never-trade over
400 paired leagues: margin −0.03 (SE 0.19) — zero accumulated skill. The 4
ratchet clicks were gate false-positives (noise expectation ≈ 9/75 at this
threshold). Self-play trade volume collapsed from ~60/league (random nets)
to ~5 within a few hundred generations: selection prunes exploitable
disagreement, so surviving nets mostly refuse. This is the correct optimum
for THIS environment — every agent shares the same public stats and the same
best estimator (exp_ppw), so any trade one side wants, the other should
refuse (Milgrom–Stokey). Emergent trading profit requires an exploitable
counterparty: owners with behavioral biases (overweight recent form →
sell-low/buy-high, preseason-name anchoring). Add biased-owner opponents
before spending more compute here; running longer just re-confirms the
theorem. Corollary for the REAL league: trade edge lives in opponents'
misvaluations, not in trade mechanics — hunt real offers with
`draftmodel.py --trades`.

## Retraining (v2 protocol — evolution is a lottery, buy several tickets)

```
python evolve.py --seed N                    # one ticket (~4 min); N=1 already spent
cp results/champion.json results_multiseed/champion_seedN.json
python results_multiseed/compare_train.py    # rank ALL champions on TRAIN years only
python validate.py --genome <winner.json>    # ONE holdout look, winner only —
                                             # and only if it beat the v1 champion
                                             # in the training-year ranking
```

Never grade every seed's champion on the holdout — that's multiple-selection
on the exam, the thing the hygiene rule exists to prevent.

## Files

| file | role |
|---|---|
| `config.py` | league constants + shared eligibility rules |
| `fetch_history.py` | download + anonymize seasons → `data/season_YYYY.json` |
| `env.py` | point-in-time view, curves, draft engine, season simulator |
| `genome.py` | gene ranges/defaults, mutation, crossover |
| `policy.py` | `GenomePolicy` (pair-value lookahead port), `AutopickPolicy` (ESPN bot) |
| `evolve.py` | generational tournament + champion selection |
| `validate.py` | holdout evaluation vs autopick bots |

## Results (2026-08-06 → 2026-08-10, all validated at 300 leagues/holdout year vs 9 autopick bots)

**⚠ v1-environment numbers.** Everything in this table was measured in the
draft-only environment against deterministic autopick bots. The v2 managed
environment (waivers/trades/variance lineups/noisy bots) is a structural
change — v2 numbers are NOT comparable to these and get their own table below.

| genome | trained on | holdout title% | verdict |
|---|---|---|---|
| fixed-run champion, seed-1 stream (2026-08-06) | 2010–2020 | **23.5%** | +13.5 pts, +17 SE — **LOST**: overwritten by a loop run 2026-08-09; a seed-1 rerun reproduces the early stream but a different champion, so the genome is not recoverable |
| fixed-run champion, seed-1 rerun (2026-08-10) | 2010–2020 | **21.4%** | +11.4 pts, +14.7 SE (seed-42 leagues) — decisive; now in `results/champion.json` |
| gen-325 loop champion | 2010–2018 | 14.0% / 14.6% | +5 pts, +5–6 SE on two independent seeds — real, era-lumpy; kept in `results_loop325_backup_2026-08-09/` |
| hand-tuned default | — | 12.9% / 12.9% | +2.9 pts, +3.8 SE — real, modest, remarkably seed-stable |
| luck baseline | — | 10.0% | |
| fixed-run champion, different rng stream | 2010–2020 | 10.2% | luck |
| loop champion, guard-selected | 2010–2018 | 10.1% | luck |
| gen-91,988 loop champion (91.7k gens, `--patience 0`) | 2010–2018 | 7.7% | below luck — marathon evolution actively harmful; population mean fitness was flat (~8.1) from gen 1, guard "bests" were max-over-noise artifacts; kept in `results_loop92k_backup_2026-08-10/` |
| loop champion, all-years-selected | 2010–2018 | 7.5% | below luck |

## Results — env v2 (2026-08-10, 300 leagues/holdout year, seed 42, vs 9 noisy managed bots)

| genome | title% | playoff% | avg wins | verdict |
|---|---|---|---|---|
| v1 fixed-run champion (default mgmt genes) | **16.2%** | 45.8% | **6.83** | +6.2 pts, +8.0 SE — best on every metric; restored to `results/champion.json` |
| hand-tuned default | 11.5% | 38.8% | 6.47 | +1.5 pts, +1.9 SE — edge compressed by opponent noise |
| v2-evolved champion (seed-1 fixed run in v2 env) | 10.9% | 40.3% | 6.58 | ≈ luck — a lottery-loser draw; its weird traits (reach_pen 0.30, waiver_thr maxed) read as noise-fit; artifacts in `results_v2_fixed_2026-08-10/` |
| luck baseline | 10.0% | 40.0% | 6.50 | |

**What v2 taught:** (1) ~¾ of the v1 champion's edge SURVIVES realism
(21.4% vs clean deterministic bots → 16.2% vs adp_sd-noisy, roster-managing
bots) — the edge is mostly real drafting skill, not bot exploitation.
(2) Opponent noise compresses everyone toward luck and makes fitness noisier,
so single-run evolution is an even worse lottery in v2 — the freshly evolved
v2 champion landed at luck. Re-running evolution is cheap; grading on holdout
is not. Evolve several seeds and select on TRAINING years before spending a
holdout look.
(3) **Multi-seed retrain in v2 failed to dethrone the v1 champion** (2026-08-10,
seeds 1–6, `results_multiseed/`): on training years vs noisy bots the v1
champion scored fitness 8.20 / 6.54 wins / 11.5% titles; the best v2 seed
managed 8.10 and none earned a holdout look. Lesson: TRAIN where the signal is
strong (clean deterministic env), GRADE where the realism is (noisy managed
env) — evolving inside the noise doesn't find better genomes, it just hides
the good ones.

**The two load-bearing lessons this table teaches:**

1. **Single-run evolution is a lottery.** Identical settings with a different
   random stream swing the holdout result from 23.5% to 10.2%. Evolved
   champions are high-variance; the 23.5% genome's own validation is clean
   (1,500 holdout leagues, selection never touched holdout), but the *process*
   does not reliably reproduce genomes that good. Treat `champion.json` as a
   found artifact, not a reproducible output — and don't overwrite it casually.
2. **Era drift dominates weight-tuning.** Loop-mode champions (trained without
   2019–2020) specialize in the wrong era (35% titles in 2022, ~0–7% in
   2023/2025). Recent training years are worth more than extra generations —
   the guard proves the generations point directly (best genome at gen 50 of
   125, then stall).

The 23.5% champion's evolved traits vs default: strong reach penalty 3.1
(ADP price discipline — independently rediscovering the strongest documented
edge in the research), deeper baselines (1.7), lower lookahead weight (0.35).
Its weights are tuned for THIS environment (ADP-proxy projections, bot
opponents); port to draftmodel.py judiciously — reach penalty and deeper
baselines transfer on principle, the low lookahead weight may not.

Holdout hygiene: eight gradings have now touched 2021–2025 (five genomes on
2026-08-06, plus the 92k loop champion, the seed-1 rerun champion, and seed-42
replications). Each additional peek weakens it as an exam. Stop validating new
champions against it unless something structural changes. Replicating an
ALREADY-graded genome on a fresh seed is fine (that's measurement, not
selection) — grading NEW genomes is what burns the exam.

## Interpreting results

- The number that matters: **champion title% on holdout years vs 10%**.
  With 150 leagues × 5 years, SE ≈ 1.1%, so ≥13–14% is a real edge.
- Compare the evolved champion to the hand-tuned default: if they're close,
  the model's structure (lookahead + dynamic VBD) carries the edge and the
  exact weights barely matter — which the research predicts.
- Caveats: opponents are pure autopick bots (real league-mates are noisier);
  projections are ADP-curve proxies (weaker than real preseason projections);
  waivers/trades don't exist, so late-round upside is undervalued relative to
  a real managed season.
