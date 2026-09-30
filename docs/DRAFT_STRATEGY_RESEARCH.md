# Algorithmic Drafting Methods That Beat Autodraft — Deep Research

*Compiled 2026-08-06. Context: a private 10-team ESPN league, snake, a late first-round slot,
16 rounds, PPR assumed, keepers. Baseline to beat = `autodraft.py`, a faithful replica of
ESPN autopick: greedy highest-board-rank available that fills a positional need, K/DST deferred.*

---

## 0. The one-paragraph answer

Yes — several methods demonstrably beat greedy autodraft, and the literature converges on **one
core mechanism**: autodraft ignores *what will still be on the board at your next pick*. Every
winning method — VONA, dynamic VBD, the Fry et al. dynamic program, MCTS, RL drafters — is a
different machine for exploiting that same blind spot. The peer-reviewed result (Fry, Lundberg
& Ohlmann 2007) is the sharpest: **against deterministic opponents, lookahead drafting strictly
dominates greedy in every test instance** (+2.4–7.5% total roster value; +3.7% vs rank-greedy).
A league full of ESPN autodrafters *is* the deterministic-opponent case — their future picks are
perfectly computable, which makes this league close to the best-case environment for these
methods. Measured edges are real but modest: ~3–4% of roster value, ~2–4 starting-lineup points
per week. Projection error (±30–70 pts MAE per player-season) dwarfs any strategy edge, so the
robust methods spend their sophistication on *price and timing* (which is measurable) rather
than out-projecting the market (which nobody reliably does).

---

## 1. Why greedy autodraft is beatable at all (theory)

- **Sincere picking is provably suboptimal.** Brams & Straffin (1979, *Amer. Math. Monthly*):
  with known preferences, a team gains by passing on its top player whenever he'll survive to
  its next pick. Autodraft never passes. https://www.jstor.org/stable/2321942
- **Optimal play is intractable, so everything practical is a heuristic.** Kalinowski,
  Narodytska & Walsh (AAAI 2013): the 2-player snake draft is solvable in linear time; for ≥3
  players computing even one subgame-perfect equilibrium is **PSPACE-hard**. Hence DP
  restrictions, MCTS, and RL approximations. https://cdn.aaai.org/ojs/9877/9877-13-13405-1-2-20201228.pdf
- **The edge exists only because opponents are naive.** Aziz, Goldberg & Walsh (2017): against
  equally sophisticated opponents, strategic gains vanish in equilibrium. The exploitable
  surplus comes from opponents not playing equilibrium — which autodrafters, by construction,
  do not. https://arxiv.org/abs/1705.09444
- **Humans are also predictably naive.** Lee & Liu (2022, *Judgment and Decision Making*;
  1,350 Sleeper leagues, 188k picks): stereotyped position sequencing, measurable positional
  herding (QB early, K/DST always), and the most common roster compositions win ~50% of games
  while less common RB/WR-heavier builds win 55%+.
  https://sjdm.org/~baron/journal/22/220318/jdm220318.html

---

## 2. The methods, ranked by strength of evidence

### 2.1 Lookahead dynamic programming — Fry, Lundberg & Ohlmann (2007) ★ the key paper

*JQAS 3(2). PDF: https://web.archive.org/web/2020id_/http://people.emich.edu/aross15/math319/player-selection-heuristic-sports-league-draft.pdf*

Models the snake draft as a stochastic DP, made deterministic by assuming you know each
opponent's valuations, needs, and strategy — then tractable via two restrictions:

- **R1:** project one future available-player pool per round (opponents pick off a known rule,
  e.g. rank order) — exactly the forward simulation `autodraft.py` already does.
- **R2:** only consider drafting players who will NOT survive to your next pick
  (`Ω_t = A_t − A_{t+1}`). Everyone who survives is free — take them later.

Objective: total projected points, with a **β = 0.6 discount on backups**. Solved as an LP,
re-solved every round with a 5–6 round horizon; seconds per solve.

**Results (10-team, 16 rounds):** with perfect knowledge of opponent strategies the DP
dominated all four baselines in all instances — avg 2200.1 pts vs 2138.5 for ESPN-style
need-filling autodraft (+2.9%) and 2036.0 for greedy-by-projected-points (+7.5%, the *worst*
baseline). With a crude opponent model (assume everyone need-fills), it still won on average;
with the right league-level *mix* of strategies (wrong per-team assignment) it dominated again.
Opponent-model quality is worth ~0.6–1.8% of roster value — you only need it right in aggregate.

Two corollaries worth stealing: greedy-by-raw-projections is worse than drafting by ADP rank
(baseline subtraction matters — §2.3), and snake slot values are non-monotonic (slot 8 out-drafted
slots 6–7).

### 2.2 VONA — Value of Next Available (the practitioner version of the same idea)

```
VONA(player) = proj(player) − proj(best same-position player available at your NEXT pick)
```

Pick the position with the max drop-off across the wrap. If a player (or his near-equal)
survives to your next pick, his VONA ≈ 0 — **never reach for what waits**. This is the purest
exploit of snake geometry and of the long gap between turn pairs at a late slot. No formal backtest exists in the
practitioner record, but it is a one-step special case of Fry's R2, which is validated.
Definitions/worked examples: https://www.fantasypros.com/2025/06/fantasy-football-draft-strategy-value-based-drafting-vorp-vols-vona/ ·
implementation writeup: https://stanfordstevens.com/value_of_vona.html

Failure modes: one-step myopia (a position can bleed value slowly and never win a single VONA
comparison), needs a roster-need multiplier bolted on, signal collapses to ~0 late.

### 2.3 Value-Based Drafting (VBD) — the correct value scale

*Joe Bryant, Footballguys 1996: https://www.footballguys.com/article/bryant_vbd*

```
value(player) = proj_points(player) − proj_points(baseline player at same position)
```

Draft by descending value across positions, not raw points or site rank. The baseline choice is
the entire design space and silently flips QB-vs-WR orderings:

| Baseline | Definition | Bias |
|---|---|---|
| VOLS / worst starter | last league-wide starter (QB10, RB20+flex share in 10-team) | inflates elite QB/TE |
| VORP | best freely-available player (first bench/waiver) | inflates RB and depth |
| 100-pick (Bryant, FFA) | Nth at position, N = how many go by overall pick 100 | balanced; the classic default |
| BEER / man-games (Dupont 2012) | enough players to cover a season of starts given byes+injuries | balanced starter/bench |

Evidence is indirect but solid: averaged projections predict season points at **R² = .53 vs
.27 for expert ranks and .23 for ADP** (FantasyFootballAnalytics, 167 rank sources, 11
projection sources: https://fantasyfootballanalytics.net/2016/04/accuracy-of-rankings-vs-projections.html),
and Fry's result that raw-points greedy is the *worst* strategy shows the baseline subtraction
is what converts projections into drafting value. Corrections that matter (FFA 12-season
accuracy panel, https://fantasyfootballanalytics.net/which-projections-are-most-accurate):
- **Average many projection sources** — the average beat individual sources in 69% of
  head-to-heads; paid sources are not better than the free crowd.
- **Shrink projected gaps** — calibration slopes are QB 0.67 / TE 0.72 / RB 0.79 / WR 0.85
  (projections exaggerate tier spread), and every source over-projects top players
  (+17 to +46 pts, QB worst).
- Expect MAE per player-season of ~QB 61–71 / RB 52–54 / WR 40–42 / TE 29–31 pts.

### 2.4 Dynamic VBD — recompute the baseline mid-draft

Static VBD goes stale the moment a positional run happens. Dynamic VBD re-evaluates
`baseline_pos(t)` after every pick from the remaining pool, remaining demand, and your roster.
Taken to its limit (baseline = best available at your next pick) it *becomes* VONA — the two
converge. Commercial implementations (Draft Sharks "Dynamic MVP", Footballguys Draft Dominator)
publish no backtests. Guard against run-chasing: anchor to ADP so a QB run doesn't make the tool
recommend joining the panic at its most expensive point.

### 2.5 ADP survival models — probabilistic "will he be there?"

The standard practitioner model (Zimmerman 2017 / DraftKick,
https://draftkick.com/blog/projected-availability/):

```
P(available at pick n) = 1 − Φ((n − ADP) / σ),   σ ≈ ADP/4
```

Decision rules: ≥60–70% survival → wait; ≤20–30% → draft now. Probabilistic VONA replaces the
point-estimate "best available next pick" with the survival-weighted expectation. Known defects:
real pick distributions are right-skewed (fallers slide far; risers can't go before pick 1), and
σ≈ADP/4 is crude. Per-player empirical distributions (Fantasy Football Calculator publishes
ADP + SD: https://fantasyfootballcalculator.com/adp) or beta fits (see §2.8) are better.

**The empirical price-discipline result (largest-N finding in this report):** Establish The Run
on Best Ball Mania V (~677k entries): teams in the top decile of accumulated ADP value advanced
**21.8% vs 11.8%** for the bottom decile (base 16.7%) — ~57 pts over 14 weeks ≈ **4 pts/week
purely from never reaching** (https://establishtherun.com/the-power-of-the-market-in-best-ball-mania/).
Underdog's 5-year study: teams reaching ≥5 picks above ADP **lost points every single year**
(https://underdognetwork.com/football/best-ball-research/what-has-worked-in-all-5-years-of-best-ball-mania).

### 2.6 Platform-rank vs ADP arbitrage — the direct exploit against an ESPN autodraft room

ESPN and Yahoo default ranks diverge from market ADP by **10–30+ spots** on specific players
(documented examples: https://thefantasyroadshow.substack.com/p/hacking-your-fantasy-football-draft).
Autodrafters consume the *platform rank*, so a player ESPN's board buries will predictably last
until exactly the pick where its rank surfaces him. Against rank-followers you know the precise
last feasible pick for every mispriced player and can harvest them at zero opportunity cost.
`board_2026.json` already carries both `rank_ppr` and `adp` — the arb signal is
`rank_ppr − adp_rank` per player, and the forward sim tells you the harvest deadline.

### 2.7 Tier-based drafting — Boris Chen's GMM tiers

*http://www.borischen.co/ (method: 2013 post; still maintained).* Fit a 1-D Gaussian mixture to
expert-consensus ranks per position; components = tiers of statistically indistinguishable
players; **take the last player of a dying tier, defer positions with deep unbroken tiers**.
It's a human-executable discretization of the dropoff curves and a poor-man's VONA. No published
win-rate study. Weakness: clusters rank-space, not points-space — late-draft rank compression
manufactures fake tiers.

### 2.8 Monte Carlo / MCTS drafters

- **Keuter MCTS** (Towards Data Science 2019, code https://github.com/ykeuter/ffl): UCT over the
  draft tree with the key trick — **actions are positions, not players** (choose position →
  auto-take its top projected player; hard position caps), collapsing branching to ≤6. Roster
  evaluation weights lineup slots and values unfilled slots at the mean of the top-3 remaining.
  ~1,000 iterations/pick suffices. Result: beats greedy and random at every slot; greedy tested
  *worse than random* (echoing Fry). Weak spot: uniform-random rollouts, no ADP opponent model.
- **Two-pick wrap optimization** (Jensen, "Simulating the Snake",
  https://bcjense6.medium.com/simulating-the-snake-an-ai-assisted-fantasy-football-draft-strategy-4064c98940f7):
  simulate the wrap (all picks until your next turn) hundreds of times with opponents =
  `argmax(−ADP + noise)`; choose the pick maximizing `E[points over BOTH your wrap picks] − λ·SD`.
  Directly built for turn slots like 9/10.
- **MOBA analog (strongest algorithmic draft results anywhere):** DraftArtist (RecSys 2018,
  https://arxiv.org/abs/1806.10130) — pure MCTS beats greedy 61.1%; JueWuDraft (IEEE ToG 2021,
  https://arxiv.org/abs/2012.10171) — AlphaZero-style policy/value nets + MCTS beats pure MCTS
  54.8→60.1%. Template: projection model → value net, ADP model → policy prior, MCTS on top.

### 2.9 Reinforcement learning drafters

- **approximatemethods.com REINFORCE drafter** (https://www.approximatemethods.com/fantasy.html):
  actions = 4 position groups; opponents = **beta distributions fit to per-player ADP stats**
  (the best published stochastic-ADP opponent model); reward = final roster projection minus a
  random-fill baseline. Converged policy ≈ +250 projected pts vs baseline; live: 2nd-best
  projected roster from a bad slot.
- **Fonseca SB3 drafter** (https://fonseca-carlos.medium.com/fantasy-football-draft-with-reinforcement-learning-78f02af37bf4):
  +72 pts (+3.6%) over best-available greedy from the same slot — but trained and evaluated on
  the same season's actuals (hindsight leakage), so treat as an upper bound.
- Both converge to **positional sequencing policies** (when to take QB/TE, RB-vs-WR order), not
  player-specific insight — consistent with the idea that the edge is timing, not projection.

### 2.10 Mixed-integer programming — Becker & Sun (2016)

*JQAS 12(1), https://www.degruyterbrill.com/document/doi/10.1515/jqas-2013-0009/html.* Robust
MIP re-solved at every pick (~700 integer vars), opponents anticipated from expert lists/mock
behavior; uniquely optimizes **weekly head-to-head wins**, not points. Outperformed
expert-ranking drafting "by a statistically significant margin under simulation"; their
salary-cap bot placed top 0.06% of 50k+ entries. Heavy machinery for a home league, but the
objective correction is right.

### 2.11 Objective corrections (what to maximize)

- **Realized best lineup, not sum of projections** (dlm1223,
  https://github.com/dlm1223/fantasy-football): sample position-specific projection *errors*
  (from 2009–2017 history: RB errors high-variance, top-RB projections biased high) and evaluate
  each candidate roster by the distribution of its realized best starting lineup.
- **P(beat opponents), not E[points]** (Summers, Swartz & Lockhart 2007, hockey pools,
  https://www.sfu.ca/~tswartz/papers/hockey.pdf): the pick criterion "maximize average
  probability my lineup beats each opponent's" beat every greedy variant (6.54/9 lineups
  defeated vs 6.40); notably max-points greedy had *higher* P(1st) — **variance-seeking wins
  outright, mean-maximizing wins on average**. Pick your objective by format.
- **Variance is worth 33–50% as much as points** in top-heavy formats (Fantasy Footballers
  regression on best-ball outcomes: holding points fixed, +100% SD lifts top-6 odds QB +6% /
  RB +9% / WR +8% / TE +4%,
  https://www.thefantasyfootballers.com/articles/embracing-volatility-quantifying-the-value-of-variance-in-best-ball-fantasy-football/).
  For a 10-team H2H league: draft near-mean with mild ceiling preference; for tournaments flip to
  upside percentiles.

### 2.12 Roster-construction priors (the named "strategies")

Each named strategy is a prior on baseline placement/variance imposed as round-window
constraints — all downstream of VBD. Current large-N evidence (Underdog BBM, ~450k+ rosters/yr;
Stathole 12k-draft sim; Fantasy Footballers sim):

- **Round-1/2 RB is back**: R1 RB ≈ 24% advance vs ≈12% without; RB-RB start ≈ 30%; zero-RB
  start ≈ 8% (BBM VI, https://www.4for4.com/2026/preseason/how-winners-draft-running-backs-underdog-best-ball-mania).
  Heavy RB beat Zero RB by ~2 pts/week, 53% head-to-head (Fantasy Footballers sim).
- **Elite QB+RB maximized championship-tail** in the Stathole shootout; pure late-round-QB
  tested *worst* there — but the QB curve is genuinely flat in the middle (QB4–9 gap ≈ 3 PPG,
  Sharp Football 2010–25), so "QB in the value window, not never-early and not panic-late."
- Consensus best-ball allocation: 2–3 QB / 5–6 RB / 6–9 WR / 2–3 TE; "bully TE" (2 TEs early)
  is the worst documented construction (10.2% advance).
- These are regime-dependent (the 2024–25 RB renaissance inverted 2015-era Zero RB evidence);
  encode them as soft priors, not hard rules.

---

## 3. What this means for a late slot in an autodraft-heavy league

1. **This league is Fry's perfect-information case.** Opponents who autodraft are deterministic
   given the board. The forward simulation in `autodraft.py` (everyone takes best-available)
   already reconstructs their picks; against them, the R2 rule is provably right: **at each pick,
   only consider players who will not survive to my next pick.**
   Everyone else is free later. Human league-mates add noise — handle them with the σ=ADP/4
   survival model instead of determinism, or better, per-player empirical ADP distributions.
2. **Swap the value scale.** Replace ESPN `rank_ppr` as the pick criterion with
   VBD value = averaged-multi-source projection − dynamic positional baseline. Keep ESPN rank
   only as the *opponent model* (it's what the other autodrafters consume — that's its real use).
3. **Harvest the rank-vs-ADP arb.** `board_2026.json` already has both columns. Players with
   ESPN rank ≫ market ADP will last exactly until their rank surfaces; schedule their capture at
   your last feasible pick.
4. **Optimize pick pairs at the turn.** A late slot's picks come in pairs (e.g. picks 8/13,
   28/33 at slot 8 of 10) with a long desert between. Evaluate {pick A now, pick B at the wrap-back} jointly over the
   simulated wrap — Jensen's two-pick utility, or full MCTS with position-abstracted actions.
5. **Keeper adjustment.** Keeper removals of elite players
   thin the top of the RB/WR pool, which shifts every positional baseline — dynamic VBD handles this
   automatically where a static cheat sheet silently misprices.
6. **Expectation management.** Total measurable edge ≈ 2–4 pts/week — real and worth taking,
   but one QB projection miss (~65 pts ≈ 4 pts/wk) cancels it. Robustness (averaged projections,
   shrunk tiers, no reaching) is worth more than cleverness.

### Implementation ladder (each step subsumes the last)

| Step | What | Cost | Expected gain |
|---|---|---|---|
| 1 | VONA on top of the existing forward sim: at each of your picks, compute per-position dropoff to your next pick; take max-dropoff position's best player; never take survivors | ~50 lines | The bulk of the documented +3–4% |
| 2 | VBD values from ≥3 averaged projection sources, shrunk toward ADP; dynamic baseline recompute each pick | data plumbing | Better value scale; robust to ESPN rank quirks |
| 3 | Stochastic opponents: `P(gone) = Φ((n−ADP)/(ADP/4))` (or per-player empirical SDs) + two-pick pair optimization at the turn | ~150 lines | Robustness to human league-mates; turn-specific edge |
| 4 | MCTS (position-abstracted actions, ADP-calibrated rollouts) with evaluation = realized best lineup under sampled position-specific projection errors | a weekend | Marginal over 1–3 in a 10-teamer; this combination is literally unpublished (see §4) |

---

## 4. Gaps nobody has filled (from the academic sweep)

1. No peer-reviewed work combines a **stochastic ADP opponent model with MCTS/RL** for the
   fantasy snake draft — Fry has the DP with deterministic opponents, the RL blogs have
   stochastic opponents without search, JueWuDraft has search+nets but 2-player.
2. **Objective mismatch is under-studied**: only Becker & Sun optimize H2H wins, only Summers
   et al. optimize P(beating opponents); everything else maximizes points, which is provably not
   the same target.
3. **Correlation-aware drafting** (stacking inside a draft simulator) has no open-source
   implementation despite Week-17 stack evidence (+~50% finals-win odds) in best ball.
4. n-player general-sum MCTS with heterogeneous opponent types has no academic treatment
   (PSPACE-hardness explains why); Lee & Liu's data suggests a few learned opponent types
   would suffice in practice.

---

## 5. Source quality map

- **Peer-reviewed:** Fry/Lundberg/Ohlmann 2007 (JQAS); Becker & Sun 2016 (JQAS); Summers et al.
  2007 (CRC); Matthews et al. 2012 (AAAI, FPL not snake); Lee & Liu 2022 (JDM); Brams & Straffin
  1979; Kalinowski et al. 2013 (AAAI); Aziz et al. 2017; Chen et al. 2018 (RecSys); JueWuDraft
  2021 (IEEE ToG); Haugh & Singal 2021 (Mgmt Sci, DFS opponent modeling, 350% season return).
- **Large-N industry:** ETR "Power of the Market" (BBM V); Underdog 5-year BBM study; Sharp
  Football ADP panels (2010–25); FFA 12-season projection-accuracy panel.
- **Transparent blog sims (no review):** Stathole Sports 12k-draft shootout; Fantasy Footballers
  Mythbusters; dlm1223; Keuter MCTS; approximatemethods RL; Jensen; Fonseca (hindsight-tainted).
- **Vendor claims (no backtests published):** Draft Sharks Dynamic MVP, Footballguys Draft
  Dominator, FantasyPros Draft Wizard / Pick Predictor.
- **Over-cited non-evidence:** FantasyFootballAnalytics is frequently cited as a strategy
  shootout — it is a VOR framework + uncertainty tool, not a head-to-head strategy simulation.
