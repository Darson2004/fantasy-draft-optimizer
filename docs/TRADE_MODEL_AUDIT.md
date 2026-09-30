# Trade Model Audit — is `trades.py` state of the art? (2026-09-19, week 2)


Short answer: **no.** The draft-day model it inherits from was well-built for the draft
(lookahead over a deterministic autopick opponent, expected-wins objective). Re-pointed at
in-season trades it makes six identifiable errors, two of which flip the sign of typical
recommendations. `trades2.py` replaces it. This document records what the state of the art
actually is, where the old model sits against it, and what the new one does.

## 1. What "state of the art" means for in-season trade evaluation

There is no single canonical algorithm; the frontier is a set of components that the
best practitioner tools and the two relevant papers agree on.

| Component | SOTA practice | Source |
|---|---|---|
| **Objective** | Season-simulated change in P(playoffs) / P(title), not projected points or even expected wins. Wins have non-linear value: a win moves a bubble team's playoff odds far more than a locked-in team's. | Monte Carlo playoff-odds sims are the standard tool (e.g. [FantasyPlayoffOdds](https://rodger-yuan.github.io/FantasyPlayoffOdds/), [ff_monte_carlo](https://github.com/cdtdev/ff_monte_carlo), [robsim](https://richabdill.com/robsim/)); Summers–Swartz–Lockhart (2007) showed P(beat opponent) beats E[points] as a pick criterion. |
| **Lineup-level valuation** | Value a player by the change in the team's *optimal weekly lineup*, summed over remaining weeks, with free-agent fill-ins for roster gaps — never by raw projection. | Both papers: [GA trade optimizer, arXiv 2511.17535](https://arxiv.org/pdf/2511.17535) uses S(R) = Σ_w L(R,w) with FA fill-ins; [IBM/ESPN Trade Assistant, arXiv 2111.02859](https://arxiv.org/abs/2111.02859) prices "the cost of losing a player" by positional depth and slot count. |
| **Per-week projections** | Rest-of-season is a sum of *weekly* projections (matchup, bye, injury-return aware), not a season total divided by games. | Both papers; every commercial trade analyzer ([RotoWire](https://www.rotowire.com/football/article/fantasy-football-trade-analyzer-find-and-grade-redraft-fantasy-football-trades-in-2026-134904), RotoTrade) projects "rest-of-season and upcoming week". |
| **Playoff weighting** | Weeks 15–17 matter more than a regular-season week for a contender. | GA paper: α_p = 1.2 default, 1.5 "high playoff bias"; a real bracket sim does this endogenously. |
| **Mutual benefit / acceptance** | A trade is only proposable if the counterparty's roster also improves (g_b > 0), with a fairness penalty on lopsidedness; commercial tools add a crowd-value "fairness %" (≥ 85% = fair). | GA paper cost c = −(α·g_a^w + β·g_b − γ·|g_a^w − g_b|); IBM "parity, pain, fairness"; [FantasySP fair-trade rating](https://www.fantasysp.com/nfl_trade_analyzer/). |
| **Uncertainty** | Projections are noisy and *over-dispersed*: seasonal projections explain only 14–26% of within-position variance, calibration slopes are QB 0.67 / TE 0.72 / RB 0.79 / WR 0.85 (projected gaps between players are too wide), aggregated sources beat any single source 69% of the time. | [FFA, 12 seasons of projections (Aug 2026)](https://fantasyfootballanalytics.net/2026/08/we-analyzed-12-seasons-of-fantasy-football-projections-heres-what-we-found.html). The GA paper lists Monte Carlo uncertainty as its main missing piece. |
| **Injury / depth** | Bench depth has value because starters miss games: top-100-ADP RBs and WRs each miss ~3 games/yr; only 46% of RB seasons are "healthy" (0–1 missed). A 2-for-1 that thins the bench must be charged for that. | [RotoBanter 2021-25 injury study](https://rotobanter.beehiiv.com/p/are-running-backs-more-injury-prone-than-receivers). The GA paper explicitly did *not* model injuries or byes. |
| **Search** | Combinatorial: 1-for-1 through 3-for-3 in the GA paper (m = 3 per side), knapsack 0-1 + team-pairing by cosine dissimilarity in IBM's system, which served 239M proposals in 2020. | Both papers. |
| **Weekly score noise** | Team weekly score sd ≈ 22–25 points (12-team PPR mean 122, sd 22.6). | [Stathole WAR](https://statholesports.substack.com/p/fantasy-football-wins-above-replacement); my league's week-1 field sd was 23. |

## 2. Where `trades.py` stands

| # | Issue | Severity | What it does to recommendations |
|---|---|---|---|
| 1 | **Preseason projections, flat per week.** `proj` is the August ESPN+Sleeper season total; every week gets `proj / 17`. Week-1 results, role changes, and ESPN's own per-week revisions are ignored: a back whose role grew in September is still carried at his August number (e.g. 91.5 when ESPN now projects 120). | **High** | Every valuation is a month stale. Any trade that hinges on a September role change is priced wrong. |
| 2 | **Injuries priced by a flat multiplier** (IR × 0.30 on the *whole* season). ESPN's weekly feed already zeroes an IR player's weeks until his projected return — the correct shape. | Medium | IR players are simultaneously over- and under-valued depending on return date. |
| 3 | **Field-average opponent every week; standings ignored.** `evaluator()` scores each week against all nine opponents equally and ignores the record already banked. The real schedule is known and seeding is record then points-for. | Medium | Expected-wins deltas are roughly right in aggregate but say nothing about playoff odds — which is the thing that matters. |
| 4 | **Wrong playoff format.** `PLAYOFF_SEEDS = 4`, random weekly pairings, playoff strength = `max(mus)`; the league is **6 teams, top-2 byes, weeks 15–17**. `REG_WEEKS = 14` also drops the three playoff weeks from valuation entirely. | **High** | P(playoffs) and P(title) in `--grade` are for a league that doesn't exist. A 6-team format with byes makes the #2 seed worth much more than the model thinks, and makes weeks 15–17 projections matter. |
| 5 | **Depth is worth nothing except on bye weeks.** No injury sampling, no projection error. In this model a 2-for-1 that lands a star and leaves a thin RB room behind is pure upside. | **High** | Systematically over-recommends consolidation and under-values handcuffs/depth. This is the error that flips signs. |
| 6 | **Unbalanced trades aren't balanced.** Give 2, get 1 → the model plays with 15 men and never fills the open spot; get 2, give 1 → 17 men, nobody dropped. Streaming fill exists only for QB/K/D-ST. | Medium | 2-for-1s look worse than they are (no FA fill credited) *and* better than they are (no depth charge) — the two errors partly cancel by accident. |
| 7 | Search stops at 4 bodies total; the shortlist keeps interchangeable filler. | Low | Misses 3-for-2 / 3-for-3 packages, which is exactly where roster-balance trades live. |
| 8 | Acceptance model: their expected-wins delta plus FantasyCalc optics. | Fine | This part was already close to SOTA; kept. |

What it got right and `trades2.py` keeps: expected-wins-via-Φ as the fast screen, best-legal-lineup
valuation, streaming-aware QB/K/D-ST slots, FantasyCalc as the counterparty's calculator,
the two-sided report.

## 3. What `trades2.py` does

Pipeline: `inseason_proj.py` (ESPN public feed → per-week projections for weeks 1–17, every
player, validated to ±0.1 against the league's own numbers) → `league_state.json` (rosters incl.
IR, full schedule, records, 6-team settings) → three-stage funnel.

1. **Stage 1 — deterministic screen, all packages.** Every shape 1-for-1 … 4-for-4 (exhaustive to
   4 bodies, top-9/top-8 pools beyond), ≈ 62k packages per opponent, 550k total, 0.6 ms each.
   Score = Δ expected wins on the *real* schedule from next week on (this week's lineups are locked
   to current rosters — a trade proposed today lands next week). Unbalanced packages are balanced:
   holes filled by the best FA at RB/WR/TE, surplus dropped from the lowest remaining projection.
   Acceptability tiers: **MUTUAL** (their own lineup math ≥ −0.02 wins and FantasyCalc optics
   ≥ −12%) and **OPTICS** (their lineup loses up to 0.2 wins but the calculator they'll look at
   favours them by > 8%).
2. **Stage 2 — vectorised season Monte Carlo, top 300.** 4,000 seasons on the real schedule with
   banked records and points-for, seeding by record then PF, the 6-team bracket (1-2 bye; 3v6,
   4v5; 1 v 4/5, 2 v 3/6; final) on weeks 15–17 projections. Common random numbers across
   candidates so paired differences are tight. Reports ΔE[wins], ΔP(playoffs), ΔP(bye), ΔP(title)
   for both sides.
3. **Stage 3 — injury + projection-error Monte Carlo, top 40.** 300 scenarios, each with a
   persistent per-player projection error (lognormal sd QB .15 / RB .25 / WR .22 / TE .25) and
   sampled injuries (weekly hazard RB 4.5% / WR & TE 3.5% / QB 2%, geometric spells mean 3 weeks,
   15% season-ending — this reproduces ≈ 3 missed games per RB-season and a 45–50% chance of
   missing at least one). Lineups are re-optimised inside every scenario, so bench depth is
   finally priced. This is the number to trust.

Calibration shrinkage (`CALIB_SLOPE`, on by default, `--no-shrink` to disable) regresses each
weekly projection toward the starter-pool mean at the position by the FFA slopes above, which is
the published correction for projections exaggerating talent gaps. The search is run both ways;
trades that survive both are robust to the choice.

Cost: ~6 minutes for the full funnel; 0.6 s for the roster audit.

### Deliberate simplifications (known, documented)

- D/ST weekly numbers are Sleeper's (ESPN's raw D/ST line needs league points-allowed tiers);
  every team fills the slot identically so it nets out of trade deltas.
- Injury hazards are position-level, not player-level (age, history): a 32-year-old RB carries the
  same hazard as a 23-year-old. A player-level model would need injury-history data we don't have.
- Weekly team noise is Normal(0, 25) at team level; player-level weekly variance and
  same-team correlations are folded into that.
- Three-team trades are not searched. ESPN executes them as two bilateral trades anyway.
- Roster position limits are not enforced beyond total size.
- Weekly projections are a single source (ESPN). FFA's finding that aggregation beats any single
  source 69% of the time argues for blending Sleeper's weekly feed next; not done yet.

## 4. What the new model found — and what it changed

Four full runs on the real league (558k packages each; ±shrinkage, ± one star unavailable),
finalists re-scored at 1,500 scenarios. The league-specific tables are not published; the
aggregate lessons were:

- **Same partner, different package.** Old and new models picked the same counterparty (the team
  structurally short at RB and deep at WR), but disagreed on the package: the new model could
  see that a benched backup QB is worth ~0 wins to me and a real starter to a team that needs one,
  and used him as the sweetener. The old model could not price the dead QB2 at all.
- **Depth changes acceptability, not just my side.** Stage 1 (deterministic) tiered every deal
  without the star as OPTICS (counterparty loses). Stage 3 (injury-sampled) showed two
  counterparties actually *gaining* +0.03–0.10 W from the bench depth they receive, which turned
  three deals from "counterparty loses" into "fair" — the depth-value effect the literature
  predicts.
- **No 1-for-1 clears the bar.** The value was in 2-for-1 and 3-for-1 consolidation; 3-for-3 and
  4-for-4 shapes add ΔW only by taking it from the counterparty.
- **Shrinkage barely moves the ranking.** Same top families both ways; deltas shift ≤ 0.1 W.
  It slightly deflates star-for-depth trades and inflates FA replacement level (a known
  extrapolation of the FFA slopes to sub-starter players).
- **Stage-3 noise is ±~1% title even at 1,500 scenarios.** The model separates families of
  trades, not neighbours within a family. Rankings inside ±1% are ties.

Judgment overlays the model can't make: player age (hazards are position-level), keeper value
for next season, what a counterparty has said about untouchable players, and the league's veto
rule (lopsided packages die regardless of the math).

### Illustrative output (synthetic demo league in `examples/inseason/`, not real data)

```
$ cd examples/inseason && python ../../src/trades2.py --quick --opp t3 --top 3
stage 1: 32,529 packages in 18s -> 15,240 improve me by >0.03 wins
         946 MUTUAL (them dWins > -0.02 and optics > -12%), 673 OPTICS (...)
stage 2: 300 candidates through 4000-sim season MC in 4s

opp  shape   give                                        get               dW me  dPO me  dTitle | dW them  calc
t3   3-for-1 Emeka Egbuka + Travis Kelce + Josh Jacobs   Javonte Williams  +0.50   +5.2%   +6.6% |   -0.15  +1242  OPTICS
            (I add FA Tyler Warren+Najee Harris; t3 drops Devaughn Vele+Rashod Bateman)
t3   3-for-1 Jaylen Waddle + Travis Kelce + Rome Odunze  Javonte Williams  +0.51   +5.4%   +6.4% |   -0.19   +924  OPTICS
```

Unbalanced packages are auto-balanced from free agents (the synthetic league's free-agent pool is
unrealistically deep, which is why these 3-for-1s look so good for "me").
