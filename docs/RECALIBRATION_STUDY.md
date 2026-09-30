# Can we out-project the consensus? Backtest + 2026 recalibration (2026-09-21, week 2)


Code: `recal_study.py` (backtest, `--apply` writes `weekly_proj_2026_recal.json`);
`trades2.py --proj weekly_proj_2026_recal.json --no-shrink` consumes it.
Data: Sleeper's public per-week **frozen** projections and per-week stats (usage: targets,
carries, red-zone opportunities, snap share) for 2022–2026, 13,000 player-weeks 2023–25.

## Verdict

**Partly.** Two of the three hypotheses fail; one holds and is worth 2–3.5% RMSE at the horizon
that matters for trades.

| Hypothesis | Result | Evidence |
|---|---|---|
| Trailing **volume** (targets, carries, RZ, snaps) beats consensus | **No.** Full volume model (M2) ≈ benchmark or worse in every fold and horizon, including weeks 2–5 when projections still lean on August. | y1 2025: M2 +0.13% RMSE (worse); yrest 2025: −2.0% but *worse than calibration alone* |
| **Calibration** (projected gaps too wide) | **Yes, grows with horizon.** Next week −0.5%; 4-week −1 to −1.9%; rest-of-season −3.1% (t = 10.4). Biggest at QB. | Benchmark slopes: 1-wk 0.88–0.95, RoS 0.81–0.94; QB RoS 0.62 (2025) / 1.14 (2024) |
| **Snap share + "beats-his-projection" residual** on top of calibration | **Yes, small and consistent.** Best model in 7 of 8 cells. | M2s: y1 −0.6/−0.7%, y4 −1.4/−2.3%, yrest −0.2/−3.6% (t 3–9) |

Everything else the consensus already prices. This matches the August momentum study (point trends
add nothing once you control for the prior) and the FFA 12-season finding (projections exaggerate
talent gaps). The exploitable residual is *shape*, not *information*: the projections know who is
good; they overstate how much better.

## Backtest table (out of sample, RMSE, players who played, proj ≥ 2)

| Target | Fold | M0 bench | M1 calib | M2 volume | **M2s** | slope of bench |
|---|---|---|---|---|---|---|
| next week | 2023→2024 | 6.478 | 6.443 (−0.5%) | 6.475 | **6.433 (−0.7%, t 2.3)** | 0.951 |
| next week | 2023-24→2025 | 6.479 | 6.441 (−0.6%) | 6.487 | **6.442 (−0.6%, t 2.6)** | 0.880 |
| next 4 wks | 2023→2024 | 4.268 | 4.226 (−1.0%) | 4.241 | **4.208 (−1.4%, t 3.1)** | 0.942 |
| next 4 wks | 2023-24→2025 | 4.224 | 4.143 (−1.9%) | 4.175 | **4.127 (−2.3%, t 8.8)** | 0.856 |
| rest of season | 2023→2024 | 3.663 | 3.655 (−0.2%) | 3.659 | 3.655 (−0.2%) | 0.944 |
| rest of season | 2023-24→2025 | 3.665 | 3.551 (−3.1%) | 3.590 | **3.534 (−3.6%, t 7.7)** | 0.814 |

By position (RoS, 2025): QB 5.18 → 4.25 (−18%); RB 3.68 → 3.61; WR 3.55 → 3.61 (slightly worse);
TE 2.74 → 2.73. **The gain is overwhelmingly QB.** Week-level paired t-tests; 17–34 weeks per fold.

Caveat that matters: calibration is **not stable year to year**. 2024's benchmark was nearly
calibrated (slopes ≈ 1, QB 1.14) and shrinkage did nothing; 2025's was badly over-dispersed (QB 0.62)
and shrinkage was worth 3%. The pooled 2023–25 fit (QB 0.85 / RB 0.92 / WR 0.97 / TE 0.96) is the
honest middle. It never cost more than 0.2% in the bad year.

## What the model is

Per position, per horizon (next week / rest of season), ridge on four features:
`proj` (ESPN+Sleeper mean for the week), `ewm_snap` (EWMA snap share, half-life 3 wks),
`ewm_resid` (EWMA of actual − projection), `n_prior` (games played). Fitted on 2023–25.
Standardised coefficients (RoS): QB proj 2.34, snap 0.00, resid 0.12; RB 3.97 / 0.89 / 0.09;
WR 4.18 / 0.30 / 0.09; TE 2.93 / 0.58 / −0.08. Snap share is the one usage number that carries
information beyond the projection — most for RBs and TEs.

## Effect on 2026 values and on the trade search

- Applied to 2026 weeks 3–17 (consensus = ESPN), the dominant move is QB shrinkage: every top
  QB loses 35–45 rest-of-season points (e.g. Allen −44, Hurts −40, Burrow −35). Outside QB the
  largest moves come from snap share (a receiver on about half his team's snaps is marked down
  ~15 points) and from Sleeper disagreeing with ESPN.
- Fed into `trades2.py` (full 558k-package search, finalists at 1,500 scenarios), the
  recalibration left the *ranking* of trade families unchanged and moved magnitudes in one
  direction: packages that sell QB depth became better for the seller and worse for the buyer,
  because a flatter QB curve makes a backup QB worth less, while the crowd calculator a
  counterparty consults still prices him at market. The same gap identifies sell-high candidates:
  players the market carries well above their calibrated value.

## Limits

- Benchmark is Sleeper, not ESPN (no public ESPN history). The two agree closely on scale
  for top players (within ~3% on season totals), so the calibration transfers, but this is an inference.
- Two weeks of 2026 usage. `n_prior` is in the model, so early-season signals are weighted down.
- QB shrink is the model's strongest and least stable claim (slope 1.14 in 2024, 0.62 in 2025).
- No Vegas lines, no injury-adjusted team context; the consensus has those and we borrow them
  through `proj`.
