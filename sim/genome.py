"""Genome = the weight vector of the draft model. Defaults reproduce the
hand-tuned draftmodel.py; evolution perturbs them.

Genes:
  flex_rb/wr/te   how FLEX demand splits across positions (baseline depth calc)
  bench_beta      Fry backup discount for pure bench picks
  base_depth      scales the dynamic-baseline demand depth (VOLS <-> VORP dial)
  v2_weight       weight on next-pick value in the two-pick pair score
  bias_QB/RB/WR/TE  flat value bonus per position (evolvable construction prior)
  reach_pen       value penalty per 10 ADP spots of reaching above market
  kdst_early      how many picks before forced to start taking K/DST
  var_tilt        lineup tilt toward volatile players when projected to lose
                  the week (and toward floor players when projected to win)
  waiver_thr      expected weekly points a free agent must beat my worst
                  benched player by before I make the waiver swap
  trade_min       expected weekly lineup gain required to propose/accept a trade
"""
import random

GENES = {
    "flex_rb":    (0.05, 0.85, 0.45),
    "flex_wr":    (0.05, 0.85, 0.45),
    "flex_te":    (0.00, 0.40, 0.10),
    "bench_beta": (0.20, 1.00, 0.60),
    "base_depth": (0.50, 2.00, 1.00),
    "v2_weight":  (0.00, 1.50, 1.00),
    "bias_QB":    (-30.0, 30.0, 0.0),
    "bias_RB":    (-30.0, 30.0, 0.0),
    "bias_WR":    (-30.0, 30.0, 0.0),
    "bias_TE":    (-30.0, 30.0, 0.0),
    "reach_pen":  (-2.0, 4.0, 0.0),
    "kdst_early": (0.0, 3.0, 0.0),
    "var_tilt":   (0.0, 1.5, 0.3),
    "waiver_thr": (0.0, 3.0, 1.0),
    "trade_min":  (0.25, 3.0, 1.0),
}


def default_genome():
    return {k: v[2] for k, v in GENES.items()}


def random_genome(rng):
    return {k: rng.uniform(lo, hi) for k, (lo, hi, _) in GENES.items()}


def mutate(g, rng, rate=0.35, scale=0.15):
    out = dict(g)
    for k, (lo, hi, _) in GENES.items():
        if rng.random() < rate:
            out[k] = min(hi, max(lo, out[k] + rng.gauss(0.0, scale * (hi - lo))))
    return out


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENES}


def fmt(g):
    return " ".join(f"{k}={g[k]:.2f}" for k in GENES)
