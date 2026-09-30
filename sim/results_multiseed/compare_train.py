"""Cross-seed champion selection on TRAINING years only (2010-2020).

Evaluates every multi-seed v2 champion (plus the v1 champion and hand-tuned
default as anchors) in leagues of 9 noisy managed bots, on the same evaluation
rng stream, and ranks by mean fitness (wins + 3*title + pts/1000). The winner
earns AT MOST one holdout look - and only if it beats the v1 champion here.

Usage: python results_multiseed/compare_train.py [--seeds-per-year 20]
"""
import argparse
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from env import available_years, load_season, knowledge_view, run_draft, run_season, DEFAULT_MGMT
from genome import default_genome
from policy import GenomePolicy, NoisyAutopickPolicy

HOLDOUT_FROM = 2021


def mgmt_of(g):
    return {"var_tilt": g["var_tilt"], "waiver_thr": g["waiver_thr"],
            "trade_min": g["trade_min"]}


def load_genome(path):
    with open(path) as f:
        d = json.load(f)
    return {**default_genome(), **d.get("genome", d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-per-year", type=int, default=20)
    args = ap.parse_args()

    years = [y for y in available_years() if y < HOLDOUT_FROM]
    cands = [("v1 champion (anchor)", load_genome(os.path.join(HERE, "results_v1_fixed_2026-08-10", "champion.json"))),
             ("hand-tuned default (anchor)", default_genome()),
             ("v2 seed 1", load_genome(os.path.join(HERE, "results_v2_fixed_2026-08-10", "champion.json")))]
    for p in sorted(glob.glob(os.path.join(HERE, "results_multiseed", "champion_seed*.json"))):
        s = os.path.basename(p).replace("champion_seed", "").replace(".json", "")
        cands.append((f"v2 seed {s}", load_genome(p)))

    views = {y: knowledge_view(y, [x for x in available_years() if x < y]) for y in years}
    seasons = {y: load_season(y) for y in years}

    print(f"training-year comparison: {len(cands)} candidates x {len(years)} years "
          f"x {args.seeds_per_year} leagues vs 9 noisy managed bots\n")
    rows = []
    for name, g in cands:
        rng = random.Random(999)          # same evaluation stream for everyone
        fit = wins = titles = n = 0
        for y in years:
            for _ in range(args.seeds_per_year):
                r2 = random.Random(rng.random())
                bots = [NoisyAutopickPolicy(random.Random(r2.random())) for _ in range(9)]
                rosters = run_draft(views[y], [GenomePolicy(g)] + bots, r2)
                mgmt = [mgmt_of(g)] + [dict(DEFAULT_MGMT) for _ in range(9)]
                res = run_season(rosters, seasons[y], r2, mgmt=mgmt, pool=views[y])[0]
                fit += res["wins"] + 3.0 * res["champion"] + res["points"] / 1000.0
                wins += res["wins"]
                titles += res["champion"]
                n += 1
        rows.append((fit / n, wins / n, titles / n, name))
        print(f"  {name:28s} fitness {fit / n:5.2f}  wins {wins / n:5.2f}  title% {100 * titles / n:5.1f}")

    rows.sort(reverse=True)
    print(f"\nbest on training years: {rows[0][3]}")


if __name__ == "__main__":
    main()
