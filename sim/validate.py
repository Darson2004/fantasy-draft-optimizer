"""Final exam on HELD-OUT seasons the evolution never saw (2021+).

Puts a genome in a league with 9 ESPN-autopick bots, many times per holdout
year with shuffled draft slots, and reports championship rate / playoff rate /
average wins against the luck baselines (10% titles, 40% playoffs, 6.5 wins).

Usage:  python validate.py                          # champion + hand-tuned default
        python validate.py --leagues 300             # tighter error bars
        python validate.py --genome default          # just the hand-tuned model
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import REG_WEEKS
from env import (available_years, load_season, knowledge_view, run_draft,
                 run_season, DEFAULT_MGMT)
from genome import default_genome
from policy import GenomePolicy, AutopickPolicy, NoisyAutopickPolicy

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
HOLDOUT_FROM = 2021


def evaluate(genome, years, leagues, seed, clean_bots=False):
    genome = {**default_genome(), **genome}   # old 12-gene genomes get v2 defaults
    mgmt = [{"var_tilt": genome["var_tilt"], "waiver_thr": genome["waiver_thr"],
             "trade_min": genome["trade_min"]}] + [dict(DEFAULT_MGMT) for _ in range(9)]
    rng = random.Random(seed)
    rows = []
    for year in years:
        prior = [y for y in available_years() if y < year]
        view = knowledge_view(year, prior)
        season = load_season(year)
        titles = playoffs = wins = pts = 0.0
        for _ in range(leagues):
            r2 = random.Random(rng.random())
            bots = ([AutopickPolicy() for _ in range(9)] if clean_bots else
                    [NoisyAutopickPolicy(random.Random(r2.random())) for _ in range(9)])
            policies = [GenomePolicy(genome)] + bots
            rosters = run_draft(view, policies, r2)
            res = run_season(rosters, season, r2, mgmt=mgmt, pool=view)[0]
            titles += res["champion"]
            playoffs += res["made_playoffs"]
            wins += res["wins"]
            pts += res["points"]
        rows.append({"year": year, "titles": titles / leagues,
                     "playoffs": playoffs / leagues, "wins": wins / leagues,
                     "points": pts / leagues})
    return rows


def report(name, rows, leagues):
    n = len(rows) * leagues
    t = sum(r["titles"] for r in rows) / len(rows)
    p = sum(r["playoffs"] for r in rows) / len(rows)
    w = sum(r["wins"] for r in rows) / len(rows)
    se = math.sqrt(0.10 * 0.90 / n)
    print(f"\n=== {name}  ({len(rows)} holdout years x {leagues} leagues) ===")
    print(f"{'year':>6} {'title%':>8} {'playoff%':>9} {'avg wins':>9} {'avg pts':>9}")
    for r in rows:
        print(f"{r['year']:>6} {100 * r['titles']:7.1f}% {100 * r['playoffs']:8.1f}% "
              f"{r['wins']:9.2f} {r['points']:9.1f}")
    print(f"{'ALL':>6} {100 * t:7.1f}% {100 * p:8.1f}% {w:9.2f}")
    print(f"luck baseline: title 10.0% (SE {100 * se:.1f}%), playoff 40.0%, "
          f"wins {REG_WEEKS / 2:.1f}")
    sigmas = (t - 0.10) / se if se else 0.0
    print(f"title edge: {100 * (t - 0.10):+.1f} pts ({sigmas:+.1f} SE)")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", type=int, default=150, help="leagues per holdout year")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--genome", default="champion",
                    help="'champion', 'default', or path to a genome json")
    ap.add_argument("--clean-bots", action="store_true",
                    help="opponents = deterministic ESPN autopick (v1 behavior) "
                         "instead of adp_sd-jittered human-like boards")
    args = ap.parse_args()

    years = [y for y in available_years() if y >= HOLDOUT_FROM]
    if not years:
        sys.exit("no holdout seasons - run fetch_history.py first")
    print(f"holdout years: {years}")

    todo = []
    if args.genome in ("champion", "both"):
        path = os.path.join(RESULTS, "champion.json")
        if os.path.exists(path):
            with open(path) as f:
                todo.append(("evolved champion", json.load(f)["genome"]))
        else:
            print("no results/champion.json yet - run evolve.py; validating default only")
    if args.genome == "default" or args.genome in ("champion", "both"):
        todo.append(("hand-tuned default (draftmodel.py weights)", default_genome()))
    if not todo:
        with open(args.genome) as f:
            d = json.load(f)
        todo.append((args.genome, d.get("genome", d)))

    for name, g in todo:
        report(name, evaluate(g, years, args.leagues, args.seed, args.clean_bots),
               args.leagues)


if __name__ == "__main__":
    main()
