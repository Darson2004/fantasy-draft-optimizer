"""Evolutionary tournament: 10 genome-drafters per league, league champions
pass their genes on (with crossover + mutation), repeated over the TRAINING
seasons only.

Fitness inside a generation = regular-season wins + 3 x championship +
points/1000 (tiebreak). Each league's playoff champion survives to the next
generation unchanged ("winners move on" - keeps variance); the rest
of the next generation is bred from fitness-weighted parents.

Two ways to run:

FIXED RUN (default): N generations, then candidates are re-evaluated against
9 ESPN-autopick bots across all training years (2010-2020); best mean fitness
is written to results/champion.json.

CONTINUOUS (--loop): runs indefinitely with an overfitting guard. The last
training-era seasons (default 2019-2020) are carved out as a GUARD slice the
evolution never trains on; every --check-every generations the current best
genomes are scored on the guard slice vs autopick bots. The all-time guard
best is remembered; after --patience checks with no new best, the loop stops
and the guard-best genome becomes the champion. Ctrl-C is safe: state is
checkpointed every generation.

Every run checkpoints population + hall of fame to results/population.json;
--resume continues from the checkpoint instead of restarting.

Holdout years (2021+) are never touched here - see validate.py.

Usage:  python evolve.py                          # fixed 40 generations
        python evolve.py --resume                 # continue previous run
        python evolve.py --loop --resume          # run until the guard stalls
        python evolve.py --gens 5 --leagues 2     # smoke test
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import (available_years, load_season, knowledge_view, run_draft,
                 run_season, DEFAULT_MGMT)
from genome import default_genome, random_genome, mutate, crossover, GENES
from policy import GenomePolicy, NoisyAutopickPolicy

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CKPT = os.path.join(RESULTS, "population.json")
HOLDOUT_FROM = 2021
TITLE_BONUS = 3.0


def fitness(res):
    return res["wins"] + TITLE_BONUS * res["champion"] + res["points"] / 1000.0


class Cache:
    def __init__(self):
        self.view, self.season = {}, {}

    def get(self, year):
        if year not in self.view:
            prior = [y for y in available_years() if y < year]
            self.view[year] = knowledge_view(year, prior)
            self.season[year] = load_season(year)
        return self.view[year], self.season[year]


def mgmt_of(g):
    return {"var_tilt": g["var_tilt"], "waiver_thr": g["waiver_thr"],
            "trade_min": g["trade_min"]}


def play_league(genomes, year, cache, rng):
    view, season = cache.get(year)
    rosters = run_draft(view, [GenomePolicy(g) for g in genomes], rng)
    return run_season(rosters, season, rng,
                      mgmt=[mgmt_of(g) for g in genomes], pool=view)


def eval_vs_autopick(genome, years, cache, seeds, rng):
    """Mean fitness of one genome in leagues of 9 noisy-autopick (human-like) bots."""
    tot, n = 0.0, 0
    for year in years:
        view, season = cache.get(year)
        for _ in range(seeds):
            r2 = random.Random(rng.random())
            bots = [NoisyAutopickPolicy(random.Random(r2.random())) for _ in range(9)]
            rosters = run_draft(view, [GenomePolicy(genome)] + bots, r2)
            mgmt = [mgmt_of(genome)] + [dict(DEFAULT_MGMT) for _ in range(9)]
            tot += fitness(run_season(rosters, season, r2, mgmt=mgmt, pool=view)[0])
            n += 1
    return tot / n


def dedupe(genomes):
    seen, out = set(), []
    for g in genomes:
        key = tuple(round(g[k], 3) for k in GENES)
        if key not in seen:
            seen.add(key)
            out.append(dict(g))
    return out


def save_ckpt(gen, pop, hall, best_guard, checks_since):
    os.makedirs(RESULTS, exist_ok=True)
    with open(CKPT, "w") as f:
        json.dump({"gen": gen, "pop": pop, "hall": [[fv, g] for fv, g in hall],
                   "best_guard": best_guard, "checks_since": checks_since}, f)


def load_ckpt():
    if not os.path.exists(CKPT):
        return None
    with open(CKPT) as f:
        d = json.load(f)
    d["hall"] = [(fv, g) for fv, g in d["hall"]]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=None,
                    help="generations (default: 40 fixed mode, unlimited with --loop)")
    ap.add_argument("--leagues", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue from results/population.json")
    ap.add_argument("--loop", action="store_true",
                    help="run until the overfitting guard stalls (--patience)")
    ap.add_argument("--check-every", type=int, default=25, help="loop: gens between guard checks")
    ap.add_argument("--patience", type=int, default=3,
                    help="loop: guard checks without a new best before stopping (0 = never stop, run until Ctrl-C)")
    ap.add_argument("--guard-from", type=int, default=2019, help="loop: first guard year (excluded from training)")
    ap.add_argument("--guard-seeds", type=int, default=8, help="loop: leagues per guard year per candidate")
    ap.add_argument("--final-seeds", type=int, default=3, help="fixed mode: seeds/year in final selection")
    args = ap.parse_args()

    all_years = [y for y in available_years() if y < HOLDOUT_FROM]
    if not all_years:
        sys.exit("no training seasons - run fetch_history.py first")
    if args.loop:
        guard_years = [y for y in all_years if y >= args.guard_from]
        train_years = [y for y in all_years if y < args.guard_from]
        if not guard_years or not train_years:
            sys.exit(f"--guard-from {args.guard_from} leaves an empty slice")
    else:
        guard_years, train_years = [], all_years
    print(f"training years: {train_years}")
    if guard_years:
        print(f"guard years (never trained on): {guard_years}")

    cache = Cache()
    os.makedirs(RESULTS, exist_ok=True)
    pop_size = args.leagues * 10

    start_gen, hall, best_guard, checks_since = 1, [], None, 0
    ckpt = load_ckpt() if args.resume else None
    if ckpt:
        pop = [{**default_genome(), **g} for g in ckpt["pop"]]
        while len(pop) < pop_size:
            pop.append(mutate(dict(pop[len(pop) % len(ckpt["pop"])]), random.Random(len(pop))))
        pop = pop[:pop_size]
        hall = [(fv, {**default_genome(), **g}) for fv, g in ckpt["hall"]]
        best_guard = ckpt.get("best_guard")
        if best_guard:
            best_guard["genome"] = {**default_genome(), **best_guard["genome"]}
        checks_since = ckpt.get("checks_since", 0)
        start_gen = ckpt["gen"] + 1
        print(f"resumed from checkpoint: gen {ckpt['gen']}, hall size {len(hall)}"
              + (f", guard best {best_guard['score']:.2f} (gen {best_guard['gen']})" if best_guard else ""))
    else:
        rng0 = random.Random(args.seed)
        pop = [default_genome()]
        while len(pop) < pop_size:
            pop.append(random_genome(rng0) if rng0.random() < 0.3
                       else mutate(default_genome(), rng0, rate=0.8, scale=0.25))

    rng = random.Random(args.seed * 1000003 + start_gen)
    end_gen = args.gens if args.gens is not None else (10 ** 9 if args.loop else 40)
    history = open(os.path.join(RESULTS, "history.jsonl"), "a" if ckpt else "w")

    champs = []
    gen = start_gen - 1
    try:
        for gen in range(start_gen, end_gen + 1):
            idx = list(range(pop_size))
            rng.shuffle(idx)
            fits = [0.0] * pop_size
            champs = []
            for L in range(args.leagues):
                members = idx[L * 10:(L + 1) * 10]
                year = rng.choice(train_years)
                res = play_league([pop[i] for i in members], year, cache, rng)
                for m, r in zip(members, res):
                    fits[m] = fitness(r)
                    if r["champion"]:
                        champs.append(pop[m])
                        hall.append((fits[m], dict(pop[m])))
            hall.sort(key=lambda x: -x[0])
            hall = hall[:30]
            mean_f = sum(fits) / pop_size
            print(f"gen {gen:4d}  mean {mean_f:5.2f}  best {max(fits):5.2f}  "
                  f"champs kept {len(champs)}")
            history.write(json.dumps({"gen": gen, "mean": round(mean_f, 3),
                                      "best": round(max(fits), 3)}) + "\n")

            def parent():
                cand = rng.sample(range(pop_size), 3)
                return pop[max(cand, key=lambda i: fits[i])]

            nxt = [dict(c) for c in champs]
            while len(nxt) < pop_size:
                nxt.append(mutate(crossover(parent(), parent(), rng), rng))
            pop = nxt

            # ---- overfitting guard (loop mode) ----
            if args.loop and gen % args.check_every == 0:
                candidates = dedupe([g for _, g in hall[:6]] + champs[:4])
                scored = sorted(((eval_vs_autopick(g, guard_years, cache, args.guard_seeds, rng), g)
                                 for g in candidates), key=lambda x: -x[0])
                top_score, top_g = scored[0]
                if best_guard is None or top_score > best_guard["score"]:
                    best_guard = {"score": top_score, "genome": top_g, "gen": gen}
                    checks_since = 0
                    print(f"  guard check: {top_score:.2f}  ** new best **")
                else:
                    checks_since += 1
                    print(f"  guard check: {top_score:.2f}  (best {best_guard['score']:.2f} "
                          f"@gen {best_guard['gen']}, stale {checks_since}/{args.patience})")
                history.write(json.dumps({"gen": gen, "guard": round(top_score, 3),
                                          "guard_best": round(best_guard["score"], 3)}) + "\n")
                if args.patience > 0 and checks_since >= args.patience:
                    print(f"guard stalled for {args.patience} checks - stopping "
                          f"(further training would be memorizing {train_years})")
                    break
            save_ckpt(gen, pop, hall, best_guard, checks_since)
    except KeyboardInterrupt:
        print(f"\ninterrupted at gen {gen} - state saved, use --resume to continue")
    finally:
        history.close()
        if gen >= start_gen:
            save_ckpt(gen, pop, hall, best_guard, checks_since)

    # ---- champion selection ----
    if args.loop:
        # The guard decides when to STOP; it is too small (2 years) to SELECT with.
        # Select by re-evaluating candidates across ALL pre-holdout years - broad
        # selection is what generalizes (verified: guard-only selection scored at
        # the luck baseline on holdout, all-years selection well above it).
        candidates = dedupe([g for _, g in hall[:10]] + champs[:4] + [default_genome()]
                            + ([best_guard["genome"]] if best_guard else []))
        sel_years = train_years + guard_years
        print(f"\nfinal selection: {len(candidates)} candidates x {len(sel_years)} years "
              f"x {args.final_seeds} seeds vs 9 autopick bots")
        scored = []
        for i, g in enumerate(candidates):
            f = eval_vs_autopick(g, sel_years, cache, args.final_seeds, rng)
            print(f"  cand {i:2d}: mean fitness {f:5.2f}")
            scored.append((f, g))
        scored.sort(key=lambda x: -x[0])
        champion = {"genome": scored[0][1], "selected_by": "all pre-holdout years vs autopick",
                    "selection_fitness": round(scored[0][0], 3),
                    "guard_years": guard_years, "train_years": train_years,
                    "stopped_gen": gen, "seed": args.seed}
        print(f"\nchampion genome (selection fitness {scored[0][0]:.2f}) -> results/champion.json")
    else:
        candidates = dedupe([g for _, g in hall[:10]] + champs[:4] + [default_genome()])
        print(f"\nfinal selection: {len(candidates)} candidates x {len(train_years)} years "
              f"x {args.final_seeds} seeds vs 9 autopick bots")
        scored = []
        for i, g in enumerate(candidates):
            f = eval_vs_autopick(g, train_years, cache, args.final_seeds, rng)
            tag = " (hand-tuned default)" if all(abs(g[k] - default_genome()[k]) < 1e-9 for k in GENES) else ""
            print(f"  cand {i:2d}: mean fitness {f:5.2f}{tag}")
            scored.append((f, g))
        scored.sort(key=lambda x: -x[0])
        champion = {"genome": scored[0][1], "selected_by": "train years vs autopick",
                    "train_fitness": round(scored[0][0], 3), "train_years": train_years,
                    "gens": gen, "leagues": args.leagues, "seed": args.seed}
        print(f"\nchampion genome (train fitness {scored[0][0]:.2f}) -> results/champion.json")

    with open(os.path.join(RESULTS, "champion.json"), "w") as f:
        json.dump(champion, f, indent=1)
    dg = default_genome()
    for k in GENES:
        print(f"  {k:11s} {champion['genome'][k]:7.2f}   (default {dg[k]:.2f})")


if __name__ == "__main__":
    main()
