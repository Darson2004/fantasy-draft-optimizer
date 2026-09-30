"""Open-ended trade-policy evolution (the "Monopoly experiment") with
AlphaZero-style gating - v2: BIASED-OWNER ecosystem.

v1 result (archived in results_tradelab_v1_notrade_2026-08-10/): against
purely rational owners the lab converged to the no-trade theorem - with
symmetric information, every trade one side wants the other should refuse,
so the best evolved trader traded ~never. Profitable trading requires an
exploitable counterparty. v2 adds them.

Owner archetypes (one dial: weight w on observed season form vs preseason
expectation; the environment's own estimator is w=0.5):
  * RATIONAL (w=0.50): the control group - accepts only true upgrades, never
    proposes. Unexploitable by construction.
  * RECENCY  (w=0.85): overvalues last month's boomers, undervalues slumping
    studs - the sell-low/buy-high owner. Proposes trades it (wrongly) likes.
  * ANCHOR   (w=0.15): never gives up on a preseason name, never believes a
    late-round breakout. Also proposes.
Biased owners propose their best perceived-gain trade each week, so nets must
learn BOTH sides: hunting trades biased owners will accept, and accepting
(only) the exploitable incoming offers. Nets see proj and observed form as
separate features, so "target players whose price diverges from value" is
learnable - the owner mix routes each trade to whoever misprices it.

Everything else as v1:
  * every team drafts IDENTICALLY (deterministic autopick, shuffled slots) -
    only trading separates teams
  * weekly 1-for-1 trades, any player for any player (K/DST excluded, roster
    legality enforced), playoffs included, one executed trade per team-week
  * value net: 14 causal features -> tanh(8) -> value; propose X->Y when
    V(Y) > V(X), accept when V(in) > V(out) - no hand-coded trade strategy
  * training leagues: 6 evolving nets + 2 RECENCY + 1 ANCHOR + 1 RATIONAL;
    the best net of each league survives unchanged, rest bred by tournament
    crossover + Gaussian mutation
  * AlphaZero ratchet: archived INCUMBENT (starts as the zero net = never
    trades); every --gate-every gens the best screened candidate must beat it
    by --gate-margin in paired leagues (+8 mixed owners) to dethrone it.
    Lineage of accepted gates = the Elo curve. trade_champion.json is always
    the incumbent. First click now means: trading biased owners beats not
    trading at all.
  * bench each cycle: incumbent vs 3 RATIONAL + 3 RECENCY + 3 ANCHOR (luck
    line: 10% titles, 6.5 wins). Holdout years (2021+) never touched.

Usage:  python trade_lab.py                        # run until Ctrl-C
        python trade_lab.py --resume               # continue from checkpoint
        python trade_lab.py --gens 3 --leagues 2   # smoke test
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TEAMS, ROUNDS, POS_CAP, STARTERS, FLEX_POS
from env import (available_years, load_season, knowledge_view, run_season,
                 exp_ppw, _observed, _cap_ok)
from policy import autopick_id, picks_left_at

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_tradelab")
CKPT = os.path.join(RESULTS, "population.json")
INCUMBENT = os.path.join(RESULTS, "incumbent.json")
HOLDOUT_FROM = 2021
TITLE_BONUS = 3.0

POS_IDX = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
MIN_KEEP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}   # can't trade below starter needs
DEMAND = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}     # baseline rank per position
EPS = 1e-6

# owner archetypes: w = weight on observed form (0.5 = the env's estimator).
# RATIONAL judges trades by lineup improvement (sophisticated, unexploitable);
# biased owners judge by raw player value under their lens, in ppw units.
RATIONAL = {"w": 0.50, "accept": 0.5, "propose": None}
RECENCY = {"w": 0.85, "accept": 0.3, "propose": 0.6}
ANCHOR = {"w": 0.15, "accept": 0.3, "propose": 0.6}
TRAIN_OWNERS = [RECENCY, RECENCY, ANCHOR, RATIONAL]   # per training league
NETS_PER_LEAGUE = TEAMS - len(TRAIN_OWNERS)           # 6
BENCH_OWNERS = [RATIONAL] * 3 + [RECENCY] * 3 + [ANCHOR] * 3
GATE_OWNERS = [RATIONAL] * 2 + [RECENCY] * 3 + [ANCHOR] * 3

F, H = 14, 8               # features, hidden units
NW = F * H + H + H + 1     # 129 weights


def is_net(b):
    return isinstance(b, list)


# ------------------------------------------------------------------ the net
def new_net(rng, scale=0.5):
    return [rng.gauss(0.0, scale) for _ in range(NW)]


def zero_net():
    return [0.0] * NW


def net_key(w):
    return tuple(round(v, 4) for v in w)


def forward(w, x):
    out = w[-1]
    for h in range(H):
        s = w[F * H + h]
        base = h * F
        for i in range(F):
            s += w[base + i] * x[i]
        out += w[F * H + H + h] * math.tanh(s)
    return out


def mutate_net(w, rng, rate=0.35, scale=0.15):
    return [min(4.0, max(-4.0, v + rng.gauss(0.0, scale))) if rng.random() < rate else v
            for v in w]


def crossover_net(a, b, rng):
    return [a[i] if rng.random() < 0.5 else b[i] for i in range(NW)]


# ------------------------------------------------------------------ features
def global_feats(p, week, seen):
    e = exp_ppw(p, seen)
    ob = _observed(seen, p["id"])
    s = seen.get(p["id"])
    f = [e / 10.0,
         p["proj"] / 160.0,
         (ob[0] / 10.0) if ob else 0.0,
         (ob[1] / 5.0) if ob else 0.0,
         (s[1] / max(week - 1, 1)) if s else 0.0,
         1.0 if p["bye"] >= week else 0.0,
         0.0, 0.0, 0.0, 0.0]
    f[6 + POS_IDX[p["pos"]]] = 1.0
    return f


def team_ctx(roster, wins_t, week, seen):
    by_pos = {}
    for p in roster:
        if p["pos"] in POS_IDX:
            by_pos.setdefault(p["pos"], []).append(exp_ppw(p, seen))
    base, counts = {}, {}
    for pos in POS_IDX:
        vals = sorted(by_pos.get(pos, []), reverse=True)
        counts[pos] = len(vals)
        k = DEMAND[pos]
        base[pos] = vals[k - 1] if len(vals) >= k else 0.0
    pace = (wins_t - (week - 1) / 2.0) / 5.0
    return {"base": base, "counts": counts, "pace": pace, "wk": week / 16.0}


def ctx_feats(ctx, gf, pos):
    return [ctx["counts"][pos] / 4.0,
            (gf[0] * 10.0 - ctx["base"][pos]) / 10.0,
            ctx["pace"], ctx["wk"]]


# ------------------------------------------------------------------ owner lens
def biased_ppw(p, seen, w):
    """An owner's per-week valuation: w on observed season form, 1-w on the
    preseason curve. w=0.5 reproduces the environment's own exp_ppw."""
    pre = p["proj"] / 16.0
    ob = _observed(seen, p["id"])
    return (1.0 - w) * pre + w * ob[0] if ob else pre


def biased_lineup_mu(roster, seen, w):
    """Typical-week starting lineup total under an owner's biased lens.
    Returns (mu, started ids)."""
    ranked = sorted(roster, key=lambda p: -biased_ppw(p, seen, w))
    need, mu, started = dict(STARTERS), 0.0, set()
    for p in ranked:
        pos = p["pos"]
        if need.get(pos, 0) > 0:
            need[pos] -= 1
        elif pos in FLEX_POS and need.get("FLEX", 0) > 0:
            need["FLEX"] -= 1
        else:
            continue
        mu += biased_ppw(p, seen, w)
        started.add(p["id"])
    return mu, started


# ------------------------------------------------------------------ trade engine
def legal(roster_give, roster_get, X, Y):
    """1-for-1 legality: caps respected, no roster drops below starter needs."""
    if X["pos"] != Y["pos"]:
        if not _cap_ok(roster_give, Y["pos"], X["pos"]):
            return False
        n_left = sum(1 for p in roster_give if p["pos"] == X["pos"]) - 1
        if n_left < MIN_KEEP[X["pos"]]:
            return False
    return True


def make_trade_fn(brains, counter):
    """brains: per-team net weights (list) or an owner archetype dict.
    counter: [all executed trades, trades involving team 0]."""

    def trade_fn(rosters, week, seen, wins, rng):
        n = len(rosters)
        gf = {}
        for r in rosters:
            for p in r:
                if p["pos"] in POS_IDX and p["id"] not in gf:
                    gf[p["id"]] = global_feats(p, week, seen)
        ctxs = [team_ctx(rosters[t], wins[t], week, seen) for t in range(n)]

        val_cache = [dict() for _ in range(n)]   # per-team net valuations, per week

        def value(t, p):
            v = val_cache[t].get(p["id"])
            if v is None:
                g = gf[p["id"]]
                v = val_cache[t][p["id"]] = forward(brains[t], g + ctx_feats(ctxs[t], g, p["pos"]))
            return v

        ppw_cache = {}                        # (player id, lens) -> value, per week

        def bp(p, w):
            k = (p["id"], w)
            v = ppw_cache.get(k)
            if v is None:
                v = ppw_cache[k] = biased_ppw(p, seen, w)
            return v

        mu_cache = {}                         # team -> current biased lineup total

        def owner_gain(j, incoming, outgoing):
            w = brains[j]["w"]
            if j not in mu_cache:
                mu_cache[j] = biased_lineup_mu(rosters[j], seen, w)[0]
            rj = [p for p in rosters[j] if p["id"] != outgoing["id"]] + [incoming]
            return biased_lineup_mu(rj, seen, w)[0] - mu_cache[j]

        def accepts(j, incoming, outgoing):
            if is_net(brains[j]):
                return value(j, incoming) - value(j, outgoing) > EPS
            if brains[j]["propose"] is None:   # rational: strict lineup-improvement rule
                return owner_gain(j, incoming, outgoing) > brains[j]["accept"]
            # biased owners think in raw player value ("am I getting the better
            # player?") under their distorted lens - the exploitable behavior
            return bp(incoming, brains[j]["w"]) - bp(outgoing, brains[j]["w"]) > brains[j]["accept"]

        def execute(i, j, X, Y):
            rosters[i][:] = [p for p in rosters[i] if p["id"] != X["id"]] + [Y]
            rosters[j][:] = [p for p in rosters[j] if p["id"] != Y["id"]] + [X]
            mu_cache.pop(i, None)
            mu_cache.pop(j, None)
            val_cache[i].clear()
            val_cache[j].clear()
            traded.update((i, j))
            counter[0] += 1
            if 0 in (i, j):
                counter[1] += 1

        def negotiate(offers):
            """Try offers best-first until someone accepts. Biased-owner checks
            are ~free; rational lineup checks are capped - they almost never
            accept and each costs a lineup rebuild."""
            rational_tries = 0
            for gain, X, Y, j in offers:
                if j in traded:
                    continue
                if not is_net(brains[j]) and brains[j]["propose"] is None:
                    if rational_tries >= 10:
                        continue
                    rational_tries += 1
                if accepts(j, X, Y):
                    return (j, X, Y)
            return None

        traded = set()
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if i in traded:
                continue
            if is_net(brains[i]):
                mine = [(value(i, p), p) for p in rosters[i] if p["pos"] in POS_IDX]
                offers = []                   # (gain_i, X, Y, j)
                for j in range(n):
                    if j == i or j in traded:
                        continue
                    theirs = [(value(i, p), p) for p in rosters[j] if p["pos"] in POS_IDX]
                    pairs = []
                    for vx, X in mine:
                        for vy, Y in theirs:
                            if vy - vx <= EPS:
                                continue
                            if not (legal(rosters[i], rosters[j], X, Y)
                                    and legal(rosters[j], rosters[i], Y, X)):
                                continue
                            pairs.append((vy - vx, X, Y))
                    offers.extend((g, X, Y, j) for g, X, Y in pairs)
                offers.sort(key=lambda o: -o[0])
                deal = negotiate(offers)
                if deal:
                    execute(i, deal[0], deal[1], deal[2])
            elif brains[i]["propose"] is not None:
                # biased owners shop by raw player value under their lens -
                # any of their players (panic-selling a slumping starter is
                # exactly the behavior) for any player they perceive as better
                w = brains[i]["w"]
                mine = [(bp(p, w), p) for p in rosters[i] if p["pos"] in POS_IDX]
                offers = []                   # (perceived gain, X, Y, j)
                for j in range(n):
                    if j == i or j in traded:
                        continue
                    for Y in rosters[j]:
                        if Y["pos"] not in POS_IDX:
                            continue
                        vy = bp(Y, w)
                        for vx, X in mine:
                            if vy - vx <= brains[i]["propose"]:
                                continue
                            if not (legal(rosters[i], rosters[j], X, Y)
                                    and legal(rosters[j], rosters[i], Y, X)):
                                continue
                            offers.append((vy - vx, X, Y, j))
                offers.sort(key=lambda o: -o[0])
                deal = negotiate(offers)
                if deal:
                    execute(i, deal[0], deal[1], deal[2])

    return trade_fn


# ------------------------------------------------------------------ leagues
def draft_slot_rosters(view):
    """Everyone drafts with the same deterministic autopick: roster per SLOT."""
    rosters = [[] for _ in range(TEAMS)]
    taken = set()
    for pick in range(1, TEAMS * ROUNDS + 1):
        rnd, idx = divmod(pick - 1, TEAMS)
        slot = idx if rnd % 2 == 0 else TEAMS - 1 - idx
        pos_list = [p["pos"] for p in rosters[slot]]
        r = autopick_id(view, taken, pos_list, picks_left_at(slot, pick))
        taken.add(r["id"])
        rosters[slot].append(r)
    return rosters


class Cache:
    def __init__(self):
        self.slot_rosters, self.season = {}, {}

    def get(self, year):
        if year not in self.season:
            prior = [y for y in available_years() if y < year]
            self.slot_rosters[year] = draft_slot_rosters(knowledge_view(year, prior))
            self.season[year] = load_season(year)
        return self.slot_rosters[year], self.season[year]


def fitness(res):
    return res["wins"] + TITLE_BONUS * res["champion"] + res["points"] / 1000.0


def play_league(brains, year, cache, rng):
    slot_rosters, season = cache.get(year)
    perm = list(range(TEAMS))
    rng.shuffle(perm)                          # participant t sits at slot perm[t]
    rosters = [list(slot_rosters[perm[t]]) for t in range(TEAMS)]
    counter = [0, 0]
    res = run_season(rosters, season, rng, pool=None,
                     trade_fn=make_trade_fn(brains, counter))
    return res, counter


def quick_bench(net, years, cache, n_leagues, rng):
    """Net vs the mixed-owner league: the external, absolute skill measure."""
    fit = wins = titles = my_trades = 0
    for k in range(n_leagues):
        r2 = random.Random(rng.random())
        res, counter = play_league([net] + [dict(o) for o in BENCH_OWNERS],
                                   years[k % len(years)], cache, r2)
        fit += fitness(res[0])
        wins += res[0]["wins"]
        titles += res[0]["champion"]
        my_trades += counter[1]
    return {"fitness": round(fit / n_leagues, 3), "wins": round(wins / n_leagues, 2),
            "title_pct": round(100.0 * titles / n_leagues, 1),
            "my_trades_per_season": round(my_trades / n_leagues, 2), "leagues": n_leagues}


def gate_match(cand, incum, years, cache, n_leagues, rng):
    """Paired head-to-head: candidate and incumbent in the SAME mixed-owner
    league. Returns (mean fitness margin, its standard error)."""
    diffs = []
    for k in range(n_leagues):
        r2 = random.Random(rng.random())
        res, _ = play_league([cand, incum] + [dict(o) for o in GATE_OWNERS],
                             years[k % len(years)], cache, r2)
        diffs.append(fitness(res[0]) - fitness(res[1]))
    m = sum(diffs) / len(diffs)
    var = sum((d - m) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
    return m, (var / len(diffs)) ** 0.5


# ------------------------------------------------------------------ the ratchet
def load_incumbent():
    if os.path.exists(INCUMBENT):
        with open(INCUMBENT) as f:
            return json.load(f)
    return {"net": zero_net(), "since_gen": 0, "arch": {"F": F, "H": H, "NW": NW},
            "lineage": [{"gen": 0, "note": "zero baseline (never trades)"}]}


def save_incumbent(inc):
    os.makedirs(RESULTS, exist_ok=True)
    with open(INCUMBENT, "w") as f:
        json.dump(inc, f)


def gate_cycle(gen, hall, champs, incumbent, years, cache, rng, args, history):
    """Screen candidates -> paired gate vs incumbent -> ratchet + bench."""
    ik = net_key(incumbent["net"])
    cands, seen_k = [], {ik}
    for w in [g for _, g in hall[:6]] + champs[:2]:
        k = net_key(w)
        if k not in seen_k:
            seen_k.add(k)
            cands.append(list(w))
    accepted, margin, se = False, None, None
    if cands:
        screened = [(quick_bench(c, years, cache, args.screen_leagues, rng)["fitness"], c)
                    for c in cands]
        screened.sort(key=lambda x: -x[0])
        best = screened[0][1]
        margin, se = gate_match(best, incumbent["net"], years, cache, args.gate_leagues, rng)
        accepted = margin > args.gate_margin
        if accepted:
            incumbent = {"net": best, "since_gen": gen, "arch": incumbent["arch"],
                         "lineage": incumbent["lineage"] + [
                             {"gen": gen, "margin": round(margin, 3), "se": round(se, 3)}]}
    bench = quick_bench(incumbent["net"], years, cache, args.bench_leagues, rng)
    if accepted:
        incumbent["lineage"][-1]["bench_title_pct"] = bench["title_pct"]
    save_incumbent(incumbent)
    tag = "** RATCHET: new incumbent **" if accepted else "held"
    if margin is not None:
        print(f"  gate: margin {margin:+.3f} (SE {se:.3f}, need >{args.gate_margin}) -> {tag}")
    print(f"  incumbent bench vs mixed owners: {bench['title_pct']}% titles, "
          f"{bench['wins']} wins, {bench['my_trades_per_season']} trades/season "
          f"(luck: 10%, 6.5) | lineage clicks: {len(incumbent['lineage']) - 1}")
    history.write(json.dumps({"gen": gen, "gate": {
        "margin": None if margin is None else round(margin, 3),
        "se": None if se is None else round(se, 3),
        "accepted": accepted}, "bench": bench}) + "\n")
    history.flush()
    return incumbent


# ------------------------------------------------------------------ evolution
def save_ckpt(gen, pop, hall):
    os.makedirs(RESULTS, exist_ok=True)
    with open(CKPT, "w") as f:
        json.dump({"gen": gen, "pop": pop, "hall": hall}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=None, help="default: run until Ctrl-C")
    ap.add_argument("--leagues", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gate-every", type=int, default=50, help="gens between gate cycles")
    ap.add_argument("--gate-leagues", type=int, default=150,
                    help="paired head-to-head leagues per gate match")
    ap.add_argument("--gate-margin", type=float, default=0.35,
                    help="mean paired fitness margin required to dethrone the incumbent")
    ap.add_argument("--screen-leagues", type=int, default=8,
                    help="quick-bench leagues per candidate in the pre-gate screen")
    ap.add_argument("--bench-leagues", type=int, default=30,
                    help="leagues for the incumbent's absolute bench each cycle")
    ap.add_argument("--inject", type=int, default=5,
                    help="incumbent mutants seeded into the population after each gate")
    ap.add_argument("--reset-incumbent", action="store_true",
                    help="wipe the archived incumbent back to the zero (no-trade) baseline")
    args = ap.parse_args()

    years = [y for y in available_years() if y < HOLDOUT_FROM]
    cache = Cache()
    pop_size = args.leagues * NETS_PER_LEAGUE
    os.makedirs(RESULTS, exist_ok=True)
    if args.reset_incumbent and os.path.exists(INCUMBENT):
        os.remove(INCUMBENT)
    incumbent = load_incumbent()
    print(f"incumbent: {len(incumbent['lineage']) - 1} ratchet clicks, "
          f"since gen {incumbent['since_gen']}")
    print(f"training leagues: {NETS_PER_LEAGUE} nets + {len(TRAIN_OWNERS)} owners "
          f"(2 recency, 1 anchor, 1 rational)")

    start_gen, hall = 1, []
    ckpt = None
    if args.resume and os.path.exists(CKPT):
        with open(CKPT) as f:
            ckpt = json.load(f)
        pop = ckpt["pop"]
        while len(pop) < pop_size:
            pop.append(mutate_net(pop[len(pop) % len(ckpt["pop"])], random.Random(len(pop))))
        pop = pop[:pop_size]
        hall = ckpt["hall"]
        start_gen = ckpt["gen"] + 1
        print(f"resumed: gen {ckpt['gen']}, hall {len(hall)}")
    else:
        rng0 = random.Random(args.seed)
        pop = [new_net(rng0, scale=0.3) for _ in range(pop_size)]

    rng = random.Random(args.seed * 1000003 + start_gen)
    end_gen = args.gens if args.gens is not None else 10 ** 9
    history = open(os.path.join(RESULTS, "history.jsonl"), "a" if ckpt else "w")

    champs, gen = [], start_gen - 1
    try:
        for gen in range(start_gen, end_gen + 1):
            idx = list(range(pop_size))
            rng.shuffle(idx)
            fits = [0.0] * pop_size
            champs, trades_tot = [], 0
            for L in range(args.leagues):
                members = idx[L * NETS_PER_LEAGUE:(L + 1) * NETS_PER_LEAGUE]
                year = rng.choice(years)
                brains = [pop[m] for m in members] + [dict(o) for o in TRAIN_OWNERS]
                res, counter = play_league(brains, year, cache, rng)
                trades_tot += counter[0]
                best_m, best_f = None, None
                for m, r in zip(members, res[:NETS_PER_LEAGUE]):
                    fits[m] = fitness(r)
                    if best_f is None or fits[m] > best_f:
                        best_m, best_f = m, fits[m]
                champs.append(pop[best_m])         # best NET moves on
                hall.append([best_f, list(pop[best_m])])
            hall.sort(key=lambda x: -x[0])
            hall = hall[:30]
            mean_f = sum(fits[m] for m in idx) / pop_size
            print(f"gen {gen:6d}  net mean {mean_f:5.2f}  best {max(fits):5.2f}  "
                  f"trades/league {trades_tot / args.leagues:5.1f}")
            history.write(json.dumps({"gen": gen, "mean": round(mean_f, 3),
                                      "best": round(max(fits), 3),
                                      "trades_per_league": round(trades_tot / args.leagues, 2)}) + "\n")

            def parent():
                cand = rng.sample(range(pop_size), 3)
                return pop[max(cand, key=lambda i: fits[i])]

            nxt = [list(c) for c in champs]
            while len(nxt) < pop_size:
                nxt.append(mutate_net(crossover_net(parent(), parent(), rng), rng))
            pop = nxt

            if gen % args.gate_every == 0:
                incumbent = gate_cycle(gen, hall, champs, incumbent, years, cache,
                                       rng, args, history)
                for k in range(min(args.inject, pop_size)):   # anchor exploration
                    pop[-(k + 1)] = mutate_net(incumbent["net"], rng)
            save_ckpt(gen, pop, hall)
    except KeyboardInterrupt:
        print(f"\ninterrupted at gen {gen} - state saved, --resume continues")
    finally:
        history.close()
        if gen >= start_gen:
            save_ckpt(gen, pop, hall)

    # final gate so late progress isn't lost, then ship the incumbent
    if gen >= start_gen and hall:
        with open(os.path.join(RESULTS, "history.jsonl"), "a") as history:
            incumbent = gate_cycle(gen, hall, champs, incumbent, years, cache,
                                   rng, args, history)
    bench = quick_bench(incumbent["net"], years, cache, args.bench_leagues, rng)
    with open(os.path.join(RESULTS, "trade_champion.json"), "w") as f:
        json.dump({"net": incumbent["net"], "bench": bench,
                   "lineage": incumbent["lineage"], "stopped_gen": gen,
                   "seed": args.seed, "arch": incumbent["arch"]}, f, indent=1)
    print(f"\nincumbent -> results_tradelab/trade_champion.json "
          f"({bench['title_pct']}% titles vs mixed owners, "
          f"{len(incumbent['lineage']) - 1} ratchet clicks)")


if __name__ == "__main__":
    main()
