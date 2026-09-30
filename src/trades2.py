"""Season-simulation trade evaluator (in-season). Successor to trades.py.

What trades.py got wrong in-season, and what this fixes
  1. Projections: preseason season-total / 17 flat per week  ->  ESPN's own PER-WEEK
     projections for every remaining week (matchup-, bye- and injury-return-aware),
     built by inseason_proj.py from the public player feed.
  2. Schedule: field-average opponent every week  ->  the actual 14-week schedule plus
     the standings already banked (record and points-for after played weeks).
  3. Playoffs: 4 seeds, random pairings, no playoff weeks  ->  the league's real
     6-team bracket (top-2 byes) played on weeks 15-17 projections.
  4. Depth: bench worth nothing (only byes)  ->  injury sampling (position hazards,
     geometric spell length) + persistent per-player projection error, so a 2-for-1
     is charged for the bench it thins and credited for the star it lands.
  5. Unbalanced trades: ignored  ->  the side receiving fewer bodies fills the open
     spot with the best free agent, the side receiving more drops its least useful
     player, both chosen to maximise that roster's remaining lineup points.
  6. Search: <=4 bodies total  ->  every shape from 1-for-1 to 4-for-4 (exhaustive up
     to 4 bodies, top-N pools beyond), three-stage funnel:
        stage 1  deterministic expected wins on the real schedule   (all packages)
        stage 2  vectorised season Monte Carlo, fixed lineups        (top 300)
        stage 3  injury + projection-error Monte Carlo, CRN-paired   (top 40)
  Objective reported per side: dE[wins], dP(playoffs), dP(bye), dP(title).
  Acceptability: counterparty's own dE[wins] and FantasyCalc crowd-value optics.

  python trades2.py                    # roster audit + full trade search (~5-8 min)
  python trades2.py --quick            # stage 1+2 only, smaller pools
  python trades2.py --grade "Player A,Player B" "Player C"
  python trades2.py --opp t4           # restrict search to one opponent
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np

from trades import market_values, fc_value

ME = "me"
STATE = "league_state.json"
PROJ = "weekly_proj_2026.json"
SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "D/ST": 1, "K": 1}
FLEX_POS = ("RB", "WR", "TE")
SKILL = ("QB", "RB", "WR", "TE")
LAST_WEEK = 17
TEAM_SD = 25.0                     # weekly team-score sd (week-1 field sd was 23)
# per-week probability a healthy player picks up an injury that costs games, by pos
HAZARD = {"QB": 0.020, "RB": 0.045, "WR": 0.035, "TE": 0.035, "K": 0.005, "D/ST": 0.0}
SPELL_P = 1 / 3                    # geometric spell length, mean 3 weeks
SEASON_END_P = 0.15                # share of injuries that end the season
PROJ_ERR_SD = {"QB": 0.15, "RB": 0.25, "WR": 0.22, "TE": 0.25, "K": 0.10, "D/ST": 0.10}
Q_OUT = {"QUESTIONABLE": 0.25, "DOUBTFUL": 0.75, "OUT": 1.0}
# Calibration slopes of multi-week projections vs realised points (Fantasy Football
# Analytics, 12-season study, Aug 2026): projected gaps between players are too wide,
# so shrink each player's weekly projection toward the starter-pool mean at his
# position before valuing a trade. --no-shrink turns this off.
CALIB_SLOPE = {"QB": 0.67, "TE": 0.72, "RB": 0.79, "WR": 0.85, "K": 1.0, "D/ST": 1.0}
STARTER_POOL = {"QB": 10, "RB": 25, "WR": 25, "TE": 10}


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ------------------------------------------------------------------ data
def shrink(players, rosters, cur):
    """Regress weekly projections toward the mean of the top-N rostered players at each
    position (per week), keeping zeros (bye/out) at zero."""
    for pos, n in STARTER_POOL.items():
        pool = sorted([p for rows in rosters.values() for p in rows if p["pos"] == pos],
                      key=lambda p: -ros(p, cur + 1))[:n]
        if not pool:
            continue
        b = CALIB_SLOPE[pos]
        for w in range(cur + 1, LAST_WEEK + 1):
            vals = [p["wk"][w] for p in pool if p["wk"][w] > 0]
            if not vals:
                continue
            m = sum(vals) / len(vals)
            for p in players.values():
                if p["pos"] == pos and p["wk"][w] > 0:
                    p["wk"][w] = max(0.0, m + b * (p["wk"][w] - m))


def load(do_shrink=True, proj_file=PROJ):
    st = json.load(open(STATE))
    cur = st["current_week"]
    players = {}
    for p in json.load(open(proj_file))["players"]:
        wk = np.zeros(LAST_WEEK + 1)
        for w, v in p["wk"].items():
            wk[int(w)] = v
        players[p["id"]] = {"id": p["id"], "name": p["name"], "pos": p["pos"],
                            "team": p["team"], "inj": p["injury"], "wk": wk,
                            "own": p["pct_owned"]}
    # manual overrides, but only fresh ones - the feed catches up within days
    try:
        ov = json.load(open("injury_overrides.json"))
        by_name = {p["name"]: p for p in players.values()}
        for name, o in ov.items():
            if name.startswith("_") or name not in by_name:
                continue
            age = (time.time() - time.mktime(time.strptime(o["date"], "%Y-%m-%d"))) / 86400
            if age <= 14:
                by_name[name]["inj"] = o["status"].upper()
    except FileNotFoundError:
        pass
    # current-week availability tag -> expected value this week (ESPN zeroes OUT/IR
    # weeks itself but does not haircut QUESTIONABLE)
    for p in players.values():
        q = Q_OUT.get(p["inj"])
        if q:
            p["wk"][cur] *= (1 - q)
    rosters = {t: [players[i] for i in ids] for t, ids in st["rosters"].items()}
    teams = list(rosters)
    if do_shrink:
        shrink(players, rosters, cur)
    rostered = {p["id"] for rows in rosters.values() for p in rows}
    fa = [p for p in players.values() if p["id"] not in rostered and p["own"] > 0.5]
    return st, players, rosters, teams, fa


def ros(p, frm, to=LAST_WEEK):
    return float(p["wk"][frm:to + 1].sum())


# ------------------------------------------------------------------ lineups
def lineup(rows, w, mult=None, fa_best=None):
    """Best legal starting lineup points for week w. mult: optional {id: factor}
    (injury/projection-error scenario). Empty QB/K/D-ST slots stream the best free
    agent (fa_best[pos][w]); empty skill slots score 0."""
    need, flex, pts = dict(SLOTS), 1, 0.0
    if mult is None:
        cands = sorted(rows, key=lambda p: -p["wk"][w])
        val = lambda p: p["wk"][w]
    else:
        def val(p):
            m = mult.get(p["id"])
            return p["wk"][w] * (m[w] if m is not None else 1.0)
        cands = sorted(rows, key=lambda p: -val(p))
    for p in cands:
        v = val(p)
        if v <= 0:
            break
        pos = p["pos"]
        if need.get(pos, 0) > 0:
            need[pos] -= 1; pts += v
        elif flex and pos in FLEX_POS:
            flex = 0; pts += v
    if fa_best:
        for pos in ("QB", "K", "D/ST"):
            if need[pos] > 0:
                pts += fa_best[pos][w]
    return pts


def fa_best_table(fa):
    out = {}
    for pos in ("QB", "K", "D/ST"):
        arr = np.zeros(LAST_WEEK + 1)
        for p in fa:
            if p["pos"] == pos:
                arr = np.maximum(arr, p["wk"])
        out[pos] = arr
    return out


def team_mus(rows, weeks, fa_best, mult=None):
    return np.array([lineup(rows, w, mult, fa_best) for w in weeks])


# ------------------------------------------------------------------ roster balancing
def balance(rows, n_target, fa_pool, weeks, fa_best, fast=False, protect=()):
    """Bring a post-trade roster to n_target bodies: fill holes with the free agent
    that adds the most remaining lineup points, drop the player whose removal costs
    the least. fast=True (stage-1 screen): fills try only the best FA at RB/WR/TE,
    drops take the lowest-RoS skill player not in `protect`. Returns (rows, adds, drops)."""
    rows, adds, drops = list(rows), [], []
    have = {p["id"] for p in rows}
    if fast:
        pool = list(fa_pool.values()) if isinstance(fa_pool, dict) else fa_pool
        while len(rows) < n_target:
            best, best_v = None, -1
            for c in pool:
                if c["id"] in have:
                    continue
                v = team_mus(rows + [c], weeks, fa_best).sum()
                if v > best_v:
                    best, best_v = c, v
            rows.append(best); have.add(best["id"]); adds.append(best)
        while len(rows) > n_target:
            worst = min((p for p in rows if p["pos"] in SKILL and p["id"] not in protect),
                        key=lambda p: ros(p, weeks[0]))
            rows = [p for p in rows if p is not worst]; drops.append(worst)
        return rows, adds, drops
    while len(rows) < n_target:
        best, best_v = None, -1
        for c in fa_pool:
            if c["id"] in have:
                continue
            v = team_mus(rows + [c], weeks, fa_best).sum()
            if v > best_v:
                best, best_v = c, v
        rows.append(best); have.add(best["id"]); adds.append(best)
    while len(rows) > n_target:
        base = None
        worst, worst_cost = None, None
        for c in rows:
            if c["pos"] not in SKILL:
                continue
            v = team_mus([p for p in rows if p is not c], weeks, fa_best).sum()
            if worst_cost is None or v > worst_cost:
                worst, worst_cost = c, v
        rows = [p for p in rows if p is not worst]; drops.append(worst)
    return rows, adds, drops


# ------------------------------------------------------------------ season simulation
class Season:
    """Vectorised Monte Carlo of the remaining season + 6-team bracket.
    mus: array (S, T, W) or (T, W) of weekly lineup means (index by week number)."""

    def __init__(self, st, teams, sims, seed=11):
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        self.cur = st["current_week"]
        self.reg = st["settings"]["reg_weeks"]
        self.sched = {int(w): [(self.idx[h], self.idx[a]) for h, a in ms]
                      for w, ms in st["schedule"].items()}
        rec = st["records"]
        self.w0 = np.array([rec[t]["w"] for t in teams], float)
        self.pf0 = np.array([rec[t]["pf"] for t in teams], float)
        rng = np.random.default_rng(seed)
        self.S = sims
        # common random numbers: one noise draw per (sim, team, week) reused for every
        # candidate trade so paired differences are low-variance
        self.noise = rng.normal(0, TEAM_SD, size=(sims, len(teams), LAST_WEEK + 1))

    def run(self, mus):
        S, T = self.S, len(self.teams)
        if mus.ndim == 2:
            mus = np.broadcast_to(mus, (S,) + mus.shape)
        wins = np.tile(self.w0, (S, 1))
        pf = np.tile(self.pf0, (S, 1))
        for w in range(self.cur, self.reg + 1):
            sc = mus[:, :, w] + self.noise[:, :, w]
            pf += sc
            for h, a in self.sched[w]:
                hw = sc[:, h] > sc[:, a]
                wins[:, h] += hw
                wins[:, a] += ~hw
        # seeding: record, then points-for
        order = np.lexsort((-pf, -wins), axis=1)          # (S, T) team indices, best first
        seeds = order[:, :6]
        playoff = np.zeros((S, T), bool)
        np.put_along_axis(playoff, seeds, True, axis=1)
        bye = np.zeros((S, T), bool)
        np.put_along_axis(bye, seeds[:, :2], True, axis=1)
        rows = np.arange(S)

        def game(a, b, w):
            sa = mus[rows, a, w] + self.noise[rows, a, w]
            sb = mus[rows, b, w] + self.noise[rows, b, w]
            return np.where(sa > sb, a, b)
        w15, w16, w17 = self.reg + 1, self.reg + 2, self.reg + 3
        q1 = game(seeds[:, 2], seeds[:, 5], w15)        # 3 v 6
        q2 = game(seeds[:, 3], seeds[:, 4], w15)        # 4 v 5
        s1 = game(seeds[:, 0], q2, w16)                 # 1 v (4/5)
        s2 = game(seeds[:, 1], q1, w16)                 # 2 v (3/6)
        champ = game(s1, s2, w17)
        title = np.zeros((S, T))
        np.add.at(title, (rows, champ), 1.0)
        return {"wins": wins.mean(0), "playoff": playoff.mean(0), "bye": bye.mean(0),
                "title": title.mean(0)}


def scenarios(players, teams_rows, sims, cur, seed=5):
    """Per-sim multipliers {id: array over weeks} for every rostered player:
    persistent projection error x injury availability."""
    rng = np.random.default_rng(seed)
    out = [dict() for _ in range(sims)]
    for rows in teams_rows.values():
        for p in rows:
            err = rng.lognormal(0, PROJ_ERR_SD[p["pos"]], size=sims)
            for s in range(sims):
                m = np.full(LAST_WEEK + 1, err[s])
                w = cur + 1
                while w <= LAST_WEEK:
                    if rng.random() < HAZARD[p["pos"]]:
                        if rng.random() < SEASON_END_P:
                            m[w:] = 0.0; break
                        L = rng.geometric(SPELL_P)
                        m[w:min(LAST_WEEK, w + L - 1) + 1] = 0.0
                        w += L
                    w += 1
                out[s][p["id"]] = m
    return out


# ------------------------------------------------------------------ evaluation
class Evaluator:
    def __init__(self, st, players, rosters, teams, fa, sims2=4000, sims3=300):
        self.st, self.players, self.rosters, self.teams, self.fa = st, players, rosters, teams, fa
        self.cur = st["current_week"]
        self.reg = st["settings"]["reg_weeks"]
        self.size = st["settings"]["roster_size"] + st["settings"]["ir_slots"]
        self.weeks = list(range(self.cur, LAST_WEEK + 1))
        self.trade_weeks = list(range(self.cur + 1, LAST_WEEK + 1))   # trade lands next week
        self.fa_best = fa_best_table(fa)
        self.fa_skill = []
        for pos, n in (("RB", 5), ("WR", 5), ("TE", 3), ("QB", 2)):
            self.fa_skill += sorted([p for p in fa if p["pos"] == pos],
                                    key=lambda p: -ros(p, self.cur + 1))[:n]
        self.fa_fast = {pos: max((p for p in self.fa_skill if p["pos"] == pos),
                                 key=lambda p: ros(p, self.cur + 1)) for pos in ("RB", "WR", "TE")}
        self.base_mus = {t: self._mus(rows) for t, rows in rosters.items()}
        self.season = Season(st, teams, sims2)
        self.sims3 = sims3
        self.scen = None
        self.base_det = self._det_wins(self.base_mus)
        self.base_mc = self.season.run(self._stack(self.base_mus))

    def _mus(self, rows):
        mus = np.zeros(LAST_WEEK + 1)
        for w in self.weeks:
            mus[w] = lineup(rows, w, None, self.fa_best)
        return mus

    def _stack(self, mus_by_team):
        return np.stack([mus_by_team[t] for t in self.teams])

    def _det_wins(self, mus):
        """Expected wins on the real schedule (deterministic lineups, Phi margins)."""
        out = {t: float(self.st["records"][t]["w"]) for t in self.teams}
        sd = TEAM_SD * math.sqrt(2)
        for w in range(self.cur, self.reg + 1):
            for h, a in self.st["schedule"][str(w)]:
                p = phi((mus[h][w] - mus[a][w]) / sd)
                out[h] += p; out[a] += 1 - p
        return out

    def apply(self, opp, give, get, fast=False):
        """Post-trade rosters for me and opp (balanced), week `cur` frozen on current
        rosters. Returns (mus dict, my_rows, opp_rows, notes)."""
        gid, rid = {p["id"] for p in give}, {p["id"] for p in get}
        mine = [p for p in self.rosters[ME] if p["id"] not in gid] + list(get)
        theirs = [p for p in self.rosters[opp] if p["id"] not in rid] + list(give)
        notes = []
        pool = self.fa_fast if fast else self.fa_skill
        mine, a1, d1 = balance(mine, len(self.rosters[ME]), pool, self.trade_weeks, self.fa_best, fast, rid)
        theirs, a2, d2 = balance(theirs, len(self.rosters[opp]), pool, self.trade_weeks, self.fa_best, fast, gid)
        if a1: notes.append("I add FA " + "+".join(p["name"] for p in a1))
        if d1: notes.append("I drop " + "+".join(p["name"] for p in d1))
        if a2: notes.append(f"{opp} adds FA " + "+".join(p["name"] for p in a2))
        if d2: notes.append(f"{opp} drops " + "+".join(p["name"] for p in d2))
        mus = dict(self.base_mus)
        for t, rows in ((ME, mine), (opp, theirs)):
            m = self.base_mus[t].copy()
            for w in self.trade_weeks:
                m[w] = lineup(rows, w, None, self.fa_best)
            mus[t] = m
        return mus, mine, theirs, notes

    def stage1(self, opp, give, get):
        mus, _, _, notes = self.apply(opp, give, get, fast=True)
        det = self._det_wins(mus)
        return det[ME] - self.base_det[ME], det[opp] - self.base_det[opp], notes

    def stage2(self, opp, give, get):
        mus, _, _, notes = self.apply(opp, give, get)
        r = self.season.run(self._stack(mus))
        i, j = self.teams.index(ME), self.teams.index(opp)
        b = self.base_mc
        return {k: (r[k][i] - b[k][i], r[k][j] - b[k][j]) for k in r}, notes

    # ---- stage 3: injury + projection-error scenarios, CRN-paired
    def _scen_mus(self, rows_by_team):
        S = len(self.scen)
        mus = np.zeros((S, len(self.teams), LAST_WEEK + 1))
        for ti, t in enumerate(self.teams):
            rows = rows_by_team[t]
            for s in range(S):
                m = self.scen[s]
                mus[s, ti, self.cur] = self.base_mus[t][self.cur]
                for w in self.trade_weeks:
                    mus[s, ti, w] = lineup(rows, w, m, self.fa_best)
        return mus

    def prepare_stage3(self):
        self.scen = scenarios(self.players, self.rosters, self.sims3, self.cur)
        for p in self.fa_skill:            # FA fills need scenario rows too
            rng = np.random.default_rng(p["id"] % 100000)
            for s in range(self.sims3):
                self.scen[s].setdefault(p["id"], np.full(LAST_WEEK + 1, rng.lognormal(0, PROJ_ERR_SD[p["pos"]])))
        self.season3 = Season(self.st, self.teams, self.sims3, seed=23)
        self.mus3_base = self._scen_mus(self.rosters)
        self.base3 = self.season3.run(self.mus3_base)

    def stage3(self, opp, give, get):
        _, mine, theirs, notes = self.apply(opp, give, get)
        mus = self.mus3_base.copy()
        S = self.sims3
        for t, rows in ((ME, mine), (opp, theirs)):
            ti = self.teams.index(t)
            for s in range(S):
                m = self.scen[s]
                for w in self.trade_weeks:
                    mus[s, ti, w] = lineup(rows, w, m, self.fa_best)
        r = self.season3.run(mus)
        i, j = self.teams.index(ME), self.teams.index(opp)
        b = self.base3
        return {k: (r[k][i] - b[k][i], r[k][j] - b[k][j]) for k in r}, notes


# ------------------------------------------------------------------ search
def packages(mine, theirs, cur, max_bodies=8, pool_big=9, pool_huge=8):
    """Yield (give, get) tuples. Exhaustive up to 4 bodies; larger shapes drawn from
    each side's top-N skill players by remaining projection."""
    key = lambda p: -ros(p, cur + 1)
    m_all = sorted(mine, key=key); t_all = sorted(theirs, key=key)
    for a in range(1, 5):
        for b in range(1, 5):
            n = a + b
            if n > max_bodies:
                continue
            if n <= 4:
                mp, tp = m_all, t_all
            elif n <= 6:
                mp, tp = m_all[:pool_big], t_all[:pool_big]
            else:
                mp, tp = m_all[:pool_huge], t_all[:pool_huge]
            for g in itertools.combinations(mp, a):
                for r in itertools.combinations(tp, b):
                    yield g, r


def fmt_names(rows):
    return " + ".join(p["name"] for p in rows)


def search(ev, opponents, quick=False, top=25, exclude=()):
    fc = market_values()
    cur = ev.cur
    mine = [p for p in ev.rosters[ME] if p["pos"] in SKILL]
    t0 = time.time()
    cands = []
    n = 0
    for opp in opponents:
        theirs = [p for p in ev.rosters[opp] if p["pos"] in SKILL and p["name"] not in exclude]
        for give, get in packages(mine, theirs, cur, max_bodies=6 if quick else 8,
                                  pool_big=8 if quick else 9, pool_huge=7 if quick else 8):
            n += 1
            dme, dthem, notes = ev.stage1(opp, give, get)
            if dme <= 0.03:
                continue
            calc = (sum(fc_value(fc, p["name"]) for p in give)
                    - sum(fc_value(fc, p["name"]) for p in get))
            gv = sum(fc_value(fc, p["name"]) for p in get)
            optics = calc / gv if gv else 0.0
            cands.append({"opp": opp, "give": give, "get": get, "d1": (dme, dthem),
                          "calc": calc, "optics": optics, "notes": notes})
    print(f"stage 1: {n:,} packages in {time.time() - t0:.0f}s -> {len(cands):,} improve me by >0.03 wins")

    # acceptability: they must not lose by their own lineup math, and the crowd value
    # they receive must be within 12% of what they give (what a calculator shows them)
    for c in cands:
        c["tier"] = ("MUTUAL" if c["d1"][1] > -0.02 and c["optics"] > -0.12 else
                     "OPTICS" if c["d1"][1] > -0.20 and c["optics"] > 0.08 else None)
    ok = [c for c in cands if c["tier"]]
    print(f"         {sum(c['tier'] == 'MUTUAL' for c in ok):,} MUTUAL (them dWins > -0.02 and optics > -12%), "
          f"{sum(c['tier'] == 'OPTICS' for c in ok):,} OPTICS (them dWins > -0.20 but calculator favours them by > 8%)")
    ok.sort(key=lambda c: -c["d1"][0])
    short = ok[:300]
    t1 = time.time()
    for c in short:
        c["d2"], _ = ev.stage2(c["opp"], c["give"], c["get"])
    print(f"stage 2: {len(short)} candidates through 4000-sim season MC in {time.time() - t1:.0f}s")
    short.sort(key=lambda c: -(c["d2"]["title"][0] + 0.5 * c["d2"]["playoff"][0]))
    json.dump([{"opp": c["opp"], "give": [p["name"] for p in c["give"]], "get": [p["name"] for p in c["get"]],
                "d1": c["d1"], "d2": {k: list(v) for k, v in c["d2"].items()}, "calc": c["calc"],
                "optics": c["optics"], "notes": c["notes"], "tier": c["tier"]} for c in short],
              open("trade_stage2_checkpoint.json", "w"), indent=0)
    if quick:
        return short[:top], None
    final = short[:40]
    t2 = time.time()
    ev.prepare_stage3()
    for c in final:
        c["d3"], _ = ev.stage3(c["opp"], c["give"], c["get"])
    print(f"stage 3: {len(final)} candidates through {ev.sims3}-scenario injury MC in {time.time() - t2:.0f}s")
    final.sort(key=lambda c: -(c["d3"]["title"][0] + 0.5 * c["d3"]["playoff"][0]))
    return short[:top], final


def show(rows, key, title):
    print(f"\n=== {title} ===")
    print(f"{'opp':4s} {'shape':6s} {'give':44s} {'get':44s} {'dW me':>6s} {'dPO me':>7s} {'dTitle':>7s} | "
          f"{'dW them':>7s} {'dPO them':>8s} {'calc':>6s}")
    for c in rows:
        d = c[key]
        shape = f"{len(c['give'])}-for-{len(c['get'])}"
        print(f"{c['opp']:4s} {shape:6s} {fmt_names(c['give'])[:44]:44s} {fmt_names(c['get'])[:44]:44s} "
              f"{d['wins'][0]:+6.2f} {d['playoff'][0]*100:+6.1f}% {d['title'][0]*100:+6.1f}% | "
              f"{d['wins'][1]:+7.2f} {d['playoff'][1]*100:+7.1f}% {c['calc']:+6.0f}  {c.get('tier', '')}")
        if c["notes"]:
            print(f"{'':11s} ({'; '.join(c['notes'])})")


def audit(ev):
    st, rosters = ev.st, ev.rosters
    cur = ev.cur
    print(f"week {cur} | remaining reg-season weeks {cur}-{ev.reg} | playoffs 15-17, 6 teams")
    print(f"\n{'team':4s} {'rec':5s} {'PF':>6s} {'avg mu':>7s} {'E[W]':>6s} {'P(PO)':>6s} {'P(bye)':>6s} {'P(title)':>8s}")
    b = ev.base_mc
    for i, t in enumerate(sorted(ev.teams, key=lambda t: -ev.base_mc["title"][ev.teams.index(t)])):
        j = ev.teams.index(t)
        r = st["records"][t]
        avg = ev.base_mus[t][cur:ev.reg + 1].mean()
        print(f"{t:4s} {r['w']}-{r['l']:<3d} {r['pf']:6.1f} {avg:7.1f} {b['wins'][j]:6.2f} "
              f"{b['playoff'][j]*100:5.0f}% {b['bye'][j]*100:5.0f}% {b['title'][j]*100:7.1f}%"
              f"{'   <- me' if t == ME else ''}")
    print(f"\n=== my roster: remaining projection and marginal value ===")
    print(f"{'player':22s} {'pos':4s} {'tag':13s} {'RoS':>6s} {'wk/g':>5s} {'starts':>6s} {'dW if gone':>10s}")
    rows = rosters[ME]
    weeks = ev.trade_weeks
    for p in sorted(rows, key=lambda p: -ros(p, cur + 1)):
        starts = 0
        for w in weeks:
            with_ = lineup(rows, w, None, ev.fa_best)
            without = lineup([q for q in rows if q is not p], w, None, ev.fa_best)
            starts += with_ > without + 1e-9
        rest, _, _ = balance([q for q in rows if q is not p], len(rows), ev.fa_skill, weeks, ev.fa_best)
        mus = dict(ev.base_mus); m = ev.base_mus[ME].copy()
        for w in weeks:
            m[w] = lineup(rest, w, None, ev.fa_best)
        mus[ME] = m
        dw = ev._det_wins(mus)[ME] - ev.base_det[ME]
        games = int((p["wk"][cur + 1:] > 0).sum())
        pg = ros(p, cur + 1) / games if games else 0
        print(f"{p['name'][:22]:22s} {p['pos']:4s} {p['inj'][:13]:13s} {ros(p, cur + 1):6.1f} {pg:5.1f} "
              f"{starts:6d} {dw:+10.2f}")
    print(f"\nbest free agents (RoS from week {cur + 1}): " +
          ", ".join(f"{p['name']} {p['pos']} {ros(p, cur + 1):.0f}" for p in ev.fa_skill[:8]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--opp", help="restrict to one opponent abbrev")
    ap.add_argument("--grade", nargs=2, metavar=("GIVE", "GET"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--no-shrink", action="store_true", help="skip calibration shrinkage")
    ap.add_argument("--out", default="trade_search_latest.json")
    ap.add_argument("--exclude", default="", help="comma-separated players the other side won't trade")
    ap.add_argument("--proj", default=PROJ, help="projection file (e.g. weekly_proj_2026_recal.json; use with --no-shrink)")
    args = ap.parse_args()

    st, players, rosters, teams, fa = load(do_shrink=not args.no_shrink, proj_file=args.proj)
    print(f"projections: {args.proj}{'' if args.no_shrink else ' + calibration shrinkage'}")
    ev = Evaluator(st, players, rosters, teams, fa)
    audit(ev)

    if args.grade:
        by_name = {p["name"].lower(): (t, p) for t, rows in rosters.items() for p in rows}

        def look(n):
            n = n.strip().lower()
            hits = [k for k in by_name if k == n] or [k for k in by_name if n in k]
            if len(hits) != 1:
                raise SystemExit(f"{n!r} matched {hits or 'nothing'}")
            return by_name[hits[0]]
        give = [look(n) for n in args.grade[0].split(",")]
        get = [look(n) for n in args.grade[1].split(",")]
        opp = {t for t, _ in get}.pop()
        g, r = [p for _, p in give], [p for _, p in get]
        d1 = ev.stage1(opp, g, r)
        d2, notes = ev.stage2(opp, g, r)
        ev.prepare_stage3()
        d3, _ = ev.stage3(opp, g, r)
        print(f"\nvs {opp}: give {fmt_names(g)} | get {fmt_names(r)}   {'; '.join(notes)}")
        print(f"  stage1 dWins        me {d1[0]:+.3f}  them {d1[1]:+.3f}")
        for k in ("wins", "playoff", "bye", "title"):
            print(f"  {k:8s} fixed-MC  me {d2[k][0]:+.3f}  them {d2[k][1]:+.3f}   |  injury-MC  me {d3[k][0]:+.3f}  them {d3[k][1]:+.3f}")
        fc = market_values()
        gv = sum(fc_value(fc, p["name"]) for p in g); tv = sum(fc_value(fc, p["name"]) for p in r)
        print(f"  FantasyCalc optics: they receive {gv:.0f} for {tv:.0f} ({gv - tv:+.0f})")
        return
    if args.no_search:
        return
    opponents = [args.opp] if args.opp else [t for t in teams if t != ME]
    excl = {n.strip() for n in args.exclude.split(",") if n.strip()}
    if excl:
        print(f"excluded from the other side: {sorted(excl)}")
    short, final = search(ev, opponents, args.quick, args.top, excl)
    show(short, "d2", f"top {len(short)} by season MC (fixed lineups) - ranked title + 0.5 playoff")
    if final:
        show(final, "d3", "top 40 re-scored under injury + projection-error scenarios (the number to trust)")
        json.dump([{"opp": c["opp"], "give": [p["name"] for p in c["give"]], "get": [p["name"] for p in c["get"]],
                    "d1": c["d1"], "d2": {k: list(v) for k, v in c["d2"].items()},
                    "d3": {k: list(v) for k, v in c["d3"].items()}, "calc": c["calc"], "notes": c["notes"], "tier": c["tier"]}
                   for c in final], open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
