"""In-season player-trade evaluator: grades trades on the CURRENT rosters.

Rosters come from live_rosters.json (ESPN playerIds, re-scraped after every
transaction) — not from drafted.txt, which is only the draft-day record.
Once rosters are fixed there is no draft to simulate, so the math is exact:
swap the players, rebuild each team's bye-aware best lineup, and diff expected
H2H wins (the same objective draftmodel.py drafted for).

  python trades.py                          # my roster + everyone's, by position
  python trades.py --scan t6                # every 1-for-1 / 2-for-2 with team t6
  python trades.py --scan all               # ...with all nine opponents
  python trades.py --grade "Player A,Player B" "Player C,Player D"
"""
import argparse
import itertools
import json
import math
import random

import draftmodel as dm

ME = "me"
PLAYOFF_SEEDS = 4
TEAM_SD = 35 / math.sqrt(2)     # per-team weekly sd consistent with SIGMA=35 margins


def market_values():
    """FantasyCalc redraft crowd values — the number COUNTERPARTIES judge offers by.
    Auto-refreshes when older than 3 days; falls back to the cached file."""
    import os, time, urllib.request
    path = "fantasycalc_values.json"
    try:
        stale = time.time() - os.path.getmtime(path) > 72 * 3600
    except OSError:
        stale = True
    if stale:
        try:
            url = ("https://api.fantasycalc.com/values/current"
                   "?isDynasty=false&numQbs=1&numTeams=10&ppr=1")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            json.dump({p["player"]["name"]: {"value": p["value"],
                                             "rank": p["overallRank"],
                                             "pos": p["player"]["position"]}
                       for p in data}, open(path, "w"), indent=0)
        except Exception as e:
            print(f"(market value refresh failed: {e}; using cached)")
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return {}


def fc_value(fc, name):
    if name in fc:
        return fc[name]["value"]
    short = name.replace(" III", "").replace(" Jr.", "").replace(" Sr.", "")
    for k in fc:
        ks = k.split()
        ss = short.split()
        if ks and ss and ks[0] == ss[0] and ks[-1] == ss[-1]:
            return fc[k]["value"]
    return 0


def mc_odds(mus, n_sims=6000, seed=7):
    """Monte Carlo season: random weekly pairings, top-4 make playoffs (wins, then
    points), 1v4/2v3 semis + final at full (no-bye) strength. Returns per team
    (expW, P(playoff), P(title))."""
    rng = random.Random(seed)
    ids = list(mus)
    playoffs = {t: 0 for t in ids}
    titles = {t: 0 for t in ids}
    wins_acc = {t: 0.0 for t in ids}
    full = {t: max(mus[t]) for t in ids}
    for _ in range(n_sims):
        wins = {t: 0 for t in ids}
        pf = {t: 0.0 for t in ids}
        for w in range(dm.REG_WEEKS):
            order = ids[:]
            rng.shuffle(order)
            for i in range(0, len(ids), 2):
                a, b = order[i], order[i + 1]
                sa = mus[a][w] + rng.gauss(0, TEAM_SD)
                sb = mus[b][w] + rng.gauss(0, TEAM_SD)
                pf[a] += sa; pf[b] += sb
                wins[b if sb > sa else a] += 1
        seeds = sorted(ids, key=lambda t: (-wins[t], -pf[t]))[:PLAYOFF_SEEDS]
        for t in seeds:
            playoffs[t] += 1
        s1 = seeds[0] if full[seeds[0]] + rng.gauss(0, TEAM_SD) > full[seeds[3]] + rng.gauss(0, TEAM_SD) else seeds[3]
        s2 = seeds[1] if full[seeds[1]] + rng.gauss(0, TEAM_SD) > full[seeds[2]] + rng.gauss(0, TEAM_SD) else seeds[2]
        champ = s1 if full[s1] + rng.gauss(0, TEAM_SD) > full[s2] + rng.gauss(0, TEAM_SD) else s2
        titles[champ] += 1
        for t in ids:
            wins_acc[t] += wins[t]
    return {t: (wins_acc[t] / n_sims, playoffs[t] / n_sims, titles[t] / n_sims) for t in ids}


# Availability multiplier on season projection. ESPN's projection is a full-season
# number that does NOT price the current injury tag, so apply it here: OUT costs
# roughly a quarter of the remaining season, QUESTIONABLE a token amount for the
# elevated re-injury/limited-snap risk. DOUBTFUL sits between.
INJURY_MULT = {"OUT": 0.75, "DOUBTFUL": 0.85, "QUESTIONABLE": 0.95,
               "INJURY_RESERVE": 0.30, "SUSPENSION": 0.60}


def live_rosters(board=None):
    """Rosters from live_rosters.json, valued off the FULL player file (deep bench
    players fall outside draftmodel's top-500 slice) with injury tags applied.
    injury_overrides.json (manual news overrides) beats the ESPN tag. Warns when the
    board itself is stale — projections and tags only move when fetch_board runs."""
    import os, time
    age_h = (time.time() - os.path.getmtime("board_2026.json")) / 3600
    if age_h > 24:
        print(f"*** WARNING: board_2026.json is {age_h:.0f}h old - run fetch_board.py "
              f"before trusting projections/injury tags ***")
    try:
        overrides = {k: v["status"].upper() for k, v in
                     json.load(open("injury_overrides.json")).items()
                     if not k.startswith("_")}
    except FileNotFoundError:
        overrides = {}
    rows = json.load(open("board_2026.json"))
    by_id = {}
    for r in rows:
        r = dict(r)
        r["proj"] = float(r.get("proj") or 0.0)
        r["bye"] = int(r.get("bye") or 0)
        r["injury"] = overrides.get(r["name"], (r.get("injury") or "ACTIVE").upper())
        r["proj_healthy"] = r["proj"]
        r["proj"] = round(r["proj"] * INJURY_MULT.get(r["injury"], 1.0), 1)
        by_id[r["id"]] = r

    raw = json.load(open("live_rosters.json"))
    rosters, missing = {}, []
    for team, ids in raw.items():
        if team.startswith("_"):
            continue
        rosters[team] = [by_id[p] for p in ids if p in by_id]
        missing += [(team, p) for p in ids if p not in by_id]
    if missing:
        print(f"warning: playerIds not in board_2026.json, treated as empty slots: {missing}")
    return rosters


def evaluator(rosters):
    """Returns (base_wins, grade_fn, base_mus, trade_mus_fn). Weekly lineups are
    STREAMING-AWARE: an unfilled QB/K/D-ST slot scores the best free agent, not
    zero — otherwise every trade shipping a backup QB is overpenalized."""
    teams = list(rosters)
    raw = json.load(open("board_2026.json"))
    rostered = {r["id"] for rows in rosters.values() for r in rows}
    fa = {}
    for pos in ("QB", "K", "D/ST"):
        b = max((r for r in raw if r["pos"] == pos and r["id"] not in rostered),
                key=lambda r: float(r.get("proj") or 0))
        fa[pos] = float(b["proj"] or 0)

    def mus_of(rows):
        out = []
        for w in range(1, dm.REG_WEEKS + 1):
            need, pts = dict(dm.STARTERS), 0.0
            for r in sorted(rows, key=lambda r: -r["proj"]):
                if r["bye"] == w:
                    continue
                if need.get(r["pos"], 0) > 0:
                    need[r["pos"]] -= 1; pts += r["proj"] / dm.GAMES
                elif r["pos"] in dm.FLEX_POS and need.get("FLEX", 0) > 0:
                    need["FLEX"] -= 1; pts += r["proj"] / dm.GAMES
            for pos, n in need.items():
                if n > 0 and pos in fa:
                    pts += n * fa[pos] / dm.GAMES
            out.append(pts)
        return out

    base_mus = {t: mus_of(rows) for t, rows in rosters.items()}

    def wins_of(t, mus):
        return sum(sum(dm.phi((mus[t][w] - mus[o][w]) / dm.SIGMA)
                       for o in teams if o != t) / (len(teams) - 1)
                   for w in range(dm.REG_WEEKS))

    base = {t: wins_of(t, base_mus) for t in teams}

    def trade_mus(opp, mine_out, theirs_in):
        out_ids = {r["id"] for r in mine_out}
        in_ids = {r["id"] for r in theirs_in}
        mus = dict(base_mus)
        mus[ME] = mus_of([r for r in rosters[ME] if r["id"] not in out_ids] + theirs_in)
        mus[opp] = mus_of([r for r in rosters[opp] if r["id"] not in in_ids] + mine_out)
        return mus

    def grade(opp, mine_out, theirs_in):
        mus = trade_mus(opp, mine_out, theirs_in)
        return wins_of(ME, mus) - base[ME], wins_of(opp, mus) - base[opp]

    return base, grade, base_mus, trade_mus


def skill(rows):
    return [r for r in rows if r["pos"] not in ("K", "D/ST")]


def fill(rows, week):
    """Best lineup for one week. Returns (points, unfilled slots, replacement-level
    starters) — a slot only goes unfilled when a team literally has nobody eligible."""
    need, pts, weak = dict(dm.STARTERS), 0.0, []
    for r in sorted(rows, key=lambda r: -r["proj"]):
        if r["bye"] == week:
            continue
        slot = None
        if need.get(r["pos"], 0) > 0:
            slot, need[r["pos"]] = r["pos"], need[r["pos"]] - 1
        elif r["pos"] in dm.FLEX_POS and need.get("FLEX", 0) > 0:
            slot, need["FLEX"] = "FLEX", need["FLEX"] - 1
        if slot:
            pts += r["proj"] / dm.GAMES
            if r["proj"] < 120:                      # a body, not a starter
                weak.append(f"{slot}:{r['name'].split()[-1]}")
    return pts, [s for s, n in need.items() if n > 0], weak


def depth_report(rosters):
    """Who is rich/poor at each position. A team's tradeable surplus is its best
    player NOT in the typical-week starting lineup; its need is its weakest starter."""
    print(f"{'team':5s} {'RBs>=150':>9} {'WRs>=150':>9} {'best benched RB':>26s} "
          f"{'best benched WR':>26s}  weakest starter")
    for t, rows in rosters.items():
        need, starters = dict(dm.STARTERS), set()
        for r in sorted(rows, key=lambda r: -r["proj"]):
            if need.get(r["pos"], 0) > 0:
                need[r["pos"]] -= 1
                starters.add(r["id"])
            elif r["pos"] in dm.FLEX_POS and need.get("FLEX", 0) > 0:
                need["FLEX"] -= 1
                starters.add(r["id"])
        bench = [r for r in rows if r["id"] not in starters]
        best = {}
        for pos in ("RB", "WR", "TE"):
            cands = [r for r in bench if r["pos"] == pos]
            best[pos] = max(cands, key=lambda r: r["proj"]) if cands else None
        weak = min((r for r in rows if r["id"] in starters
                    and r["pos"] not in ("K", "D/ST")), key=lambda r: r["proj"])
        fmt = lambda r: f"{r['name'][:18]:18s} {r['proj']:6.1f}" if r else f"{'-':18s} {'':6s}"
        nrb = sum(1 for r in rows if r["pos"] == "RB" and r["proj"] >= 150)
        nwr = sum(1 for r in rows if r["pos"] == "WR" and r["proj"] >= 150)
        print(f"{t:5s} {nrb:9d} {nwr:9d} {fmt(best['RB']):>26s} {fmt(best['WR']):>26s}  "
              f"{weak['pos']}:{weak['name'][:16]} {weak['proj']:.0f}")


def bye_report(rosters):
    """Every team's structural bye holes: weeks far below their own typical week."""
    print(f"{'team':5s} {'avg':>6} {'worst weeks (pts, gap vs avg, cause)':50s}")
    holes = {}
    for t, rows in rosters.items():
        wk = {w: fill(rows, w) for w in range(1, dm.REG_WEEKS + 1)}
        avg = sum(v[0] for v in wk.values()) / dm.REG_WEEKS
        bad = sorted(wk.items(), key=lambda kv: kv[1][0])[:3]
        holes[t] = [(w, avg - p) for w, (p, _, _) in bad if avg - p > 3.0]
        desc = []
        for w, (p, empty, weak) in bad:
            if avg - p < 3.0:
                continue
            cause = ("EMPTY " + ",".join(empty)) if empty else (",".join(weak[:2]) or "depth")
            desc.append(f"wk{w} {p:.0f} (-{avg - p:.0f}, {cause})")
        print(f"{t:5s} {avg:6.1f}  {'; '.join(desc) if desc else 'no material bye hole'}")
    return holes


def scan(rosters, grade, opponents, top, max_players=4, exclude=()):
    """Every package with at most `max_players` bodies TOTAL across both sides
    (so 4 covers 1-for-1 up to 2-for-2 and 3-for-1). Names in `exclude` are
    untouchable on the other side - use it when a player is known unavailable."""
    fc = market_values()
    sizes = [(a, b) for a in range(1, max_players) for b in range(1, max_players)
             if a + b <= max_players]
    found = []
    for opp in opponents:
        mine = skill(rosters[ME])
        theirs = [r for r in skill(rosters[opp]) if r["name"] not in exclude]
        for na, nb in sizes:
            thr = 0.02 if na + nb == 2 else 0.05
            for g in itertools.combinations(mine, na):
                for r in itertools.combinations(theirs, nb):
                    dme, dt = grade(opp, list(g), list(r))
                    if dme > thr:
                        found.append((dme, dt, opp, list(g), list(r)))

    def show(rows, title):
        print(f"\n=== {title} ===")
        if not rows:
            print("  none")
        seen = set()
        shown = 0
        for dme, dt, opp, give, get in rows:
            # dedupe interchangeable bench filler, but keep each package SHAPE
            key = (opp, tuple(sorted(r["name"] for r in get)), len(give))
            if key in seen:
                continue
            seen.add(key)
            calc = (sum(fc_value(fc, r["name"]) for r in give)
                    - sum(fc_value(fc, r["name"]) for r in get)) if fc else 0
            print(f"  {opp}: give {' + '.join(r['name'] for r in give):40s} "
                  f"get {' + '.join(r['name'] for r in get):40s} "
                  f"me {dme:+.3f}  them {dt:+.3f}  calc-for-them {calc:+6.0f}")
            shown += 1
            if shown >= top:
                break

    # A counterparty needs a REASON to accept: "+0.01 for them" is a rounding error
    # they'd rationally refuse. Only the MUTUAL tier is worth actually proposing.
    show(sorted([f for f in found if f[1] > 0.04], key=lambda x: -min(x[0], x[1])),
         "MUTUAL (both sides gain >0.04 - actually proposable)")
    show(sorted([f for f in found if 0.005 < f[1] <= 0.04], key=lambda x: -x[0]),
         "MARGINAL for them (they gain <0.04 - expect a rational refusal)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", metavar="TEAM", help="'all' or a team abbrev")
    ap.add_argument("--byes", action="store_true", help="every team's bye-week holes")
    ap.add_argument("--depth", action="store_true", help="positional surplus/need per team")
    ap.add_argument("--grade", nargs=2, metavar=("GIVE", "GET"),
                    help="comma-separated player names: what I give, what I get")
    ap.add_argument("--top", type=int, default=10, help="rows per category")
    ap.add_argument("--max-players", type=int, default=4,
                    help="max bodies TOTAL across both sides of a scanned package")
    ap.add_argument("--exclude", default="",
                    help="comma-separated players the other side will not trade")
    args = ap.parse_args()

    rosters = live_rosters()
    base, grade, base_mus, trade_mus = evaluator(rosters)

    if args.grade:
        by_name = {r["name"]: (t, r) for t, rows in rosters.items() for r in rows}

        def look(n):
            n = n.strip()
            hits = [k for k in by_name if k.lower() == n.lower()] or \
                   [k for k in by_name if n.lower() in k.lower()]
            if len(hits) != 1:
                raise SystemExit(f"{n!r} matched {hits or 'nothing'} - be more specific")
            return by_name[hits[0]]

        give = [look(n) for n in args.grade[0].split(",")]
        get = [look(n) for n in args.grade[1].split(",")]
        if any(t != ME for t, _ in give):
            raise SystemExit(f"not mine: {[r['name'] for t, r in give if t != ME]}")
        opps = {t for t, _ in get}
        if len(opps) != 1:
            raise SystemExit(f"players are on different teams ({opps}) - not one trade")
        opp = opps.pop()
        give_r = [r for _, r in give]
        get_r = [r for _, r in get]
        dme, dt = grade(opp, give_r, get_r)
        print(f"vs {opp}: give {', '.join(r['name'] for r in give_r)} | "
              f"get {', '.join(r['name'] for r in get_r)}")
        print(f"  me {dme:+.3f} wins   {opp} {dt:+.3f} wins")

        # objective that actually matters: playoff/title odds (Monte Carlo)
        b = mc_odds(base_mus)[ME]
        a = mc_odds(trade_mus(opp, give_r, get_r))[ME]
        print(f"  Monte Carlo: playoffs {b[1]*100:.1f}% -> {a[1]*100:.1f}% "
              f"({(a[1]-b[1])*100:+.1f}pp)   title {b[2]*100:.1f}% -> {a[2]*100:.1f}% "
              f"({(a[2]-b[2])*100:+.1f}pp)")

        # what the counterparty's trade calculator will say (acceptance predictor)
        fc = market_values()
        if fc:
            gv = sum(fc_value(fc, r["name"]) for r in give_r)
            tv = sum(fc_value(fc, r["name"]) for r in get_r)
            lean = "ACCEPT" if gv > tv else "DECLINE"
            print(f"  market optics: they receive {gv:.0f} for {tv:.0f} "
                  f"-> their calculator leans {lean} ({gv - tv:+.0f})")
        return

    if args.depth:
        depth_report(rosters)
        return

    if args.byes:
        bye_report(rosters)
        return

    if args.scan:
        opponents = [t for t in rosters if t != ME] if args.scan == "all" else [args.scan]
        scan(rosters, grade, opponents, args.top, args.max_players,
             {n.strip() for n in args.exclude.split(",") if n.strip()})
        return

    print(f"{'team':5s} {'expW':>6}")
    for t in sorted(base, key=lambda t: -base[t]):
        print(f"{t:5s} {base[t]:6.2f}{'   <- me' if t == ME else ''}")
    for t in (ME,):
        print(f"\n=== {t} ===")
        for r in sorted(rosters[t], key=lambda r: (r["pos"], -r["proj"])):
            tag = "" if r["injury"] == "ACTIVE" else f"  {r['injury']} (healthy {r['proj_healthy']:.0f})"
            print(f"  {r['pos']:5s} {r['name']:24s} {r['proj']:6.1f} bye {r['bye']}{tag}")


if __name__ == "__main__":
    main()
