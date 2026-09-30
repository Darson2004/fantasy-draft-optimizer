"""Lookahead draft model for a private 10-team ESPN league — architecture per DRAFT_STRATEGY_RESEARCH.md.

The model plays the draft like a perfect-information game, which it (almost) is:
opponents on ESPN autopick are deterministic given the board, so their future picks
can be simulated exactly (Fry, Lundberg & Ohlmann 2007 — lookahead provably dominates
greedy in this regime).

  * Opponent model: the ESPN autopick replica from autodraft.py, replayed forward.
    Optional softmax noise for human league-mates (--noise, --sims).
  * My in-simulation policy: Fry restriction R2 + dynamic VBD — only consider players
    who will NOT survive to my next pick (survivors are claimable later), take the one
    with the most value over a demand-weighted dynamic positional baseline.
  * On the clock: 2-ply lookahead (this pick x my next pick jointly — the turn-pair
    play around the snake turn), draft completed with the greedy policy, scored on the final roster.
  * Objective: expected weekly head-to-head WINS over the 14-week regular season —
    bye-aware best starting lineup per week, win prob = Phi(points diff / sigma),
    averaged over the field or over schedule.json opponents when provided.
  * Pick trades: priced by re-simulating the entire draft under alternative pick
    ownership and diffing expected wins for both sides (--trades).

Inputs
  board_2026.json    fetch_board.py output (proj + bye fields required)
  autodraft.py       ESPN autopick replica; league config from league_config.json
  league_config.json optional slot / draft order / keepers (see autodraft.py)
  drafted.txt        live draft log, one name per line, in overall pick order
  pick_owners.json   optional {"<pick#>": "<team>"} overrides for traded picks
  schedule.json      optional {"<week>": "<opponent>"} my H2H schedule

Usage
  python draftmodel.py                    # recommend my next pick (2-ply lookahead)
  python draftmodel.py --trades           # which pick trades to pursue
  python draftmodel.py --matchups         # weekly matchup / win-probability outlook
  python draftmodel.py --pick-values      # model-vs-autodraft edge + projected board
"""
import argparse
import bisect
import json
import math
import random

from autodraft import (TEAMS, ROUNDS, SCORING, DRAFT_ORDER, STARTERS, FLEX_POS,
                       POS_CAP, ROSTERS, EXCLUDE_POS, KEEPER_PICKS, LIVE_PICKS,
                       resolve, slots_open, eligible, load_drafted, owner_of)

ME = "me"
REG_WEEKS = 14              # ESPN default regular season, playoffs weeks 15-17
GAMES = 17                  # NFL games per player-season -> weekly points = proj/17
SIGMA = 35.0                # SD of a weekly H2H points margin (two teams, ~25 each)
BOARD_DEPTH = 500           # nothing below this rank matters in a 160-pick draft
FLEX_SHARE = {"RB": 0.45, "WR": 0.45, "TE": 0.10}   # how FLEX demand splits


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def round_of(pick):
    return (pick - 1) // TEAMS + 1


# ------------------------------------------------------------------ inputs / state
def load_board():
    rows = sorted(json.load(open("board_2026.json")), key=lambda r: r[f"rank_{SCORING}"])
    rows = [r for r in rows if r["pos"] in POS_CAP][:BOARD_DEPTH]   # drops IDP/TQB/coach rows
    for r in rows:
        r["proj"] = float(r.get("proj") or 0.0)
        r["bye"] = int(r.get("bye") or 0)
    return rows


def load_pick_owners():
    try:
        return {int(k): v for k, v in json.load(open("pick_owners.json")).items()}
    except FileNotFoundError:
        return {}


def load_schedule():
    try:
        return {int(k): v for k, v in json.load(open("schedule.json")).items()}
    except FileNotFoundError:
        return None


def pick_owner(pick, ov):
    return ov.get(pick, owner_of(pick))


def team_picks(ov):
    tp = {t: [] for t in DRAFT_ORDER}
    for p in LIVE_PICKS:
        tp[pick_owner(p, ov)].append(p)
    return tp


def picks_left_of(tp, team, pick):
    lst = tp[team]
    return len(lst) - bisect.bisect_left(lst, pick)


def initial_state(board, ov):
    """Rosters (keepers + picks made so far), taken ids, next overall pick.
    drafted.txt holds only LIVE picks (keeper-reserved slots make no selection)."""
    rosters = {t: [resolve(board, n) for n in names] for t, names in ROSTERS.items()}
    drafted = load_drafted(board)
    for row, p in zip(drafted, LIVE_PICKS):
        rosters[pick_owner(p, ov)].append(row)
    taken = {r["id"] for rows in rosters.values() for r in rows}
    cur = LIVE_PICKS[len(drafted)] if len(drafted) < len(LIVE_PICKS) else TEAMS * ROUNDS + 1
    return {"rosters": rosters, "taken": taken, "cur": cur}


def clone(st):
    return {"rosters": {t: list(v) for t, v in st["rosters"].items()},
            "taken": set(st["taken"]), "cur": st["cur"]}


# ------------------------------------------------------------------ pick policies
def espn_autopick(board, taken, roster_rows, picks_left, rng=None, temp=1.5, top=6):
    """ESPN autopick replica; with rng, a softmax over the top candidates (human noise)."""
    pos_list = [r["pos"] for r in roster_rows]
    cands = []
    for r in board:
        if r["id"] not in taken and eligible(r, pos_list, picks_left):
            cands.append(r)
            if len(cands) >= (1 if rng is None else top):
                break
    if not cands:
        return None
    if rng is None or len(cands) == 1:
        return cands[0]
    return rng.choices(cands, weights=[math.exp(-i / temp) for i in range(len(cands))])[0]


def forward_gone(board, st, ov, tp, frm, to):
    """Ids opponents take in picks frm..to-1 under deterministic autopick."""
    taken = set(st["taken"])
    rposes = {t: [r["pos"] for r in rows] for t, rows in st["rosters"].items()}
    gone = set()
    for p in range(frm, to):
        if p in KEEPER_PICKS:
            continue
        t = pick_owner(p, ov)
        cand = None
        for r in board:
            if r["id"] not in taken and eligible(r, rposes[t], picks_left_of(tp, t, p)):
                cand = r
                break
        if cand:
            taken.add(cand["id"])
            gone.add(cand["id"])
            rposes[t].append(cand["pos"])
    return gone


def dyn_baselines(board, taken, rosters):
    """Replacement level per position = projection of the player just beyond
    remaining league-wide starter demand (dynamic VBD baseline)."""
    demand = {p: 0.0 for p in POS_CAP}
    for rows in rosters.values():
        for pos, n in slots_open([r["pos"] for r in rows]).items():
            if pos == "FLEX":
                for fp, sh in FLEX_SHARE.items():
                    demand[fp] += n * sh
            else:
                demand[pos] += n
    base = {}
    for pos in demand:
        k, seen = max(1, round(demand[pos])), 0
        base[pos] = 0.0
        for r in board:
            if r["id"] in taken or r["pos"] != pos:
                continue
            seen += 1
            if seen > k:
                base[pos] = r["proj"]
                break
    return base


def slot_value(pos, proj, open_now, base, flex_base):
    """Value over the replacement level of the slot this player would actually fill:
    positional baseline for an open starter slot, flex baseline for a FLEX fill,
    Fry's beta=0.6 backup discount for a pure bench pick."""
    if open_now.get(pos, 0) > 0:
        return proj - base[pos]
    if pos in FLEX_POS and open_now.get("FLEX", 0) > 0:
        return proj - flex_base
    return 0.6 * (proj - base[pos])


def greedy_pick(board, st, ov, tp, pick):
    """Two-pick pair value (Fry 2007 lookahead): opponents' picks up to my next turn
    are simulated exactly, so score(candidate) = value(candidate now) + value(best
    player still attainable at my next pick given that choice). Survivors are never
    reached for unless the pair math says so."""
    mine = tp[ME]
    i = bisect.bisect_right(mine, pick)
    nxt = mine[i] if i < len(mine) else None
    left = picks_left_of(tp, ME, pick)
    gone = forward_gone(board, st, ov, tp, pick + 1, nxt) if nxt else set()
    base = dyn_baselines(board, st["taken"], st["rosters"])
    flex_base = max(base[p] for p in FLEX_POS)
    pos_list = [r["pos"] for r in st["rosters"][ME]]
    open_now = slots_open(pos_list)

    cands = []
    for r in board:
        if r["id"] in st["taken"] or not eligible(r, pos_list, left):
            continue
        cands.append((slot_value(r["pos"], r["proj"], open_now, base, flex_base), r))
    if not cands:
        return None
    cands.sort(key=lambda x: -x[0])
    if nxt is None:
        return cands[0][1]

    # two best survivors per position = what my next pick can still claim
    top2 = {p: [] for p in POS_CAP}
    for r in board:
        if r["id"] in st["taken"] or r["id"] in gone:
            continue
        lst = top2[r["pos"]]
        if len(lst) < 2:
            lst.append(r["proj"])
        elif r["proj"] > lst[1]:
            lst[1] = r["proj"]
            lst.sort(reverse=True)

    best, bestv = None, None
    for v1, x in cands[:10]:
        pos_list2 = pos_list + [x["pos"]]
        open2 = slots_open(pos_list2)
        skill2 = {k: n for k, n in open2.items() if k not in ("K", "D/ST")}
        v2 = 0.0
        for pos, projs in top2.items():
            if pos in ("K", "D/ST") or not projs:
                continue
            if pos_list2.count(pos) >= POS_CAP.get(pos, 99):
                continue
            if skill2 and not (pos in skill2 or (pos in FLEX_POS and "FLEX" in skill2)):
                continue
            proj = projs[0]
            if pos == x["pos"] and x["proj"] >= projs[0]:
                proj = projs[1] if len(projs) > 1 else 0.0
            v2 = max(v2, slot_value(pos, proj, open2, base, flex_base))
        if bestv is None or v1 + v2 > bestv:
            best, bestv = x, v1 + v2
    return best


def sim_draft(board, st0, ov, tp, me_policy="greedy", rng=None):
    """Play out picks cur..end. Opponents: ESPN autopick (optionally noisy).
    Me: the greedy R2/VBD policy, or 'autopick' for the pure-autodraft baseline."""
    st = clone(st0)
    for p in range(st0["cur"], TEAMS * ROUNDS + 1):
        if p in KEEPER_PICKS:
            continue
        t = pick_owner(p, ov)
        left = picks_left_of(tp, t, p)
        if t == ME and me_policy == "greedy":
            st["cur"] = p
            row = greedy_pick(board, st, ov, tp, p)
        else:
            row = espn_autopick(board, st["taken"], st["rosters"][t], left,
                                rng if t != ME else None)
        if row:
            st["taken"].add(row["id"])
            st["rosters"][t].append(row)
    return st["rosters"]


# ------------------------------------------------------------------ evaluation
def weekly_mus(rows):
    """Best-starting-lineup expected points for each regular-season week (bye-aware)."""
    rows = sorted(rows, key=lambda r: -r["proj"])
    mus = []
    for w in range(1, REG_WEEKS + 1):
        need, pts = dict(STARTERS), 0.0
        for r in rows:
            if r["bye"] == w:
                continue
            if need.get(r["pos"], 0) > 0:
                need[r["pos"]] -= 1
                pts += r["proj"] / GAMES
            elif r["pos"] in FLEX_POS and need.get("FLEX", 0) > 0:
                need["FLEX"] -= 1
                pts += r["proj"] / GAMES
        mus.append(pts)
    return mus


def exp_wins_all(rosters, schedule=None):
    """Expected H2H wins per team over the regular season. My weeks use schedule.json
    opponents when given; otherwise (and for everyone else) the field average."""
    mus = {t: weekly_mus(rows) for t, rows in rosters.items()}
    wins = {}
    for t in rosters:
        tot = 0.0
        for w in range(REG_WEEKS):
            if t == ME and schedule and (w + 1) in schedule:
                opps = [schedule[w + 1]]
            else:
                opps = [o for o in rosters if o != t]
            tot += sum(phi((mus[t][w] - mus[o][w]) / SIGMA) for o in opps) / len(opps)
        wins[t] = tot
    return wins, mus


def score_me(board, st, ov, tp, schedule, noise=0.0, sims=1):
    if not noise:
        return exp_wins_all(sim_draft(board, st, ov, tp), schedule)[0][ME]
    tot = 0.0
    for s in range(sims):
        rosters = sim_draft(board, st, ov, tp, rng=random.Random(1234 + s))
        tot += exp_wins_all(rosters, schedule)[0][ME]
    return tot / sims


# ------------------------------------------------------------------ modes
def candidates_at(board, st, tp, pick, breadth):
    pos_list = [r["pos"] for r in st["rosters"][ME]]
    left = picks_left_of(tp, ME, pick)
    out, per = [], {}
    for r in board:
        if r["id"] in st["taken"] or per.get(r["pos"], 0) >= breadth:
            continue
        if eligible(r, pos_list, left):
            out.append(r)
            per[r["pos"]] = per.get(r["pos"], 0) + 1
    return out


def eval_candidate(board, st, ov, tp, pick, row, depth, breadth, schedule, noise, sims):
    st1 = clone(st)
    st1["taken"].add(row["id"])
    st1["rosters"][ME].append(row)
    mine = tp[ME]
    i = bisect.bisect_right(mine, pick)
    nxt = mine[i] if i < len(mine) else None
    if depth >= 2 and nxt:
        for q in range(pick + 1, nxt):        # advance opponents to my next pick
            if q in KEEPER_PICKS:
                continue
            t = pick_owner(q, ov)
            r2 = espn_autopick(board, st1["taken"], st1["rosters"][t],
                               picks_left_of(tp, t, q))
            if r2:
                st1["taken"].add(r2["id"])
                st1["rosters"][t].append(r2)
        st1["cur"] = nxt
        best = None
        for c2 in candidates_at(board, st1, tp, nxt, breadth):
            st2 = clone(st1)
            st2["taken"].add(c2["id"])
            st2["rosters"][ME].append(c2)
            st2["cur"] = nxt + 1
            w = score_me(board, st2, ov, tp, schedule, noise, sims)
            if best is None or w > best[0]:
                best = (w, c2)
        return best if best else (score_me(board, st1, ov, tp, schedule, noise, sims), None)
    st1["cur"] = pick + 1
    return score_me(board, st1, ov, tp, schedule, noise, sims), None


def mode_recommend(args, board, ov, schedule):
    st = initial_state(board, ov)
    tp = team_picks(ov)
    mine = [p for p in tp[ME] if p >= st["cur"]]
    if not mine:
        print("no picks left - see --matchups")
        return
    target = mine[0]

    # advance the board to my pick (deterministic autopick replay)
    st2 = clone(st)
    print(f"picks made: {st['cur'] - 1}   my next pick: #{target} (round {round_of(target)})")
    for p in range(st["cur"], target):
        if p in KEEPER_PICKS:
            continue
        t = pick_owner(p, ov)
        row = espn_autopick(board, st2["taken"], st2["rosters"][t], picks_left_of(tp, t, p))
        if row:
            st2["taken"].add(row["id"])
            st2["rosters"][t].append(row)
            print(f"  projected #{p:3d} {t:4s} -> {row['name']} ({row['pos']})")
    st2["cur"] = target

    nxt = mine[1] if len(mine) > 1 else None
    gone_next = forward_gone(board, st2, ov, tp, target + 1, nxt) if nxt else set()
    base = dyn_baselines(board, st2["taken"], st2["rosters"])

    cands = candidates_at(board, st2, tp, target, args.breadth)
    results = []
    for c in cands:
        w, follow = eval_candidate(board, st2, ov, tp, target, c,
                                   args.depth, args.breadth, schedule, args.noise, args.sims)
        results.append((w, c, follow))
    results.sort(key=lambda x: -x[0])

    print(f"\n=== PICK #{target}: 2-ply lookahead, objective = expected H2H wins (of {REG_WEEKS}) ===")
    hdr = f"{'expW':>6} {'name':22s} {'pos':4s} {'proj':>6} {'vbd':>6} {'bye':>3} {'adp':>6} {'back?':6s}  best follow-up"
    print(hdr)
    top_w = results[0][0]
    for w, c, follow in results:
        vbd = c["proj"] - base[c["pos"]]
        if nxt is None:
            back = "-"
        elif c["id"] in gone_next:
            back = "GONE"
        else:
            back = "safe"
        fname = f"{follow['name']} ({follow['pos']})" if follow else "-"
        flag = " <<<" if w == top_w else ""
        print(f"{w:6.2f} {c['name'][:22]:22s} {c['pos']:4s} {c['proj']:6.1f} {vbd:6.1f} "
              f"{c['bye']:3d} {str(c['adp'])[:6]:>6s} {back:6s}  {fname}{flag}")
    best = results[0]
    print(f"\nMODEL TAKES: {best[1]['name']} ({best[1]['pos']})"
          + (f", planning {best[2]['name']} at #{nxt}" if best[2] and nxt else ""))


def mode_trades(args, board, ov, schedule):
    st = initial_state(board, ov)
    tp = team_picks(ov)
    base_rosters = sim_draft(board, st, ov, tp)
    base_wins, _ = exp_wins_all(base_rosters, schedule)
    print(f"baseline (current picks, model drafting): my expected wins = {base_wins[ME]:.2f}/{REG_WEEKS}")

    mine = [p for p in tp[ME] if p >= st["cur"] and round_of(p) <= args.max_round]
    trades = []
    n_done = 0
    for t in DRAFT_ORDER:
        if t == ME:
            continue
        theirs = [p for p in tp[t] if p >= st["cur"] and round_of(p) <= args.max_round]
        for i in mine:
            for j in theirs:
                ov2 = dict(ov)
                ov2[i], ov2[j] = t, ME
                tp2 = team_picks(ov2)
                wins2, _ = exp_wins_all(sim_draft(board, st, ov2, tp2), schedule)
                trades.append((wins2[ME] - base_wins[ME], wins2[t] - base_wins[t], i, j, t))
                n_done += 1
        print(f"  evaluated vs {t}: {n_done} swaps so far")

    gains = sorted([x for x in trades if x[0] > 0.02], key=lambda x: -x[0])
    print(f"\n=== 1-for-1 pick swaps that raise my expected wins (rounds 1-{args.max_round}) ===")
    print(f"{'dMyW':>6} {'dTheirW':>8}  trade")
    for dm, dt, i, j, t in gains[:args.top]:
        sell = "sellable" if dt > -0.10 else ("fair" if dt > -0.25 else "hard sell")
        print(f"{dm:+6.2f} {dt:+8.2f}  give #{i} (R{round_of(i)}) to {t} for #{j} (R{round_of(j)})  [{sell}]")
    if not gains:
        print("  none - current pick set is already near-optimal against this field")
    win_win = [x for x in gains if x[1] > -0.02]
    if win_win:
        print("\nwin-win (they also gain or lose ~nothing - propose these first):")
        for dm, dt, i, j, t in win_win[:10]:
            print(f"  {t}: my #{i} (R{round_of(i)}) for their #{j} (R{round_of(j)})  "
                  f"me {dm:+.2f}W, them {dt:+.2f}W")


def mode_matchups(args, board, ov, schedule):
    st = initial_state(board, ov)
    tp = team_picks(ov)
    rosters = sim_draft(board, st, ov, tp)
    wins, mus = exp_wins_all(rosters, schedule)

    print("=== projected standings (expected H2H wins vs field) ===")
    for t in sorted(rosters, key=lambda t: -wins[t]):
        season = sum(mus[t])
        tag = "  <- me" if t == ME else ""
        print(f"  {t:4s} expW {wins[t]:5.2f}/{REG_WEEKS}   season pts {season:7.1f}{tag}")

    print(f"\n=== my week-by-week outlook (sigma={SIGMA:.0f}) ===")
    print(f"{'wk':>3} {'me':>7} {'field':>7} {'top opp':>14} {'win%':>6}")
    opps = [t for t in rosters if t != ME]
    for w in range(REG_WEEKS):
        field = sum(mus[o][w] for o in opps) / len(opps)
        top = max(opps, key=lambda o: mus[o][w])
        if schedule and (w + 1) in schedule:
            o = schedule[w + 1]
            wp = phi((mus[ME][w] - mus[o][w]) / SIGMA)
            vs = f"vs {o}"
        else:
            wp = sum(phi((mus[ME][w] - mus[o][w]) / SIGMA) for o in opps) / len(opps)
            vs = "vs field"
        print(f"{w + 1:3d} {mus[ME][w]:7.1f} {field:7.1f} {top:>6s} {mus[top][w]:6.1f} {wp * 100:5.1f}% {vs}")

    print("\n=== per-opponent edge (avg weekly win prob) ===")
    for o in sorted(opps, key=lambda o: -sum(phi((mus[ME][w] - mus[o][w]) / SIGMA) for w in range(REG_WEEKS))):
        wp = sum(phi((mus[ME][w] - mus[o][w]) / SIGMA) for w in range(REG_WEEKS)) / REG_WEEKS
        label = "favorable" if wp > 0.55 else ("even" if wp > 0.45 else "UNFAVORABLE")
        print(f"  {o:4s} {wp * 100:5.1f}%  {label}")

    print("\n=== my projected final roster ===")
    for r in sorted(rosters[ME], key=lambda r: -r["proj"]):
        print(f"  {r['name']:24s} {r['pos']:5s} {r['team']:4s} proj {r['proj']:6.1f} bye {r['bye']}")


def mode_pick_values(args, board, ov, schedule):
    st = initial_state(board, ov)
    tp = team_picks(ov)
    r_auto = sim_draft(board, st, ov, tp, me_policy="autopick")
    r_model = sim_draft(board, st, ov, tp, me_policy="greedy")
    w_auto, _ = exp_wins_all(r_auto, schedule)
    w_model, mus = exp_wins_all(r_model, schedule)
    print(f"me on ESPN autopick : {w_auto[ME]:5.2f} expected wins, "
          f"{sum(weekly_mus(r_auto[ME])):7.1f} season pts")
    print(f"me on this model    : {w_model[ME]:5.2f} expected wins, "
          f"{sum(mus[ME]):7.1f} season pts")
    print(f"model edge          : {w_model[ME] - w_auto[ME]:+5.2f} wins\n")

    print("=== projected draft (model drafting for me, everyone else autopick) ===")
    st2 = clone(st)
    for p in range(st["cur"], TEAMS * ROUNDS + 1):
        if p in KEEPER_PICKS:
            continue
        t = pick_owner(p, ov)
        left = picks_left_of(tp, t, p)
        if t == ME:
            st2["cur"] = p
            row = greedy_pick(board, st2, ov, tp, p)
        else:
            row = espn_autopick(board, st2["taken"], st2["rosters"][t], left)
        if row:
            st2["taken"].add(row["id"])
            st2["rosters"][t].append(row)
            if round_of(p) <= args.show_rounds:
                tag = "  <== ME" if t == ME else ""
                print(f"  #{p:3d} R{round_of(p):2d} {t:4s} {row['name'][:24]:24s} "
                      f"{row['pos']:5s} proj {row['proj']:6.1f}{tag}")


def main():
    global SIGMA
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trades", action="store_true", help="evaluate pick trades")
    ap.add_argument("--matchups", action="store_true", help="weekly matchup outlook")
    ap.add_argument("--pick-values", action="store_true", help="model edge + projected board")
    ap.add_argument("--depth", type=int, default=2, help="lookahead plies (1 or 2)")
    ap.add_argument("--breadth", type=int, default=3, help="candidates per position")
    ap.add_argument("--max-round", type=int, default=8, help="trade search depth in rounds")
    ap.add_argument("--top", type=int, default=15, help="trades to display")
    ap.add_argument("--show-rounds", type=int, default=6, help="rounds shown by --pick-values")
    ap.add_argument("--sigma", type=float, default=SIGMA, help="weekly matchup SD")
    ap.add_argument("--noise", type=float, default=0.0, help="opponent softmax temperature (0=deterministic)")
    ap.add_argument("--sims", type=int, default=25, help="simulations when --noise > 0")
    args = ap.parse_args()

    SIGMA = args.sigma
    board = load_board()
    ov = load_pick_owners()
    schedule = load_schedule()
    if ov:
        print(f"pick trades in effect: {sorted(ov.items())}")

    if args.trades:
        mode_trades(args, board, ov, schedule)
    elif args.matchups:
        mode_matchups(args, board, ov, schedule)
    elif args.pick_values:
        mode_pick_values(args, board, ov, schedule)
    else:
        mode_recommend(args, board, ov, schedule)


if __name__ == "__main__":
    main()
