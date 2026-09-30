"""Simulated draft + season environment with strict point-in-time knowledge.

What a draft policy is allowed to see for season Y (the "view"):
  * that year's preseason ADP + ADP stdev + bye week (market knowledge, public
    before every real draft)
  * a projection PROXY: expected season points for a player at that ADP rank,
    from positional rank->points curves fitted ONLY on seasons < Y

What stays hidden inside the environment: season Y actual weekly points, used
solely to score completed seasons. Player identities are anonymized hashes.

Season play (v2, managed leagues): each week every team may make one waiver
move (drop its worst benched skill player for a clearly better free agent;
worst record claims first) and one 1-for-1 cross-position trade (executed only
if BOTH sides' expected starting lineups improve past their thresholds; trade
deadline week 10). Lineups are matchup-aware: best expected lineup, optionally
tilted toward volatile players when the team projects to LOSE the week and
toward floor players when it projects to WIN (var_tilt). Started players score
their ACTUAL points. All in-season knowledge is causal: preseason expectation
blended with observed points from earlier weeks of the same season only -
including free agents, whose stats are public.

Regular season weeks 1-13 (round-robin), semifinal week 14 (1v4, 2v3 by record
then points), final = cumulative weeks 15+16.
"""
import json
import os

from config import (TEAMS, ROUNDS, REG_WEEKS, SEMI_WEEK, FINAL_WEEKS, STARTERS,
                    FLEX_POS, POS_CAP, K_WEEKLY, DST_WEEKLY, slots_open)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# weekly sd proxy per expected point until 4 weeks of the season are observed
POS_CV = {"QB": 0.40, "RB": 0.55, "WR": 0.60, "TE": 0.70, "K": 0.0, "DST": 0.0}
DEFAULT_MGMT = {"var_tilt": 0.0, "waiver_thr": 1.0, "trade_min": 1.0}
TRADE_DEADLINE = 10
LAST = {"waivers": 0, "trades": 0}       # per-run_season counters (observability)


def available_years():
    return sorted(int(f[7:11]) for f in os.listdir(DATA_DIR)
                  if f.startswith("season_") and f.endswith(".json"))


def load_season(year):
    with open(os.path.join(DATA_DIR, f"season_{year}.json")) as f:
        d = json.load(f)
    seen, uniq = set(), []
    for p in d["players"]:               # ADP-sorted; keep the best-ADP copy of dupes
        if p["id"] in seen:
            continue
        seen.add(p["id"])
        p["weekly"] = {int(w): v for w, v in p["weekly"].items()}
        uniq.append(p)
    # some ADP feeds list fewer players than a 160-pick draft consumes; pad the
    # pool with replacement-level scrubs (0 actual points) so drafts complete
    i = 0
    while len(uniq) < TEAMS * ROUNDS + 10:
        uniq.append({"id": f"PAD_{year}_{i}", "pos": ("RB", "WR")[i % 2],
                     "adp": 400.0 + i, "adp_sd": 30.0, "bye": 0, "weekly": {}})
        i += 1
    d["players"] = uniq
    return d


# ------------------------------------------------------------------ knowledge
def fit_curves(prior_years, max_rank=80):
    """pos -> [expected season pts at ADP-rank 1, 2, ...] from prior seasons only."""
    samples = {}
    for y in prior_years:
        d = load_season(y)
        by_pos = {}
        for p in sorted(d["players"], key=lambda r: r["adp"]):
            by_pos.setdefault(p["pos"], []).append(p)
        scale = 16.0 / d["games"]          # normalize 17-game seasons to 16
        for pos, rows in by_pos.items():
            for rank, p in enumerate(rows[:max_rank]):
                total = sum(p["weekly"].values()) * scale
                samples.setdefault(pos, {}).setdefault(rank, []).append(total)
    curves = {}
    for pos, ranks in samples.items():
        n = max(ranks) + 1
        raw = [sum(ranks[r]) / len(ranks[r]) if r in ranks else 0.0 for r in range(n)]
        smooth = [sum(raw[max(0, r - 1):r + 2]) / len(raw[max(0, r - 1):r + 2]) for r in range(n)]
        for r in range(1, n):              # enforce non-increasing in rank
            smooth[r] = min(smooth[r], smooth[r - 1])
        curves[pos] = smooth
    return curves


def knowledge_view(year, prior_years):
    """The ONLY object a draft policy receives. No actual season-Y stats inside."""
    d = load_season(year)
    curves = fit_curves(prior_years)
    view, pos_rank = [], {}
    for p in sorted(d["players"], key=lambda r: r["adp"]):
        rank = pos_rank.get(p["pos"], 0)
        pos_rank[p["pos"]] = rank + 1
        curve = curves.get(p["pos"], [0.0])
        proj = curve[min(rank, len(curve) - 1)]
        if p["pos"] == "K":
            proj = K_WEEKLY * 16
        elif p["pos"] == "DST":
            proj = DST_WEEKLY * 16
        view.append({"id": p["id"], "pos": p["pos"], "adp": p["adp"],
                     "adp_sd": p["adp_sd"], "bye": p["bye"], "proj": round(proj, 1)})
    return view


# ------------------------------------------------------------------ draft
def run_draft(view, policies, rng):
    """Snake draft; policies is a list of TEAMS pick(view, ctx)->player_id.
    Returns rosters as lists of view rows. Draft slots are shuffled by rng."""
    order = list(range(TEAMS))
    rng.shuffle(order)                    # order[i] = policy index drafting slot i
    rosters = [[] for _ in range(TEAMS)]  # indexed by policy index
    taken = set()
    byid = {p["id"]: p for p in view}
    for pick in range(1, TEAMS * ROUNDS + 1):
        rnd, idx = divmod(pick - 1, TEAMS)
        slot = idx if rnd % 2 == 0 else TEAMS - 1 - idx
        pol_i = order[slot]
        ctx = {"pick": pick, "slot": slot, "rosters": rosters, "taken": taken,
               "my_index": pol_i, "order": order}
        pid = policies[pol_i].pick(view, ctx)
        if pid is None or pid in taken or pid not in byid:
            raise RuntimeError(f"policy {pol_i} returned invalid pick at #{pick}")
        taken.add(pid)
        rosters[pol_i].append(byid[pid])
    return rosters


def my_picks_for_slot(slot):
    out = []
    for pick in range(1, TEAMS * ROUNDS + 1):
        rnd, idx = divmod(pick - 1, TEAMS)
        s = idx if rnd % 2 == 0 else TEAMS - 1 - idx
        if s == slot:
            out.append(pick)
    return out


# ------------------------------------------------------------------ knowledge during the season
def actual_points(player, week, season):
    if player["pos"] == "K":
        return K_WEEKLY
    if player["pos"] == "DST":
        return DST_WEEKLY
    return season["byid"][player["id"]]["weekly"].get(week, 0.0)


def _observed(seen, pid):
    """(mean, sd) of weekly points seen so far, or None before 4 weeks."""
    s = seen.get(pid)
    if not s or s[1] < 4:
        return None
    mean = s[0] / s[1]
    var = max(s[2] / s[1] - mean * mean, 0.0)
    return mean, var ** 0.5


def exp_ppw(player, seen):
    """Manager's blended weekly expectation, ignoring byes (roster value)."""
    pre = player["proj"] / 16.0
    ob = _observed(seen, player["id"])
    return 0.5 * pre + 0.5 * ob[0] if ob else pre


def expected_points(player, week, seen):
    """This week's expectation: blended ppw, 0 on bye."""
    if player["bye"] == week:
        return 0.0
    return exp_ppw(player, seen)


def expected_sd(player, week, seen):
    """This week's volatility estimate: position CV prior, blended with the
    observed weekly sd once 4+ weeks are seen. K/DST are flat constants."""
    if player["bye"] == week or player["pos"] in ("K", "DST"):
        return 0.0
    prior = POS_CV[player["pos"]] * exp_ppw(player, seen)
    ob = _observed(seen, player["id"])
    return 0.5 * prior + 0.5 * ob[1] if ob else prior


# ------------------------------------------------------------------ lineups
def build_lineup(roster, week, seen):
    """Greedy best-expected lineup. Returns ([(player, slot), ...], expected total)."""
    ranked = sorted(roster, key=lambda p: -expected_points(p, week, seen))
    need, out, mu = dict(STARTERS), [], 0.0
    for p in ranked:
        pos = p["pos"]
        if need.get(pos, 0) > 0:
            need[pos] -= 1
            slot = pos
        elif pos in FLEX_POS and need.get("FLEX", 0) > 0:
            need["FLEX"] -= 1
            slot = "FLEX"
        else:
            continue
        out.append((p, slot))
        mu += expected_points(p, week, seen)
    return out, mu


def matchup_lineup(roster, week, seen, opp_mu, var_tilt, max_swaps=2):
    """Best expected lineup, then (if var_tilt > 0) up to max_swaps bench swaps
    that trade expected points for variance when trailing opp_mu, or for floor
    when leading. Direction is set once from the naive expected totals."""
    lineup, mu = build_lineup(roster, week, seen)
    if var_tilt <= 1e-9:
        return lineup
    started = {p["id"] for p, _ in lineup}
    bench = [p for p in roster if p["id"] not in started and p["pos"] not in ("K", "DST")]
    trailing = mu < opp_mu
    swaps = 0
    for b in sorted(bench, key=lambda p: -expected_sd(p, week, seen), reverse=not trailing):
        if swaps >= max_swaps:
            break
        b_mu, b_sd = expected_points(b, week, seen), expected_sd(b, week, seen)
        best = None
        for i, (s, slot) in enumerate(lineup):
            if not (b["pos"] == s["pos"] or (slot == "FLEX" and b["pos"] in FLEX_POS)):
                continue
            dmu = expected_points(s, week, seen) - b_mu          # cost (>= 0 typically)
            dsd = b_sd - expected_sd(s, week, seen)
            gain = var_tilt * (dsd if trailing else -dsd) - dmu
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, i)
        if best:
            lineup[best[1]] = (b, lineup[best[1]][1])
            swaps += 1
    return lineup


def lineup_actual(lineup, week, season):
    return sum(actual_points(p, week, season) for p, _ in lineup)


# ------------------------------------------------------------------ waivers & trades
def do_waivers(rosters, fa, week, seen, mgmt, order):
    """One add/drop per team: drop the worst benched skill player for the best
    legal free agent if the upgrade clears the team's threshold. `order` is
    worst-record-first (waiver priority)."""
    fa.sort(key=lambda p: -exp_ppw(p, seen))
    for t in order:
        roster = rosters[t]
        lineup, _ = build_lineup(roster, week, seen)
        started = {p["id"] for p, _ in lineup}
        droppable = [p for p in roster if p["id"] not in started and p["pos"] not in ("K", "DST")]
        if not droppable:
            continue
        worst = min(droppable, key=lambda p: exp_ppw(p, seen))
        counts = {}
        for p in roster:
            counts[p["pos"]] = counts.get(p["pos"], 0) + 1
        counts[worst["pos"]] -= 1
        for cand in fa:                   # first legal candidate = best available
            if cand["pos"] in ("K", "DST"):
                continue
            if counts.get(cand["pos"], 0) >= POS_CAP.get(cand["pos"], 99):
                continue
            if exp_ppw(cand, seen) - exp_ppw(worst, seen) > mgmt[t]["waiver_thr"]:
                roster.remove(worst)
                roster.append(cand)
                fa.remove(cand)
                fa.append(worst)
                LAST["waivers"] += 1
            break


def typical_lineup_mu(roster, seen):
    """Expected total of the best TYPICAL-week lineup (byes ignored) - the
    right basis for valuing permanent roster moves like trades."""
    ranked = sorted(roster, key=lambda p: -exp_ppw(p, seen))
    need, mu, started = dict(STARTERS), 0.0, set()
    for p in ranked:
        pos = p["pos"]
        if need.get(pos, 0) > 0:
            need[pos] -= 1
        elif pos in FLEX_POS and need.get("FLEX", 0) > 0:
            need["FLEX"] -= 1
        else:
            continue
        mu += exp_ppw(p, seen)
        started.add(p["id"])
    return mu, started


def _surplus(roster, seen):
    """Best typically-benched skill player per position (trade bait), + typical mu."""
    mu, started = typical_lineup_mu(roster, seen)
    out = {}
    for p in roster:
        if p["id"] in started or p["pos"] in ("K", "DST"):
            continue
        if p["pos"] not in out or exp_ppw(p, seen) > exp_ppw(out[p["pos"]], seen):
            out[p["pos"]] = p
    return list(out.values()), mu


def _cap_ok(roster, incoming_pos, outgoing_pos):
    n = sum(1 for p in roster if p["pos"] == incoming_pos) - (incoming_pos == outgoing_pos)
    return n < POS_CAP.get(incoming_pos, 99)


def do_trades(rosters, seen, mgmt, rng):
    """Each team may propose one 1-for-1 cross-position trade: its best surplus
    player for a counterparty's surplus player. Executes only if BOTH typical
    starting lineups improve past each side's trade_min (bye-neutral valuation)."""
    n = len(rosters)
    surp = [_surplus(rosters[t], seen) for t in range(n)]
    traded = set()
    order = list(range(n))
    rng.shuffle(order)
    for i in order:
        if i in traded or not surp[i][0]:
            continue
        offer = max(surp[i][0], key=lambda p: exp_ppw(p, seen))
        done = False
        for j in order:
            if j == i or j in traded:
                continue
            for want in surp[j][0]:
                if want["pos"] == offer["pos"]:
                    continue
                if not (_cap_ok(rosters[i], want["pos"], offer["pos"])
                        and _cap_ok(rosters[j], offer["pos"], want["pos"])):
                    continue
                ri = [p for p in rosters[i] if p["id"] != offer["id"]] + [want]
                if typical_lineup_mu(ri, seen)[0] - surp[i][1] <= mgmt[i]["trade_min"]:
                    continue
                rj = [p for p in rosters[j] if p["id"] != want["id"]] + [offer]
                if typical_lineup_mu(rj, seen)[0] - surp[j][1] <= mgmt[j]["trade_min"]:
                    continue
                rosters[i][:] = ri
                rosters[j][:] = rj
                traded.update((i, j))
                LAST["trades"] += 1
                done = True
                break
            if done:
                break


# ------------------------------------------------------------------ season
def run_season(rosters, season_data, rng, mgmt=None, pool=None, trade_fn=None):
    """Round-robin regular season + playoffs with weekly management.
    mgmt: per-team {var_tilt, waiver_thr, trade_min} (DEFAULT_MGMT if None).
    pool: the full knowledge view - enables waivers/trades from free agents;
    None disables management entirely (bare v1 behavior).
    trade_fn(rosters, week, seen, wins, rng): replaces the built-in threshold
    trade rule with an external policy (e.g. evolved nets); called before
    every week from 2 on, playoffs included, regardless of pool."""
    season = {"byid": {p["id"]: p for p in season_data["players"]}}
    n = len(rosters)
    mgmt = mgmt or [dict(DEFAULT_MGMT) for _ in range(n)]
    rosters = [list(r) for r in rosters]
    fa = None
    if pool is not None:
        owned = {p["id"] for r in rosters for p in r}
        fa = [p for p in pool if p["id"] not in owned]
    watched = pool if pool is not None else [p for r in rosters for p in r]
    seen = {}
    LAST["waivers"] = LAST["trades"] = 0

    # circle-method round robin, repeated to fill REG_WEEKS
    teams = list(range(n))
    rounds = []
    for r in range(n - 1):
        pairs = [(teams[i], teams[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]

    wins = [0] * n
    points = [0.0] * n

    def manage(week):
        if week < 2:
            return
        if trade_fn is not None:
            trade_fn(rosters, week, seen, wins, rng)
        if fa is None:
            return
        standing = sorted(range(n), key=lambda t: (wins[t], points[t]))
        do_waivers(rosters, fa, week, seen, mgmt, standing)
        if trade_fn is None and week <= TRADE_DEADLINE:
            do_trades(rosters, seen, mgmt, rng)

    def play(a, b, week):
        """Both sides set matchup-aware lineups; returns actual scores."""
        na = build_lineup(rosters[a], week, seen)[1]
        nb = build_lineup(rosters[b], week, seen)[1]
        la = matchup_lineup(rosters[a], week, seen, nb, mgmt[a]["var_tilt"])
        lb = matchup_lineup(rosters[b], week, seen, na, mgmt[b]["var_tilt"])
        return lineup_actual(la, week, season), lineup_actual(lb, week, season)

    def note_week(week):
        for p in watched:
            a = actual_points(p, week, season)
            s = seen.get(p["id"], (0.0, 0, 0.0))
            seen[p["id"]] = (s[0] + a, s[1] + 1, s[2] + a * a)

    for week in range(1, REG_WEEKS + 1):
        manage(week)
        for a, b in rounds[(week - 1) % len(rounds)]:
            sa, sb = play(a, b, week)
            points[a] += sa
            points[b] += sb
            if sa > sb or (sa == sb and rng.random() < 0.5):
                wins[a] += 1
            else:
                wins[b] += 1
        note_week(week)

    seeds = sorted(range(n), key=lambda t: (-wins[t], -points[t]))
    semi = [(seeds[0], seeds[3]), (seeds[1], seeds[2])]
    manage(SEMI_WEEK)
    finalists = []
    semi_scores = {}
    for a, b in semi:
        sa, sb = play(a, b, SEMI_WEEK)
        semi_scores[a], semi_scores[b] = sa, sb
        finalists.append(a if sa >= sb else b)
    note_week(SEMI_WEEK)

    fa_tot = fb_tot = 0.0
    for w in FINAL_WEEKS:
        manage(w)
        sa, sb = play(finalists[0], finalists[1], w)
        fa_tot += sa
        fb_tot += sb
        note_week(w)
    champ = finalists[0] if fa_tot >= fb_tot else finalists[1]

    return [{"wins": wins[t], "points": round(points[t], 1),
             "made_playoffs": t in seeds[:4], "champion": t == champ}
            for t in range(n)]
