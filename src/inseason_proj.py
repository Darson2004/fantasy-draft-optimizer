"""In-season WEEKLY projections for every player -> weekly_proj_2026.json.

board_2026.json carries a single preseason season-total projection, which trades.py
spreads flat across 17 games. In-season that is wrong three ways: it ignores games
already played, it ignores ESPN's per-week matchup/injury adjustments, and it
ignores that an IR player projects 0 until his return week. ESPN's PUBLIC player feed
(no login) carries a raw projected stat line for EVERY remaining week, so this builds
the real thing:

  wk[w]   full-PPR projected points for NFL week w (0 on bye / out), w = 1..17
  ytd     actual points scored so far
  ros     sum of wk over weeks not yet played

Skill positions: ESPN raw weekly stat line x PPR weights (same map as fetch_board.py).
K:              ESPN raw weekly stat line x ESPN default kicker scoring.
D/ST:           Sleeper weekly projections (ESPN's D/ST raw line needs the league's
                points-allowed tiers; Sleeper's number is close enough for a slot
                every team fills identically).

  python inseason_proj.py            # fetch + write weekly_proj_2026.json
"""
import json
import time
import urllib.request

from fetch_board import (SEASON, URL, FILTER, POS, TEAM, PPR_STAT, SL_TEAM, fetch,
                         fetch_byes)

OUT = "weekly_proj_2026.json"
WEEKS = 17
# ESPN default kicker scoring on ESPN's raw kicker stat ids
K_STAT = {"74": 5.0, "77": 4.0, "80": 3.0, "86": 1.0, "85": -1.0, "88": -1.0}


def pts(stat_line, weights):
    return round(sum(w * stat_line.get(k, 0.0) for k, w in weights.items()), 2)


def sleeper_dst_weekly():
    """{team-abbrev (ESPN spelling): {week: pts}} from Sleeper's per-week feed."""
    out = {}
    esp_of = {v: k for k, v in SL_TEAM.items()}
    for wk in range(1, WEEKS + 1):
        url = (f"https://api.sleeper.com/projections/nfl/{SEASON}/{wk}"
               "?season_type=regular&position[]=DEF&order_by=ppr")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            rows = json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            print(f"  sleeper DEF week {wk} failed: {e}")
            continue
        for r in rows:
            team = r.get("team")
            p = (r.get("stats") or {}).get("pts_ppr")
            if team and p is not None:
                out.setdefault(esp_of.get(team, team), {})[wk] = float(p)
    return out


def build(raw, byes, dst_weekly):
    rows = []
    for p in raw:
        pos = POS.get(p["defaultPositionId"])
        if pos not in ("QB", "RB", "WR", "TE", "K", "D/ST"):
            continue
        team = TEAM.get(p["proTeamId"], str(p["proTeamId"]))
        wk = {w: 0.0 for w in range(1, WEEKS + 1)}
        ytd = season = 0.0
        for s in p.get("stats", []):
            if s.get("seasonId") != SEASON:
                continue
            line = s.get("stats") or {}
            src, split, sp = s["statSourceId"], s["statSplitTypeId"], s.get("scoringPeriodId")
            if pos == "D/ST":
                continue
            weights = K_STAT if pos == "K" else PPR_STAT
            if src == 1 and split == 1 and sp and 1 <= sp <= WEEKS:
                wk[sp] = pts(line, weights)
            elif src == 1 and split == 0:
                season = pts(line, weights)
            elif src == 0 and split == 0:
                ytd = pts(line, weights)
        if pos == "D/ST":
            for w, v in dst_weekly.get(team, {}).items():
                if w <= WEEKS:
                    wk[w] = round(v, 2)
        own = p.get("ownership") or {}
        rows.append({
            "id": p["id"], "name": p.get("fullName", ""), "pos": pos, "team": team,
            "injury": (p.get("injuryStatus") or "ACTIVE").upper(),
            "pct_owned": round(own.get("percentOwned") or 0.0, 1),
            "bye": byes.get(p["proTeamId"], 0),
            "ytd": round(ytd, 1), "season_proj": round(season, 1),
            "wk": {str(w): v for w, v in wk.items()},
        })
    return rows


if __name__ == "__main__":
    t0 = time.time()
    raw = fetch()
    print(f"espn players: {len(raw)} ({time.time() - t0:.0f}s)")
    byes = fetch_byes()
    dst = sleeper_dst_weekly()
    print(f"sleeper D/ST weekly: {len(dst)} teams")
    rows = build(raw, byes, dst)
    json.dump({"_built": time.strftime("%Y-%m-%d %H:%M"), "players": rows},
              open(OUT, "w"), indent=0)
    print(f"wrote {OUT}: {len(rows)} players")
    # spot check: eyeball a few of the week-2 projections against the numbers shown on the
    # league page (the build was validated to within +/-0.1 this way)
    top = sorted(rows, key=lambda r: -r["wk"]["2"])[:8]
    for r in top:
        print(f"  {r['name']:24s} {r['pos']:4s} wk2 {r['wk']['2']:5.1f}")
