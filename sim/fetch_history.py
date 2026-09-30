"""Download and ANONYMIZE historical seasons into sim/data/season_YYYY.json.

Sources (both public):
  * Fantasy Football Calculator historical ADP (10-team PPR) - genuine pre-draft
    market knowledge for each season, including per-player ADP stdev and bye.
  * nflverse weekly player stats (fantasy_points_ppr) - used ONLY to score
    completed simulated seasons, never shown to draft policies.

Anonymization: every player name is replaced at ingestion by
POS_sha1(salt+name)[:8], e.g. "RB_1a2b3c4d". Team abbreviations are dropped
(bye week is kept as a number). No real name is ever written to the output
files or printed by this script - only counts and match rates.

Usage:  python fetch_history.py            # all available years
        python fetch_history.py --years 2015 2016
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SALT = "prince-fantasy-2026"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
UA = {"User-Agent": "Mozilla/5.0"}
YEARS = [y for y in range(2010, 2026) if y != 2012]   # FFC has no 10-team PPR ADP for 2012
POS_MAP = {"DEF": "DST", "PK": "K"}
SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm(name):
    s = name.lower().replace("'", "").replace(".", "").replace("-", " ")
    s = SUFFIXES.sub("", s)
    return " ".join(s.split())


def anon(pos, name):
    return f"{pos}_{hashlib.sha1((SALT + norm(name)).encode()).hexdigest()[:8]}"


def get(url, timeout=180):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def fetch_adp(year):
    with get(f"https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=10&year={year}") as r:
        return json.load(r)["players"]


def fetch_weekly(year):
    """norm_name|pos -> {week: ppr_pts} for REG season, plus team -> played weeks."""
    base = "https://github.com/nflverse/nflverse-data/releases/download/"
    try:
        r = get(base + f"player_stats/player_stats_{year}.csv")      # legacy, <=2024
    except urllib.error.HTTPError:
        r = get(base + f"stats_player/stats_player_week_{year}.csv")  # current naming
    with r:
        text = io.TextIOWrapper(r, encoding="utf-8", errors="ignore")
        reader = csv.DictReader(text)
        weekly, team_weeks = {}, {}
        for row in reader:
            if row.get("season_type") != "REG":
                continue
            pos = row.get("position") or ""
            week = int(row["week"])
            team_weeks.setdefault(row.get("recent_team") or row.get("team") or "", set()).add(week)
            if pos not in ("QB", "RB", "WR", "TE"):
                continue
            pts = float(row.get("fantasy_points_ppr") or 0.0)
            key = norm(row.get("player_display_name") or row.get("player_name") or "") + "|" + pos
            wk = weekly.setdefault(key, {})
            wk[week] = wk.get(week, 0.0) + pts
    return weekly, team_weeks


# FFC team abbrevs that differ from nflverse
TEAM_FIX = {"JAC": "JAX", "WSH": "WAS", "LAR": "LA", "KCC": "KC", "TBB": "TB",
            "NEP": "NE", "NOS": "NO", "SFO": "SF", "GBP": "GB"}


def infer_bye(team, team_weeks, max_week):
    tw = team_weeks.get(team) or team_weeks.get(TEAM_FIX.get(team, ""), set())
    if not tw:
        return 0
    missing = [w for w in range(1, min(max_week, 14) + 1) if w not in tw]
    return missing[0] if len(missing) == 1 else 0


def build_year(year):
    adp_rows = fetch_adp(year)
    weekly, team_weeks = fetch_weekly(year)
    max_week = max((w for ws in team_weeks.values() for w in ws), default=17)
    games = 17 if max_week >= 18 else 16

    players, matched, skill = [], 0, 0
    for p in adp_rows:
        pos = POS_MAP.get(p["position"], p["position"])
        if pos not in ("QB", "RB", "WR", "TE", "K", "DST"):
            continue
        bye = p.get("bye") or infer_bye(p.get("team") or "", team_weeks, max_week)
        wk = {}
        if pos in ("QB", "RB", "WR", "TE"):
            skill += 1
            wk = weekly.get(norm(p["name"]) + "|" + pos, {})
            if wk:
                matched += 1
        players.append({
            "id": anon(pos, p["name"]),
            "pos": pos,
            "adp": float(p["adp"]),
            "adp_sd": float(p["stdev"]) if p.get("stdev") else round(float(p["adp"]) / 4, 2),
            "bye": int(bye or 0),
            "weekly": {str(w): round(v, 2) for w, v in sorted(wk.items())},
        })
    players.sort(key=lambda r: r["adp"])
    out = {"year": year, "games": games, "players": players}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, f"season_{year}.json"), "w") as f:
        json.dump(out, f)
    print(f"  {year}: {len(players)} ADP players, skill match {matched}/{skill} "
          f"({100 * matched / max(1, skill):.0f}%), {games}-game season")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=YEARS)
    args = ap.parse_args()
    print(f"building anonymized seasons -> {DATA_DIR}")
    for y in args.years:
        try:
            build_year(y)
        except Exception as e:
            print(f"  {y}: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
