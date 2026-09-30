"""Pull ESPN's live draft board into board_2026.json.

`draftRanksByRankType` is the same overall rank ESPN shows as "Best Available" in the
draft room, and it is what their autodraft consumes. `ownership.averageDraftPosition`
is real-world ADP (what humans actually do) and is kept alongside it for comparison.

The league itself is private, so this pulls the public league-agnostic
player endpoint.

Also attached per player (consumed by draftmodel.py):
  proj      - season full-PPR projection: mean of ESPN and Sleeper where both exist.
              ESPN's is computed from the raw projected stat line (statSourceId=1)
              because ESPN's own appliedTotal is glitched for some players.
  bye       - 2026 bye week (0 if unknown)
"""
import json
import unicodedata
import urllib.request

SEASON = 2026
URL = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}/players?view=kona_player_info"
FILTER = json.dumps({"players": {"limit": 2000,
                                 "sortDraftRanks": {"sortPriority": 1, "sortAsc": True, "value": "PPR"}}})

POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 15: "TQB", 16: "D/ST"}
TEAM = {0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
        9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA", 16: "MIN",
        17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC",
        25: "SF", 26: "SEA", 27: "TB", 28: "WSH", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU"}


# ESPN raw projected-stat ids -> full-PPR points
PPR_STAT = {"3": 0.04, "4": 4.0, "19": 2.0, "20": -2.0,      # pass yds/TD/2pt/INT
            "24": 0.1, "25": 6.0, "26": 2.0,                 # rush yds/TD/2pt
            "42": 0.1, "43": 6.0, "44": 2.0, "53": 1.0,      # rec yds/TD/2pt/receptions
            "72": -2.0}                                      # fumbles lost

SL_TEAM = {"WSH": "WAS"}  # ESPN -> Sleeper team abbrev differences


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return s.lower().replace("'", "").replace(".", "").replace("-", " ").strip()


def fetch():
    req = urllib.request.Request(URL, headers={"x-fantasy-filter": FILTER,
                                               "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def fetch_byes():
    url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{SEASON}?view=proTeamSchedules_wl"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        info = json.load(r)
    return {t["id"]: t.get("byeWeek", 0) for t in info["settings"]["proTeams"]}


def fetch_sleeper():
    """Sleeper season PPR projections: ({norm-name|pos: pts}, {team-abbrev: DEF pts})."""
    url = (f"https://api.sleeper.com/projections/nfl/{SEASON}?season_type=regular"
           "&position[]=QB&position[]=RB&position[]=WR&position[]=TE&position[]=K&position[]=DEF"
           "&order_by=ppr")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = json.load(r)
    skill, dst = {}, {}
    for row in rows:
        pts = (row.get("stats") or {}).get("pts_ppr")
        pl = row.get("player") or {}
        if pts is None:
            continue
        if pl.get("position") == "DEF":
            dst[row.get("player_id")] = pts
        else:
            key = norm(f"{pl.get('first_name', '')} {pl.get('last_name', '')}") + "|" + (pl.get("position") or "")
            skill[key] = max(pts, skill.get(key, 0))
    return skill, dst


def espn_projection(p):
    """Full-PPR season points from ESPN's raw projected stat line (appliedTotal is unreliable)."""
    for s in p.get("stats", []):
        if s.get("seasonId") == SEASON and s.get("statSourceId") == 1 and s.get("statSplitTypeId") == 0:
            raw = s.get("stats") or {}
            pts = sum(w * raw.get(k, 0.0) for k, w in PPR_STAT.items())
            return round(pts, 1) if pts > 5 else None
    return None


def slim(raw, byes, sl_skill, sl_dst):
    rows = []
    for p in raw:
        ranks = p.get("draftRanksByRankType")
        if not ranks:
            continue
        own = p.get("ownership") or {}
        pos = POS.get(p["defaultPositionId"], str(p["defaultPositionId"]))
        team = TEAM.get(p["proTeamId"], str(p["proTeamId"]))
        name = p.get("fullName", "")
        pe = espn_projection(p) if pos in ("QB", "RB", "WR", "TE") else None
        if pos == "D/ST":
            ps = sl_dst.get(SL_TEAM.get(team, team))
        else:
            ps = sl_skill.get(norm(name) + "|" + pos)
        proj = round(sum(v for v in (pe, ps) if v) / max(1, len([v for v in (pe, ps) if v])), 1)
        rows.append({
            "id": p["id"],
            "name": name,
            "pos": pos,
            "team": team,
            "rank_ppr": ranks.get("PPR", {}).get("rank", 9999),
            "rank_std": ranks.get("STANDARD", {}).get("rank", 9999),
            "auction_ppr": ranks.get("PPR", {}).get("auctionValue"),
            "adp": own.get("averageDraftPosition"),
            "pct_owned": own.get("percentOwned"),
            "injury": p.get("injuryStatus"),
            "proj": proj or None,
            "proj_espn": pe,
            "proj_sl": ps,
            "bye": byes.get(p["proTeamId"], 0),
        })
    rows.sort(key=lambda r: r["rank_ppr"])

    # impute missing projections from same-position rank neighbours (skill) or a
    # flat endgame default (K/D/ST beyond Sleeper's coverage)
    for pos in ("QB", "RB", "WR", "TE", "K", "D/ST"):
        at_pos = [r for r in rows if r["pos"] == pos]
        for i, r in enumerate(at_pos):
            if r["proj"]:
                continue
            neigh = [x["proj"] for x in at_pos[max(0, i - 2):i + 3] if x["proj"]]
            r["proj"] = round(sum(neigh) / len(neigh), 1) if neigh else 0.0
    return rows


if __name__ == "__main__":
    byes = fetch_byes()
    try:
        sl_skill, sl_dst = fetch_sleeper()
        print(f"sleeper projections: {len(sl_skill)} skill, {len(sl_dst)} D/ST")
    except Exception as e:
        print(f"sleeper fetch failed ({e}); using ESPN projections only")
        sl_skill, sl_dst = {}, {}
    rows = slim(fetch(), byes, sl_skill, sl_dst)
    with open("board_2026.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote board_2026.json: {len(rows)} ranked players")
    for r in rows[:10]:
        print(f"  {r['rank_ppr']:3d} {r['name']:24s} {r['pos']:5s} {r['team']:4s} "
              f"adp={r['adp']} proj={r['proj']} bye={r['bye']}")
