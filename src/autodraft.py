"""Replica of ESPN's autodraft (autopick) for a private 10-team ESPN league.

ESPN's rule, as implemented here:
  1. If your pre-draft queue has an available player, take the top one.
  2. Otherwise take the highest-ranked available player on ESPN's draft board
     (draftRanksByRankType) that fills a position of need - i.e. an unfilled
     starting slot, FLEX included - and that stays under the roster cap.
  3. K and D/ST are deferred until the final rounds.

Usage:
    python fetch_board.py          # refresh ESPN's board (do this before the draft)
    python autodraft.py            # candidates for your next pick
    python autodraft.py --round 3  # look ahead to a specific round
"""
import argparse
import json
import os
import unicodedata

# ---------------------------------------------------------------- league config
# League-specific settings (draft slot, draft order, keepers and the picks keepers consume)
# live in an optional JSON file, never in code: $LEAGUE_CONFIG if set, else
# ./league_config.json if present, else a keeper-free default with me at slot 1.
# See examples/draft/league_config.json (an illustrative, made-up league) for the schema.
# The ESPN league id is read from the environment and only printed.
LEAGUE_ID = os.environ.get("ESPN_LEAGUE_ID", "<ESPN_LEAGUE_ID unset>")


def _load_config():
    path = os.environ.get("LEAGUE_CONFIG") or "league_config.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if os.environ.get("LEAGUE_CONFIG"):
            raise SystemExit(f"LEAGUE_CONFIG={path} not found")
        return {}


_CFG = _load_config()
TEAMS = int(_CFG.get("teams", 10))
ROUNDS = int(_CFG.get("rounds", 16))
SCORING = _CFG.get("scoring", "ppr")          # "ppr" or "std"
MY_SLOT = int(_CFG.get("my_slot", 1))         # my 1st-round slot
# Round-1 order of team keys; "me" is my own team, opponents are anonymous labels.
DRAFT_ORDER = _CFG.get("draft_order") or (
    [f"t{i}" for i in range(1, MY_SLOT)] + ["me"] + [f"t{i}" for i in range(MY_SLOT, TEAMS)])
assert len(DRAFT_ORDER) == TEAMS and DRAFT_ORDER[MY_SLOT - 1] == "me", "draft_order / my_slot mismatch"

# Overall pick numbers consumed by keepers (no selection happens there; they never appear
# in drafted.txt). Leagues where keepers do not cost a pick leave this empty.
KEEPER_PICKS = {int(p) for p in _CFG.get("keeper_picks", [])}
LIVE_PICKS = [p for p in range(1, TEAMS * ROUNDS + 1) if p not in KEEPER_PICKS]

# Starting lineup (ESPN default). FLEX takes RB/WR/TE.
STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "D/ST": 1, "K": 1}
FLEX_POS = {"RB", "WR", "TE"}
POS_CAP = {"QB": 2, "RB": 6, "WR": 7, "TE": 2, "K": 1, "D/ST": 1}

# Pre-draft rosters (kept players, off the board before pick 1), one list per team.
ROSTERS = {t: list(_CFG.get("keepers", {}).get(t, [])) for t in DRAFT_ORDER}

QUEUE = []                  # my pre-draft queue; overrides the board when non-empty
EXCLUDE_POS = {"TQB"}       # team-QB entries: only real in leagues with a TQB slot


# ------------------------------------------------------------------- name lookup
def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return s.lower().replace("'", "").replace(".", "").replace("-", " ").strip()


def resolve(board, query):
    """Board row for a name. Exact match wins; else unique substring match."""
    q = norm(query)
    hits = [r for r in board if norm(r["name"]) == q]
    if not hits:
        hits = [r for r in board if q in norm(r["name"])]
    if not hits:
        raise SystemExit(f"no player matches {query!r}")
    return min(hits, key=lambda r: r[f"rank_{SCORING}"])


# ------------------------------------------------------------------ roster needs
def slots_open(roster_positions):
    """Unfilled starting slots given the positions already on a roster."""
    left = dict(STARTERS)
    flex_pool = []
    for pos in roster_positions:
        if pos in left and left[pos] > 0:
            left[pos] -= 1
        elif pos in FLEX_POS:
            flex_pool.append(pos)
    for _ in flex_pool:
        if left["FLEX"] > 0:
            left["FLEX"] -= 1
    return {k: v for k, v in left.items() if v > 0}


def is_need(pos, open_slots):
    return pos in open_slots or (pos in FLEX_POS and "FLEX" in open_slots)


def eligible(row, roster_positions, picks_left):
    """ESPN autodraft eligibility for one candidate."""
    pos = row["pos"]
    if pos in EXCLUDE_POS:
        return False
    if roster_positions.count(pos) >= POS_CAP.get(pos, 99):
        return False
    open_slots = slots_open(roster_positions)
    must_fill = sum(open_slots.get(p, 0) for p in ("K", "D/ST"))
    if pos in ("K", "D/ST"):
        # only in the endgame, and only if that slot is still empty
        return pos in open_slots and picks_left <= must_fill + 1
    if picks_left <= must_fill:
        return False        # remaining picks are reserved for empty K/D/ST slots
    skill_open = {k: v for k, v in open_slots.items() if k not in ("K", "D/ST")}
    if skill_open and not is_need(pos, skill_open):
        return False        # starters first (K/D/ST slots don't block bench picks)
    return True


# ----------------------------------------------------------------- pick geometry
def my_picks():
    out = []
    for r in range(1, ROUNDS + 1):
        slot = MY_SLOT if r % 2 == 1 else TEAMS - MY_SLOT + 1
        p = (r - 1) * TEAMS + slot
        if p not in KEEPER_PICKS:
            out.append(p)
    return out


def owner_of(pick):
    rnd, idx = divmod(pick - 1, TEAMS)
    order = DRAFT_ORDER if rnd % 2 == 0 else DRAFT_ORDER[::-1]
    return order[idx]


# ------------------------------------------------------------------------ engine
def load_drafted(board):
    rows = []
    try:
        lines = open("drafted.txt").read().splitlines()
    except FileNotFoundError:
        return rows
    for line in lines:
        line = line.split("#")[0].strip()
        if not line:
            continue
        rows.append(resolve(board, line.split("|")[0].strip()))
    return rows


def autopick(board, taken_ids, roster_positions, picks_left):
    for q in QUEUE:
        r = resolve(board, q)
        if r["id"] not in taken_ids and eligible(r, roster_positions, picks_left):
            return r, "queue"
    for r in board:
        if r["id"] not in taken_ids and eligible(r, roster_positions, picks_left):
            return r, "board"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, help="show candidates for this round instead of the next pick")
    ap.add_argument("--depth", type=int, default=12, help="how many candidates to list")
    args = ap.parse_args()

    board = sorted(json.load(open("board_2026.json")), key=lambda r: r[f"rank_{SCORING}"])
    rostered = {resolve(board, n)["id"] for team in ROSTERS.values() for n in team}
    drafted = load_drafted(board)
    taken = rostered | {r["id"] for r in drafted}

    picks = my_picks()
    on_the_clock = LIVE_PICKS[len(drafted)] if len(drafted) < len(LIVE_PICKS) else TEAMS * ROUNDS + 1
    target = picks[args.round - 1] if args.round else next(p for p in picks if p >= on_the_clock)

    my_pos = [resolve(board, n)["pos"] for n in ROSTERS["me"]]
    my_pos += [r["pos"] for p, r in zip(LIVE_PICKS, drafted) if owner_of(p) == "me"]

    print(f"league {LEAGUE_ID} | {TEAMS} teams, snake, slot {MY_SLOT} | scoring={SCORING.upper()}")
    print(f"my picks: {', '.join(str(p) for p in picks[:8])} ...")
    print(f"picks made: {len(drafted)}   on the clock: #{on_the_clock} ({owner_of(on_the_clock)})")
    print(f"my roster: {', '.join(ROSTERS['me'] + [r['name'] for p, r in zip(LIVE_PICKS, drafted) if owner_of(p) == 'me'])}")
    print(f"open starting slots: {slots_open(my_pos) or 'none'}\n")

    # advance the board to my target pick: everyone else takes best available
    sim_taken = set(taken)
    gone = []
    for pick in range(on_the_clock, target):
        if pick in KEEPER_PICKS:
            continue
        for r in board:
            if r["id"] not in sim_taken and r["pos"] not in EXCLUDE_POS | {"K", "D/ST"}:
                sim_taken.add(r["id"])
                gone.append((pick, owner_of(pick), r))
                break
    if gone:
        print(f"projected off the board before #{target} (best-available model):")
        for pick, who, r in gone:
            print(f"  #{pick:3d} {who:4s} {r['name']:24s} {r['pos']:5s} rank {r[f'rank_{SCORING}']}")
        print()

    picks_left = len(picks) - picks.index(target) if target in picks else len(picks)
    pick_choice, source = autopick(board, sim_taken, my_pos, picks_left)
    print(f"=== PICK #{target} (round {picks.index(target) + 1 if target in picks else '?'}) ===")
    if pick_choice:
        print(f"AUTODRAFT TAKES: {pick_choice['name']} ({pick_choice['pos']}, {pick_choice['team']}) "
              f"- via {source}, ESPN rank {pick_choice[f'rank_{SCORING}']}\n")

    print(f"{'espn':>4} {'name':24s} {'pos':5s} {'tm':4s} {'adp':>6s} {'$':>3s} {'own%':>5s}  elig")
    shown = 0
    for r in board:
        if r["id"] in sim_taken or r["pos"] in EXCLUDE_POS:
            continue
        ok = "yes" if eligible(r, my_pos, picks_left) else "no (need/cap)"
        print(f"{r[f'rank_{SCORING}']:4d} {r['name'][:24]:24s} {r['pos']:5s} {r['team']:4s} "
              f"{str(r['adp'])[:6]:>6s} {str(r['auction_ppr']):>3s} {str(r['pct_owned'])[:5]:>5s}  {ok}")
        shown += 1
        if shown >= args.depth:
            break


if __name__ == "__main__":
    main()
