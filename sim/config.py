"""League configuration for the backtest environment - mirrors my private ESPN league
(10-team PPR snake, 16 rounds, ESPN default lineup), WITHOUT keepers.

Season structure is uniform across eras so 16-game and 17-game NFL seasons are
comparable: 13-week regular season, semifinals week 14 (seeds 1v4, 2v3),
final = cumulative weeks 15+16.
"""
TEAMS = 10
ROUNDS = 16
REG_WEEKS = 13
SEMI_WEEK = 14
FINAL_WEEKS = (15, 16)

STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
FLEX_POS = ("RB", "WR", "TE")
POS_CAP = {"QB": 2, "RB": 6, "WR": 7, "TE": 2, "K": 1, "DST": 1}
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DST")

# K and DST score this constant every week: they carry no draft skill, but the
# roster slot (and the opportunity cost of drafting them early) stays real.
K_WEEKLY = 7.5
DST_WEEKLY = 7.0


def slots_open(pos_list):
    """Unfilled starting slots given the positions already on a roster."""
    left = dict(STARTERS)
    flex_extra = 0
    for pos in pos_list:
        if left.get(pos, 0) > 0:
            left[pos] -= 1
        elif pos in FLEX_POS:
            flex_extra += 1
    for _ in range(flex_extra):
        if left["FLEX"] > 0:
            left["FLEX"] -= 1
    return {k: v for k, v in left.items() if v > 0}


def eligible(pos, pos_list, picks_left, kdst_early=0):
    """ESPN-autopick eligibility: fill starters first, defer K/DST to the endgame,
    reserve the final picks for empty K/DST slots. kdst_early relaxes the deferral
    by that many picks (an evolvable gene)."""
    if pos_list.count(pos) >= POS_CAP.get(pos, 99):
        return False
    open_slots = slots_open(pos_list)
    must_fill = sum(open_slots.get(p, 0) for p in ("K", "DST"))
    if pos in ("K", "DST"):
        return pos in open_slots and picks_left <= must_fill + 1 + kdst_early
    if picks_left <= must_fill:
        return False
    skill_open = {k: n for k, n in open_slots.items() if k not in ("K", "DST")}
    if skill_open and not (pos in skill_open or (pos in FLEX_POS and "FLEX" in skill_open)):
        return False
    return True
