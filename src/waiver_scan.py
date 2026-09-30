"""Waiver scan: value every free agent / waiver player by what he adds to MY season,
not by raw points. For each candidate: pick the bench drop that maximises my
deterministic expected wins (real schedule, weeks cur..14), then re-score the best
add/drop under injury + projection-error scenarios (paired) so bench depth counts.
Raw ESPN weekly projections (no calibration shrink).

  python waiver_scan.py                                   # top adds overall + by position
  python waiver_scan.py --protect "Player A,Player B"     # never consider dropping these
"""
import argparse, json, math, sys
import numpy as np
from trades2 import (load, lineup, fa_best_table, phi, ros, TEAM_SD, HAZARD, SPELL_P,
                     SEASON_END_P, PROJ_ERR_SD, LAST_WEEK, ME)

ap = argparse.ArgumentParser()
ap.add_argument("--protect", default="", help="comma-separated players never considered as drops")
ap.add_argument("--sims", type=int, default=400, help="injury / projection-error scenarios")
args = ap.parse_args()
PROTECT = {n.strip() for n in args.protect.split(",") if n.strip()}
S = args.sims
st, players, rosters, teams, fa = load(do_shrink=False)
cur, reg = st["current_week"], st["settings"]["reg_weeks"]
weeks = list(range(cur, LAST_WEEK + 1))
fa_best = fa_best_table(fa)
ir = set(st["ir"][ME])
mine = [p for p in rosters[ME] if p["id"] not in ir]
drops = [p for p in mine if p["name"] not in PROTECT and p["pos"] in ("QB", "RB", "WR", "TE")]
try:
    wv = json.load(open("waivers_now.json"))
except FileNotFoundError:
    wv = {}
cands = sorted([p for p in fa if p["pos"] in ("QB", "RB", "WR", "TE")], key=lambda p: -ros(p, cur))[:40]
for p in fa:
    if p["name"] in wv and p not in cands:
        cands.append(p)
opp_of = {}
for w in range(cur, reg + 1):
    for h, a in st["schedule"][str(w)]:
        if h == ME: opp_of[w] = a
        if a == ME: opp_of[w] = h
opp_mus = {t: np.array([lineup(rows, w, None, fa_best) if w >= cur else 0 for w in range(LAST_WEEK + 1)])
           for t, rows in rosters.items() if t != ME}
sd = TEAM_SD * math.sqrt(2)

def det_wins(mus):
    return sum(phi((mus[w] - opp_mus[opp_of[w]][w]) / sd) for w in range(cur, reg + 1))

def mus_of(rows, mult=None):
    m = np.zeros(LAST_WEEK + 1)
    for w in weeks:
        m[w] = lineup(rows, w, mult, fa_best)
    return m

base_mus = mus_of(mine)
base_w = det_wins(base_mus)
base_po = base_mus[15:18].sum()
print(f"week {cur}: my deterministic E[reg-season wins] {base_w:.2f}; playoff-weeks lineup pts {base_po:.0f}")
print("cost of dropping each bench player alone (det dW):", end=" ")
for d in drops:
    print(f"{d['name'].split()[-1]} {det_wins(mus_of([p for p in mine if p is not d])) - base_w:+.3f}", end="; ")
print()

# ---- injury / projection-error scenarios, shared across candidates (paired)
rng = np.random.default_rng(7)
def scenario(pool):
    mult = {}
    for p in pool:
        f = max(0.2, rng.normal(1.0, PROJ_ERR_SD.get(p["pos"], 0.2)))
        m = np.full(LAST_WEEK + 1, f)
        w = cur
        while w <= LAST_WEEK:
            if rng.random() < HAZARD.get(p["pos"], 0.0):
                L = LAST_WEEK if rng.random() < SEASON_END_P else rng.geometric(SPELL_P)
                m[w:min(LAST_WEEK, w + L - 1) + 1] = 0.0
                w += L
            else:
                w += 1
        mult[p["id"]] = m
    return mult
pool = mine + cands
scen = [scenario(pool) for _ in range(S)]
base_mc = np.mean([det_wins(mus_of(mine, m)) for m in scen])

rows_out = []
for c in cands:
    best = None
    for d in drops:
        var = [p for p in mine if p is not d] + [c]
        dw = det_wins(mus_of(var)) - base_w
        if best is None or dw > best[0]:
            best = (dw, d, var)
    dw0 = best[0]
    pick = None
    for d in drops:
        var = [p for p in mine if p is not d] + [c]
        dw = det_wins(mus_of(var)) - base_w
        if dw < dw0 - 0.005:
            continue
        dmc = np.mean([det_wins(mus_of(var, m)) for m in scen]) - base_mc
        if pick is None or dmc > pick[0]:
            pick = (dmc, dw, mus_of(var)[15:18].sum() - base_po, c, d)
    rows_out.append(pick)
rows_out.sort(key=lambda r: -r[0])
def show(rs, title):
    print(f"\n=== {title} ===")
    print(f"{'candidate':22s} {'pos':4s} {'tm':4s} {'st':6s} {'own':>4s} {'inj':5s} {'wk3':>5s} {'RoS/g':>5s} | {'drop':14s} {'dW det':>7s} {'dW inj':>7s} {'dPO pts':>7s}")
    for dmc, dw, dpo, c, d in rs:
        g = max(1, sum(1 for w in range(cur, LAST_WEEK + 1) if c["wk"][w] > 0))
        stt = "WAIVER" if c["name"] in wv else "FA"
        print(f"{c['name'][:22]:22s} {c['pos']:4s} {c['team']:4s} {stt:6s} {c['own']:4.0f} {c['inj'][:5]:5s} {c['wk'][cur]:5.1f} {ros(c, cur)/g:5.1f} | {d['name'][:14]:14s} {dw:+7.3f} {dmc:+7.3f} {dpo:+7.1f}")
show(rows_out[:18], "top adds by injury-aware dWins (best drop shown)")
for pos in ("WR", "RB", "TE", "QB"):
    show([r for r in rows_out if r[3]["pos"] == pos][:6], f"best {pos} adds")
