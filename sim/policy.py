"""Draft policies over the anonymized point-in-time view.

GenomePolicy is the backtest port of draftmodel.py's pair-value lookahead:
score(candidate) = value(now) + v2_weight * value(best attainable at my next
pick), where opponents' intervening picks are predicted with an ADP-order
autopick model, and value is slot-aware dynamic VBD shaped by the genome.

AutopickPolicy is the ESPN-autodraft bot: highest ADP-ranked player that
fills a need (the baseline every league opponent runs, and the benchmark the
champion genome must beat in validation).
"""
from functools import lru_cache

from config import TEAMS, ROUNDS, FLEX_POS, POS_CAP, slots_open, eligible


@lru_cache(maxsize=None)
def picks_for_slot(slot):
    out = []
    for pick in range(1, TEAMS * ROUNDS + 1):
        rnd, idx = divmod(pick - 1, TEAMS)
        s = idx if rnd % 2 == 0 else TEAMS - 1 - idx
        if s == slot:
            out.append(pick)
    return tuple(out)


@lru_cache(maxsize=None)
def slot_of_pick(pick):
    rnd, idx = divmod(pick - 1, TEAMS)
    return idx if rnd % 2 == 0 else TEAMS - 1 - idx


def picks_left_at(slot, pick):
    return sum(1 for p in picks_for_slot(slot) if p >= pick)


def pos_list_of(roster):
    return [p["pos"] for p in roster]


def autopick_id(board, taken, pos_list, picks_left, kdst_early=0):
    for r in board:                      # board is preference-sorted (view = ADP order)
        if r["id"] not in taken and eligible(r["pos"], pos_list, picks_left, kdst_early):
            return r
    for r in board:                      # safety: any player under cap
        if r["id"] not in taken and pos_list.count(r["pos"]) < POS_CAP.get(r["pos"], 99):
            return r
    for r in board:                      # pool exhausted at legal positions: take anyone
        if r["id"] not in taken:
            return r
    return None


class AutopickPolicy:
    """ESPN autodraft: best ADP-ranked available filling a need."""

    def pick(self, view, ctx):
        pos_list = pos_list_of(ctx["rosters"][ctx["my_index"]])
        left = picks_left_at(ctx["slot"], ctx["pick"])
        r = autopick_id(view, ctx["taken"], pos_list, left)
        return r["id"] if r else None


class NoisyAutopickPolicy:
    """Human-like drafter: autopick over a personally jittered board. Each
    player's ADP is perturbed by N(0, scale * adp_sd) - adp_sd is the MEASURED
    variance in where real drafters took that player, so scale=1.0 reproduces
    one plausible human's board."""

    def __init__(self, rng, scale=1.0):
        self.rng, self.scale, self.board = rng, scale, None

    def pick(self, view, ctx):
        if self.board is None:
            self.board = sorted(view, key=lambda r: r["adp"]
                                + self.rng.gauss(0.0, self.scale * max(r["adp_sd"], 1.0)))
        pos_list = pos_list_of(ctx["rosters"][ctx["my_index"]])
        left = picks_left_at(ctx["slot"], ctx["pick"])
        r = autopick_id(self.board, ctx["taken"], pos_list, left)
        return r["id"] if r else None


class GenomePolicy:
    def __init__(self, genome):
        self.g = genome

    # -------------------------------------------------- internal opponent model
    def _forward_gone(self, view, ctx, frm, to):
        """Ids predicted taken in picks frm..to-1: opponents = ADP autopick."""
        taken = set(ctx["taken"])
        pos_lists = {}
        gone = set()
        for q in range(frm, to):
            slot = slot_of_pick(q)
            pol_i = ctx["order"][slot]
            pl = pos_lists.get(slot)
            if pl is None:
                pl = pos_lists[slot] = pos_list_of(ctx["rosters"][pol_i])
            r = autopick_id(view, taken, pl, picks_left_at(slot, q))
            if r:
                taken.add(r["id"])
                gone.add(r["id"])
                pl.append(r["pos"])
        return gone

    # -------------------------------------------------- valuation
    def _baselines(self, view, taken, rosters):
        g = self.g
        flex_share = {"RB": g["flex_rb"], "WR": g["flex_wr"], "TE": g["flex_te"]}
        demand = {p: 0.0 for p in POS_CAP}
        for roster in rosters:
            for pos, n in slots_open(pos_list_of(roster)).items():
                if pos == "FLEX":
                    for fp, sh in flex_share.items():
                        demand[fp] += n * sh
                else:
                    demand[pos] += n
        base = {}
        for pos in demand:
            k = max(1, round(demand[pos] * g["base_depth"]))
            seen = 0
            base[pos] = 0.0
            for r in view:
                if r["id"] in taken or r["pos"] != pos:
                    continue
                seen += 1
                if seen > k:
                    base[pos] = r["proj"]
                    break
        return base

    def _slot_value(self, pos, proj, open_now, base, flex_base):
        if open_now.get(pos, 0) > 0:
            v = proj - base[pos]
        elif pos in FLEX_POS and open_now.get("FLEX", 0) > 0:
            v = proj - flex_base
        else:
            v = self.g["bench_beta"] * (proj - base[pos])
        return v + self.g.get(f"bias_{pos}", 0.0)

    # -------------------------------------------------- the pick
    def pick(self, view, ctx):
        g = self.g
        kdst_early = int(round(g["kdst_early"]))
        my = ctx["rosters"][ctx["my_index"]]
        pos_list = pos_list_of(my)
        pick = ctx["pick"]
        left = picks_left_at(ctx["slot"], pick)
        mine = picks_for_slot(ctx["slot"])
        nxt = next((p for p in mine if p > pick), None)

        base = self._baselines(view, ctx["taken"], ctx["rosters"])
        flex_base = max(base[p] for p in FLEX_POS)
        open_now = slots_open(pos_list)

        cands = []
        for r in view:
            if r["id"] in ctx["taken"]:
                continue
            if not eligible(r["pos"], pos_list, left, kdst_early):
                continue
            v1 = self._slot_value(r["pos"], r["proj"], open_now, base, flex_base)
            v1 -= g["reach_pen"] * max(0.0, r["adp"] - pick) / 10.0
            cands.append((v1, r))
        if not cands:
            r = autopick_id(view, ctx["taken"], pos_list, left, kdst_early)
            return r["id"] if r else None
        cands.sort(key=lambda x: -x[0])
        if nxt is None:
            return cands[0][1]["id"]

        gone = self._forward_gone(view, ctx, pick + 1, nxt)
        top2 = {p: [] for p in POS_CAP}
        for r in view:
            if r["id"] in ctx["taken"] or r["id"] in gone:
                continue
            lst = top2[r["pos"]]
            if len(lst) < 2:
                lst.append(r["proj"])

        best_id, best_score = None, None
        for v1, x in cands[:10]:
            pos_list2 = pos_list + [x["pos"]]
            open2 = slots_open(pos_list2)
            skill2 = {k: n for k, n in open2.items() if k not in ("K", "DST")}
            v2 = 0.0
            for pos, projs in top2.items():
                if pos in ("K", "DST") or not projs:
                    continue
                if pos_list2.count(pos) >= POS_CAP.get(pos, 99):
                    continue
                if skill2 and not (pos in skill2 or (pos in FLEX_POS and "FLEX" in skill2)):
                    continue
                proj = projs[0]
                if pos == x["pos"] and x["proj"] >= projs[0]:
                    proj = projs[1] if len(projs) > 1 else 0.0
                v2 = max(v2, self._slot_value(pos, proj, open2, base, flex_base))
            score = v1 + g["v2_weight"] * v2
            if best_score is None or score > best_score:
                best_id, best_score = x["id"], score
        return best_id
