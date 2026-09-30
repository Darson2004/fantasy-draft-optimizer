"""Live-draft turn helper: validate + append new picks, then run the recommender.

  python turn.py "Player A" "Player B"             # picks made since last update, in order
  python turn.py                                   # nothing new - just recommend my pick
  python turn.py --undo                            # remove the last recorded pick (typo fix)

All names are validated BEFORE anything is written (one bad name aborts the whole
batch). Every append echoes pick#, owner, and the resolved player - eyeball it.
Keeper-reserved slots are skipped by the engine; never enter keepers here.
"""
import os
import subprocess
import sys

from autodraft import LIVE_PICKS, ROSTERS, SCORING, load_drafted, norm
from draftmodel import load_board, load_pick_owners, pick_owner


def main():
    args = sys.argv[1:]
    board = load_board()

    if args and args[0] == "--undo":
        lines = open("drafted.txt").read().splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].split("#")[0].strip():
                print(f"removing: {lines[i].strip()}")
                del lines[i]
                break
        else:
            print("nothing to undo")
        open("drafted.txt", "w").write("\n".join(lines) + "\n")
        args = args[1:]

    already = load_drafted(board)
    ov = load_pick_owners()

    if args:
        keeper_ids = set()
        for names in ROSTERS.values():
            for n in names:
                q = norm(n)
                hits = [r for r in board if norm(r["name"]) == q] or \
                       [r for r in board if q in norm(r["name"])]
                if hits:
                    keeper_ids.add(min(hits, key=lambda r: r[f"rank_{SCORING}"])["id"])

        taken = {r["id"] for r in already} | keeper_ids
        rows = []
        for q in args:                                   # validate everything first
            qn = norm(q)
            hits = [r for r in board if norm(r["name"]) == qn] or \
                   [r for r in board if qn in norm(r["name"])]
            if not hits:
                raise SystemExit(f"ABORT (nothing written): no player matches {q!r}")
            best = min(hits, key=lambda r: r[f"rank_{SCORING}"])
            if len(hits) > 1:
                others = ", ".join(r["name"] for r in hits[:4] if r["id"] != best["id"])
                print(f"  note: {q!r} matched {len(hits)} -> {best['name']} (not: {others})")
            if best["id"] in taken:
                raise SystemExit(f"ABORT (nothing written): {best['name']} is already "
                                 f"drafted or kept - typo?")
            taken.add(best["id"])
            rows.append(best)
        if len(already) + len(rows) > len(LIVE_PICKS):
            raise SystemExit("ABORT: more picks than live draft slots")

        with open("drafted.txt", "a") as f:              # then write the whole batch
            for j, r in enumerate(rows):
                p = LIVE_PICKS[len(already) + j]
                f.write(r["name"] + "\n")
                print(f"  recorded #{p:3d} {pick_owner(p, ov):4s} {r['name']} ({r['pos']})")

    sys.stdout.flush()                                   # echo lines before model output
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "draftmodel.py")])


if __name__ == "__main__":
    main()
