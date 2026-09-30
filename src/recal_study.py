"""Can we beat the consensus weekly projection? Backtest + 2026 recalibration.

Benchmark: Sleeper's own weekly projection (frozen per week, public, 2023-2025 history).
Question:  does adding (a) a calibration fit and (b) trailing VOLUME (targets, carries,
           red-zone opportunities, snap share) beat it out of sample?

  M0  Sleeper projection as-is
  M1  a + b*proj                          (calibration only)
  M2  ridge on proj + EWMA volume/points + last-season ppg   (per position)
  M3  gradient boosting on the same features (non-linearity check), if xgboost present

Folds: train 2023 -> test 2024;  train 2023-24 -> test 2025.  Rows = player-weeks with a
projection >= 2 and the player active (gp >= 1). Horizons: next week (y1) and mean of the
next 4 played weeks (y4; benchmark for y4 is still proj_w, which is what a rest-of-season
user has at time w).

  python recal_study.py                 # backtest report
  python recal_study.py --apply         # + fit on 2023-25, recalibrate 2026 -> weekly_proj_2026_recal.json
"""
import argparse
import json
import math
import os
import urllib.request

import numpy as np

# Local cache of Sleeper public JSON: projections_{season}_{week}.json, stats_{season}_{week}.json
# and players.json. Override with the SLEEPER_CACHE environment variable.
CACHE = os.environ.get("SLEEPER_CACHE", os.path.join("data", "sleeper_cache"))
POS = ("QB", "RB", "WR", "TE")
HALF_LIFE = 3.0
ALPHA = 1 - 0.5 ** (1 / HALF_LIFE)
FEATS = ["proj", "ewm_pts", "ewm_tgt", "ewm_rush", "ewm_rz", "ewm_snap", "ewm_pass",
         "n_prior", "prior_ppg", "has_prior", "gap", "ewm_resid"]


def load(kind, season, week):
    p = os.path.join(CACHE, f"{kind}_{season}_{week}.json")
    if not os.path.exists(p):
        return []
    return json.load(open(p, encoding="utf-8"))


def players_master():
    pm = json.load(open(os.path.join(CACHE, "players.json"), encoding="utf-8"))
    return {pid: {"name": v.get("full_name") or "", "pos": v.get("position"),
                  "espn": v.get("espn_id"), "age": v.get("age"), "team": v.get("team")}
            for pid, v in pm.items()}


def season_panel(season, pm, weeks=range(1, 18)):
    """{pid: {week: row}} with actuals, usage and the frozen weekly projection."""
    out = {}
    for w in weeks:
        for r in load("projections", season, w):
            pid = r.get("player_id"); s = r.get("stats") or {}
            if pid not in pm or pm[pid]["pos"] not in POS or s.get("pts_ppr") is None:
                continue
            out.setdefault(pid, {}).setdefault(w, {})["proj"] = float(s["pts_ppr"])
        for r in load("stats", season, w):
            pid = r.get("player_id"); s = r.get("stats") or {}
            if pid not in pm or pm[pid]["pos"] not in POS:
                continue
            row = out.setdefault(pid, {}).setdefault(w, {})
            row["pts"] = float(s.get("pts_ppr") or 0.0)
            row["played"] = (s.get("gp") or 0) >= 1 or (s.get("off_snp") or 0) > 0
            row["tgt"] = float(s.get("rec_tgt") or 0.0)
            row["rush"] = float(s.get("rush_att") or 0.0)
            row["rz"] = float((s.get("rush_rz_att") or 0) + (s.get("rec_rz_tgt") or 0)
                              + (s.get("pass_rz_att") or 0))
            row["pass"] = float(s.get("pass_att") or 0.0)
            tm = s.get("tm_off_snp") or 0
            row["snap"] = float(s.get("off_snp") or 0) / tm if tm else float("nan")
    return out


def prior_ppg(panel_prev):
    out = {}
    for pid, wk in panel_prev.items():
        g = [r["pts"] for r in wk.values() if r.get("played")]
        if len(g) >= 4:
            out[pid] = (sum(g) / len(g), len(g))
    return out


def build_rows(season, panel, prior, pm):
    """One row per (player, week) with features from weeks < w only."""
    rows = []
    for pid, wk in panel.items():
        pos = pm[pid]["pos"]
        ewm = {k: None for k in ("pts", "tgt", "rush", "rz", "snap", "pass", "resid")}
        n_prior, last_played = 0, None
        pp = prior.get(pid)
        for w in range(1, 18):
            r = wk.get(w)
            if r is not None and "pts" in r:
                r["resid"] = (r["pts"] - r["proj"]) if "proj" in r else None
            if r is not None and "proj" in r and r["proj"] >= 2.0 and "pts" in r:
                f = {"season": season, "week": w, "pid": pid, "pos": pos,
                     "proj": r["proj"], "y1": r["pts"], "played": bool(r["played"]),
                     "n_prior": n_prior,
                     "gap": 0 if last_played is None else (w - last_played),
                     "prior_ppg": pp[0] if pp else float("nan"), "has_prior": 1.0 if pp else 0.0}
                for k in ewm:
                    f["ewm_" + k] = ewm[k] if ewm[k] is not None else float("nan")
                # 4-week-ahead target: mean of next 4 PLAYED weeks incl. w
                fut = [wk[v]["pts"] for v in range(w, min(17, w + 3) + 1)
                       if v in wk and wk[v].get("played")]
                f["y4"] = sum(fut) / len(fut) if len(fut) >= 2 else float("nan")
                rest = [wk[v]["pts"] for v in range(w, 18) if v in wk and wk[v].get("played")]
                f["yrest"] = sum(rest) / len(rest) if len(rest) >= 4 else float("nan")
                rows.append(f)
            # update state with week w (available from week w+1 on)
            if r is not None and "pts" in r and r["played"]:
                for k in ewm:
                    v = r[k]
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    ewm[k] = v if ewm[k] is None else (1 - ALPHA) * ewm[k] + ALPHA * v
                n_prior += 1
                last_played = w
    return rows


# ------------------------------------------------------------------ models
def design(rows, pos_means):
    X = []
    for r in rows:
        x = []
        for k in FEATS:
            v = r[k]
            if isinstance(v, float) and math.isnan(v):
                v = pos_means[r["pos"]][k]
            x.append(v)
        X.append(x)
    return np.array(X, float)


def pos_feature_means(rows):
    out = {}
    for pos in POS:
        sub = [r for r in rows if r["pos"] == pos]
        out[pos] = {}
        for k in FEATS:
            vals = [r[k] for r in sub if not (isinstance(r[k], float) and math.isnan(r[k]))]
            out[pos][k] = sum(vals) / len(vals) if vals else 0.0
    return out


def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    A = Z1.T @ Z1 + lam * np.diag([1.0] * Z.shape[1] + [0.0])
    beta = np.linalg.solve(A, Z1.T @ y)
    return {"mu": mu, "sd": sd, "beta": beta}


def ridge_pred(m, X):
    Z = (X - m["mu"]) / m["sd"]
    return np.hstack([Z, np.ones((len(Z), 1))]) @ m["beta"]


def fit_models(train, target, pos_means, use_xgb):
    models = {}
    for pos in POS:
        sub = [r for r in train if r["pos"] == pos and r["played"]
               and not math.isnan(r[target])]
        X = design(sub, pos_means); y = np.array([r[target] for r in sub])
        m1 = ridge_fit(X[:, :1], y, lam=0.0)
        m2 = ridge_fit(X, y, lam=3.0)
        ks = [FEATS.index(k) for k in ("proj", "ewm_snap", "ewm_resid", "n_prior")]
        m2s = ridge_fit(X[:, ks], y, lam=3.0)
        m3 = None
        if use_xgb:
            import xgboost as xgb
            m3 = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                                  reg_lambda=5.0, verbosity=0)
            m3.fit(X, y)
        models[pos] = (m1, m2, m3, m2s)
    return models


def predict(models, rows, pos_means):
    out = {"M0": [], "M1": [], "M2": [], "M2s": [], "M3": []}
    ks = [FEATS.index(k) for k in ("proj", "ewm_snap", "ewm_resid", "n_prior")]
    for r in rows:
        m1, m2, m3, m2s = models[r["pos"]]
        X = design([r], pos_means)
        out["M0"].append(r["proj"])
        out["M1"].append(float(ridge_pred(m1, X[:, :1])[0]))
        out["M2"].append(float(ridge_pred(m2, X)[0]))
        out["M2s"].append(float(ridge_pred(m2s, X[:, ks])[0]))
        out["M3"].append(float(m3.predict(X)[0]) if m3 is not None else float("nan"))
    return {k: np.array(v) for k, v in out.items()}


def report(test, preds, target, label):
    y = np.array([r[target] for r in test])
    weeks = np.array([r["season"] * 100 + r["week"] for r in test])
    pos = np.array([r["pos"] for r in test])
    proj = np.array([r["proj"] for r in test])
    print(f"\n--- {label}: n={len(test):,} player-weeks ---")
    # calibration of the benchmark itself
    b = np.polyfit(proj, y, 1)
    slopes = " ".join(f"{q}:{np.polyfit(proj[pos == q], y[pos == q], 1)[0]:.3f}" for q in POS)
    print(f"benchmark calibration: actual = {b[1]:.2f} + {b[0]:.3f} * proj  | by pos {slopes}")
    base_se = (preds["M0"] - y) ** 2
    print(f"{'model':6s} {'RMSE':>7s} {'MAE':>7s} {'dRMSE%':>7s} {'t(wk)':>6s} | " +
          " ".join(f"{p:>8s}" for p in POS) + f" | {'proj>=10':>9s}")
    for k in ("M0", "M1", "M2", "M2s", "M3"):
        p = preds[k]
        if np.isnan(p).all():
            continue
        se = (p - y) ** 2
        rmse = math.sqrt(se.mean()); mae = np.abs(p - y).mean()
        d = (rmse / math.sqrt(base_se.mean()) - 1) * 100
        # week-level paired test on mean squared error
        diffs = [(se[weeks == w].mean() - base_se[weeks == w].mean()) for w in np.unique(weeks)]
        t = -np.mean(diffs) / (np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 2 and np.std(diffs) > 0 else 0.0
        bypos = " ".join(f"{math.sqrt(se[pos == q].mean()):8.3f}" for q in POS)
        top = proj >= 10
        print(f"{k:6s} {rmse:7.3f} {mae:7.3f} {d:+6.2f}% {t:6.2f} | {bypos} | {math.sqrt(se[top].mean()):9.3f}")
    return y


def coef_table(models, pos_means):
    print("\nM2 standardised coefficients (points per 1-sd of feature):")
    print(f"{'pos':4s} " + " ".join(f"{k[:9]:>9s}" for k in FEATS) + f" {'const':>7s}")
    for pos in POS:
        beta = models[pos][1]["beta"]
        print(f"{pos:4s} " + " ".join(f"{b:9.3f}" for b in beta[:-1]) + f" {beta[-1]:7.2f}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-xgb", action="store_true")
    ap.add_argument("--watch", default="", help="comma-separated players to print in the --apply table")
    args = ap.parse_args()
    try:
        import xgboost  # noqa
        use_xgb = not args.no_xgb
    except ImportError:
        use_xgb = False
    pm = players_master()
    panels = {s: season_panel(s, pm) for s in (2022, 2023, 2024, 2025)}
    rows = {}
    for s in (2023, 2024, 2025):
        rows[s] = build_rows(s, panels[s], prior_ppg(panels[s - 1]), pm)
        print(f"{s}: {len(rows[s]):,} rows, {sum(r['played'] for r in rows[s]):,} played")

    for target in ("y1", "y4", "yrest"):
        for train_s, test_s in (((2023,), 2024), ((2023, 2024), 2025)):
            train = [r for s in train_s for r in rows[s]]
            test = [r for r in rows[test_s] if r["played"] and not math.isnan(r[target])]
            pmn = pos_feature_means(train)
            models = fit_models(train, target, pmn, use_xgb)
            preds = predict(models, test, pmn)
            report(test, preds, target, f"target {target} | train {train_s} -> test {test_s}")
            if target == "y1":
                early = [i for i, r in enumerate(test) if 2 <= r["week"] <= 5]
                report([test[i] for i in early], {k: v[early] for k, v in preds.items()}, target,
                       f"   subset weeks 2-5 only (projections still anchored on August)")
            if target == "yrest" and test_s == 2025:
                coef_table(models, pmn)

    if args.apply:
        apply_2026(pm, panels, rows, use_xgb, {n.strip() for n in args.watch.split(",") if n.strip()})


def apply_2026(pm, panels, rows, use_xgb, watch=frozenset()):
    """Fit on 2023-25 (y1 and y4), build 2026 features from weeks 1-2, recalibrate the
    ESPN/Sleeper blend for weeks 3-17 and write weekly_proj_2026_recal.json."""
    from inseason_proj import OUT as ESPN_FILE
    train = [r for s in (2023, 2024, 2025) for r in rows[s]]
    pmn = pos_feature_means(train)
    m_y1 = fit_models(train, "y1", pmn, use_xgb)
    m_y4 = fit_models(train, "yrest", pmn, use_xgb)
    ks = [FEATS.index(k) for k in ("proj", "ewm_snap", "ewm_resid", "n_prior")]
    print("\nM2s coefficients (standardised; proj, snap share, resid, n_prior, const):")
    for pos in POS:
        b1 = m_y1[pos][3]["beta"]; br = m_y4[pos][3]["beta"]
        print(f"  {pos:3s} next-week {np.round(b1, 3)}   rest-of-season {np.round(br, 3)}")
    # implied calibration slopes of the pooled fit, for the record
    for pos in POS:
        sub = [r for r in train if r["pos"] == pos and r["played"] and not math.isnan(r["yrest"])]
        pr = np.array([r["proj"] for r in sub]); yy = np.array([r["yrest"] for r in sub])
        print(f"  pooled 2023-25 RoS calibration {pos}: slope {np.polyfit(pr, yy, 1)[0]:.3f}")
    me_ids = set(json.load(open("league_state.json"))["rosters"]["me"])
    # 2026 panel: stats weeks 1-2, projections weeks 1-17 (download missing future weeks)
    for w in range(3, 18):
        p = os.path.join(CACHE, f"projections_2026_{w}.json")
        if not os.path.exists(p):
            url = (f"https://api.sleeper.com/projections/nfl/2026/{w}?season_type=regular"
                   "&position[]=QB&position[]=RB&position[]=WR&position[]=TE&order_by=pts_ppr")
            try:
                open(p, "wb").write(urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read())
            except Exception as e:
                print(f"sleeper 2026 wk{w} projection fetch failed: {e}")
    panel26 = season_panel(2026, pm)
    prior = prior_ppg(panels[2025])
    espn = json.load(open(ESPN_FILE, encoding="utf-8"))
    import unicodedata, re as _re
    def norm(s):
        s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
        s = _re.sub(r"[.'\-]", " ", s)
        s = _re.sub("(^| )(jr|sr|ii|iii|iv|v)( |$)", " ", s)
        return " ".join(s.split())
    by_espn, by_name = {}, {}
    for pid, info in pm.items():
        if info["espn"]:
            by_espn[int(info["espn"])] = pid
        if info["pos"] in POS and info["team"]:
            by_name.setdefault((norm(info["name"]), info["pos"]), []).append(pid)
    def lookup(p):
        pid = by_espn.get(p["id"])
        if pid:
            return pid
        c = by_name.get((norm(p["name"]), p["pos"]), [])
        if len(c) > 1:
            c = [x for x in c if pm[x]["team"] == p["team"]] or c
        return c[0] if len(c) == 1 else None
    rostered = {i for ids in json.load(open("league_state.json"))["rosters"].values() for i in ids}
    unmapped = [p["name"] for p in espn["players"] if p["pos"] in POS and p["id"] in rostered and lookup(p) is None]
    print(f"rostered skill players without a Sleeper match: {unmapped}")
    cur = 2                      # last completed week
    changed, table, mine = 0, [], []
    for p in espn["players"]:
        if p["pos"] not in POS:
            continue
        pid = lookup(p)
        if pid is None:
            continue
        wk = panel26.get(pid, {})
        # state after weeks 1..cur
        ewm = {k: None for k in ("pts", "tgt", "rush", "rz", "snap", "pass", "resid")}
        n_prior, last_played = 0, None
        for w in range(1, cur + 1):
            r = wk.get(w)
            if r and "pts" in r and r["played"]:
                r["resid"] = (r["pts"] - r["proj"]) if "proj" in r else None
                for k in ewm:
                    v = r[k]
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    ewm[k] = v if ewm[k] is None else (1 - ALPHA) * ewm[k] + ALPHA * v
                n_prior += 1; last_played = w
        pp = prior.get(pid)
        base_ros = sum(p["wk"][str(w)] for w in range(cur + 1, 18))
        new_wk = dict(p["wk"])
        recal_ros = 0.0
        for w in range(cur + 1, 18):
            e = p["wk"][str(w)]
            if e <= 0:                      # bye / out: keep zero
                continue
            s = (wk.get(w) or {}).get("proj")
            blend = (e + s) / 2 if (s is not None and s > 0) else e
            f = {"pos": p["pos"], "proj": blend, "n_prior": n_prior,
                 "gap": 0 if last_played is None else (cur + 1 - last_played),
                 "prior_ppg": pp[0] if pp else float("nan"), "has_prior": 1.0 if pp else 0.0}
            for k in ewm:
                f["ewm_" + k] = ewm[k] if ewm[k] is not None else float("nan")
            X = design([f], pmn)
            # next week uses the 1-week model, later weeks the rest-of-season model
            m = m_y1[p["pos"]] if w == cur + 1 else m_y4[p["pos"]]
            pred = max(0.0, float(ridge_pred(m[3], X[:, ks])[0]))
            new_wk[str(w)] = round(pred, 2)
            recal_ros += pred
        p["wk_espn"] = p["wk"]; p["wk"] = new_wk
        p["ros_espn"], p["ros_recal"] = round(base_ros, 1), round(recal_ros, 1)
        p["usage"] = {"n": n_prior, **{k: (round(v, 2) if v is not None else None) for k, v in ewm.items()}}
        changed += 1
        sl_ros = sum((wk.get(w) or {}).get("proj") or 0 for w in range(cur + 1, 18))
        p["ros_sleeper"] = round(sl_ros, 1)
        if base_ros >= 60:
            table.append((p["name"], p["pos"], p["team"], base_ros, recal_ros, p["usage"]))
        if p["id"] in me_ids or p["name"] in watch:
            mine.append((p["id"] in me_ids, p["name"], p["pos"], p["team"], base_ros, sl_ros, recal_ros,
                         p["wk_espn"][str(cur + 1)], new_wk[str(cur + 1)], p["usage"]))
    json.dump(espn, open("weekly_proj_2026_recal.json", "w"), indent=0)
    print(f"\nwrote weekly_proj_2026_recal.json ({changed} players recalibrated for weeks {cur + 1}-17)")
    table.sort(key=lambda t: -(t[4] - t[3]))
    print("\nbiggest UPGRADES vs ESPN RoS (proj>=60):")
    for n, pos, tm, b, r, u in table[:15]:
        print(f"  {n:24s} {pos:3s} {tm:4s} espn {b:6.1f} -> recal {r:6.1f} ({r - b:+6.1f})  snap {u['snap']} tgt {u['tgt']} rush {u['rush']} rz {u['rz']} n={u['n']}")
    print("\nbiggest DOWNGRADES:")
    for n, pos, tm, b, r, u in table[-15:]:
        print(f"  {n:24s} {pos:3s} {tm:4s} espn {b:6.1f} -> recal {r:6.1f} ({r - b:+6.1f})  snap {u['snap']} tgt {u['tgt']} rush {u['rush']} rz {u['rz']} n={u['n']}")


if __name__ == "__main__":
    main()
