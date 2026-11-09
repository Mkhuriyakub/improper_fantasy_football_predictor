"""Season-average spine + situational adjustments, tested against the season average.

Walk-forward on 2025 (weeks 7-18) and on 2026 live weeks (features_2026_wk*.parquet),
reporting MAE and rank vs. the plain season average and vs. the usage ridge.

Usage: python src/situational.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "src" / "project.py").read_text()
ns = {}
exec(re.search(r"COMMON_ENV = .*?MIN_USAGE = \{[^\n]*\}", src, re.S).group(0), ns)
GROUPS, MIN_USAGE = ns["GROUPS"], ns["MIN_USAGE"]
K_SHRINK = 3  # games of position-average prior on the season mean


def add_situational(df, pos_mean):
    d = df.copy()
    n = d.games_prior.clip(lower=1)
    d["base"] = (n * d.fp_season + K_SHRINK * pos_mean) / (n + K_SHRINK)
    d["streak"] = d.fp_l3 - d.fp_season
    d["star_excess"] = (d.fp_season - 15).clip(lower=0)
    d["dog"] = d.spread.clip(lower=0)
    d["fav"] = (-d.spread).clip(lower=0)
    d["mate_out"] = d.get("lead_back_out", 0) + d.get("lead_receiver_out", 0)
    d["qb_new"] = d.get("new_starter", d.get("qb_changed", 0))
    for c in ["mate_out", "qb_new", "lead_back_out", "lead_receiver_out"]:
        if c in d:
            d[c] = d[c].fillna(0)
    return d


SPECS = {
    "season": None,  # plain fp_season
    "base_only": ["base"],
    "base+streak": ["base", "streak"],
    "base+streak+star": ["base", "streak", "star_excess"],
    "base+streak+star+spread": ["base", "streak", "star_excess", "dog", "fav"],
    "situational": ["base", "streak", "star_excess", "dog", "fav", "mate_out", "qb_new"],
    "situational+itt": ["base", "streak", "star_excess", "dog", "fav", "mate_out", "qb_new", "implied_total"],
}


def fit_predict(a, b, cols, alpha=5):
    med = a[cols].median()
    m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(a[cols].fillna(med), a.fp)
    return m.predict(b[cols].fillna(med))


def prep(f, pos):
    col, mn = MIN_USAGE[pos]
    return f[(f.position == pos) & (f.played == 1) & (f[col] >= mn) & (f.games_prior >= 3)].copy()


def evaluate(train_season, test_frames, label):
    f25 = pd.read_parquet(ROOT / "results" / f"features_{train_season}.parquet")
    rows = []
    for pos in ["RB", "WR", "TE", "QB"]:
        t = prep(f25, pos)
        pos_mean = t.fp.mean()
        t = add_situational(t, pos_mean)
        usage_cols = [c for c in GROUPS[pos]["usage"] if c in t]
        preds = {k: [] for k in list(SPECS) + ["usage"]}
        acts, weeks = [], []
        if test_frames is None:  # walk-forward within training season
            for w in range(7, 19):
                a, b = t[t.week < w], t[t.week == w]
                if b.empty:
                    continue
                for k, cols in SPECS.items():
                    preds[k].append(b.fp_season.values if cols is None else fit_predict(a, b, cols))
                preds["usage"].append(fit_predict(a, b, usage_cols, alpha=20))
                acts.append(b.fp.values); weeks.append(b.week.values)
        else:  # fit on full training season, test on live frames
            a = t
            for b in test_frames:
                b = prep(b, pos)
                b = b[b.fp.notna()]
                b = add_situational(b, pos_mean)
                if b.empty:
                    continue
                for k, cols in SPECS.items():
                    preds[k].append(b.fp_season.values if cols is None else fit_predict(a, b, cols))
                preds["usage"].append(fit_predict(a, b, usage_cols, alpha=20))
                acts.append(b.fp.values); weeks.append(b.week.values)
        act = np.concatenate(acts); wk = np.concatenate(weeks)
        for k, v in preds.items():
            p = np.concatenate(v)
            df = pd.DataFrame(dict(p=p, a=act, w=wk))
            rows.append(dict(set=label, pos=pos, spec=k, n=len(df), mae=np.abs(p - act).mean(),
                             rank=df.groupby("w").apply(lambda g: spearmanr(g.p, g.a).correlation if len(g) > 4 else np.nan).mean(),
                             bias=(act - p).mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    r25 = evaluate(2025, None, "2025 wf")
    live = []
    for wk in [7, 8, 9]:
        p = ROOT / "results" / f"features_2026_wk{wk}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            if d.fp.isna().all():  # fill actuals from the stats file
                st = pd.read_parquet(ROOT / "data" / "player_stats_2026.parquet")
                st = st[(st.season_type == "REG") & (st.week == wk)].set_index("player_id")
                d["fp"] = d.player_id.map(st.fantasy_points_ppr)
                d["played"] = d.fp.notna().astype(int)
            d = d[~((d.week == 9) & d.team.isin(["BUF", "MIN"]))]
            live.append(d)
    r26 = evaluate(2025, live, "2026 live")
    out = pd.concat([r25, r26])
    for s, g in out.groupby("set"):
        print(f"\n== {s}")
        print(g.pivot(index="spec", columns="pos", values="mae").round(3).loc[list(SPECS) + ["usage"]])
        print(g.pivot(index="spec", columns="pos", values="rank").round(3).loc[list(SPECS) + ["usage"]])
    out.to_csv(ROOT / "results" / "situational_eval.csv", index=False)
