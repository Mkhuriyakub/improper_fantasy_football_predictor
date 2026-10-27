"""Fit per-position models on a training season (walk-forward validated) and
project a target week of another season. Outputs median, 10th/90th percentile
and the contribution of each feature group.

Usage: python src/project.py 2025 2026 7 [player names...]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
TRAIN, SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
NAMES = sys.argv[4:]
ROOT = Path(__file__).resolve().parents[1]
tr_all = pd.read_parquet(ROOT / "results" / f"features_{TRAIN}.parquet")
te_all = pd.read_parquet(ROOT / "results" / f"features_{SEASON}_wk{WEEK}.parquet")


def add_expl(df, season):
    d = pd.read_parquet(ROOT / "results" / f"expl_def_{season}.parquet").rename(columns={"defteam": "opp"})
    r = pd.read_parquet(ROOT / "results" / f"expl_rb_{season}.parquet").rename(columns={"rusher_player_id": "player_id"})
    df = df.merge(d, on=["opp", "week"], how="left").merge(r, on=["player_id", "week"], how="left")
    df["def_expl_sus"] = df.def_expl_sus.fillna(df.def_expl_sus.median())
    df["def_expl_tercile"] = df.def_expl_tercile.fillna("mid")
    return df


tr_all, te_all = add_expl(tr_all, TRAIN), add_expl(te_all, SEASON)


def add_qb(df, season):
    q = pd.read_parquet(ROOT / "results" / f"qb_traits_{season}.parquet")
    df = df.merge(q[["team", "week", "qb_id", "qb_name", "qb_epa", "qb_expl", "qb_changed"]], on=["team", "week"], how="left")
    df["new_starter"] = df.qb_changed.fillna(0).astype(int)  # starter differs from 3 weeks earlier (what the trailing window reflects)
    return df


tr_all, te_all = add_qb(tr_all, TRAIN), add_qb(te_all, SEASON)

COMMON_ENV = ["spread", "implied_total", "total_line", "dome", "off_plays_pg", "off_proe", "off_plays_per_drive", "off_pass_epa",
              "opp_off_plays_pg", "opp_off_plays_per_drive", "opp_def_three_out_rate", "opp_def_plays_allowed_pg", "pace_mismatch"]
GROUPS = {
    "RB": {
        "usage": ["fp_l3", "fp_season", "touches_l3", "carries_l3", "targets_l3", "touch_share_l3", "gl_share", "rz_share",
                  "is_lead_back", "lead_back_out", "script_sens", "script_risk"],
        "environment": COMMON_ENV + ["off_neutral_run_rate", "opp_off_run_rate", "opp_def_blitz_rate", "blitz_x_targets"],
        "matchup": ["def_expl_sus"],
    },
    "WR": {
        "usage": ["fp_l3", "fp_season", "targets_l3", "rec_l3", "rec_yds_l3", "target_share_l3", "air_share_l3", "adot_l3", "rz_share",
                  "is_lead_receiver", "lead_receiver_out", "carries_l3", "new_starter"],
        "environment": COMMON_ENV + ["off_dropbacks_pg", "off_pass_sr"],
        "matchup": ["opp_def_pass_epa_allowed", "opp_def_ypa_allowed", "opp_def_pressure_rate", "opp_def_blitz_rate", "pass_edge"],
    },
    "QB": {
        "usage": ["fp_l3", "fp_season", "attempts_l3", "pass_yds_l3", "pass_td_l3", "rush_yds_l3", "carries_l3"],
        "environment": COMMON_ENV + ["off_dropbacks_pg", "off_pass_sr", "off_rz_plays_pg"],
        "matchup": ["opp_def_pass_epa_allowed", "opp_def_ypa_allowed", "opp_def_pressure_rate", "opp_def_sack_rate", "opp_def_blitz_rate", "pass_edge"],
    },
}
GROUPS["TE"] = {**GROUPS["WR"], "matchup": GROUPS["WR"]["matchup"] + ["opp_def_te_target_share"]}
MIN_USAGE = {"RB": ("touches_l3", 6), "WR": ("targets_l3", 3), "TE": ("targets_l3", 3), "QB": ("attempts_l3", 15)}


def prep(df, pos):
    col, mn = MIN_USAGE[pos]
    d = df[(df.position == pos) & (df[col] >= mn) & (df.status != "Out") & (df.weeks_since_last <= 3)].copy()
    return d


def fill(train, test, cols):
    med = train[cols].median()
    return train[cols].fillna(med), test[cols].fillna(med)


report = []
projections = []
for pos, grp in GROUPS.items():
    tr = prep(tr_all, pos)
    tr = tr[tr.played == 1]
    all_cols = grp["usage"] + grp["environment"] + grp["matchup"]
    ladder = {"naive": ["fp_l3"], "usage": grp["usage"], "usage+env": grp["usage"] + grp["environment"], "full": all_cols}
    # ---- walk-forward validation on the training season
    scores = {}
    for name, cols in ladder.items():
        out = []
        for w in range(7, 19):
            a, b = tr[tr.week < w], tr[tr.week == w].copy()
            if b.empty:
                continue
            Xa, Xb = fill(a, b, cols)
            b["pred"] = b.fp_l3 if name == "naive" else make_pipeline(StandardScaler(), Ridge(alpha=20)).fit(Xa, a.fp).predict(Xb)
            out.append(b)
        P = pd.concat(out)
        scores[name] = dict(MAE=round((P.pred - P.fp).abs().mean(), 2),
                            rank=round(P.groupby("week").apply(lambda g: spearmanr(g.pred, g.fp).correlation, include_groups=False).mean(), 3))
    best = min([k for k in scores if k != "naive"], key=lambda k: scores[k]["MAE"])
    report.append((pos, len(tr), scores, best))
    # ---- final fit on full training season with the best feature set, plus quantiles
    cols = ladder[best]
    te = prep(te_all, pos)
    Xa, Xb = fill(tr, te, cols)
    sc = StandardScaler().fit(Xa)
    Xa_s, Xb_s = sc.transform(Xa), sc.transform(Xb)
    med = Ridge(alpha=20).fit(Xa_s, tr.fp)
    # ceiling model: for RBs the opponent's explosive-run susceptibility improves p90 (validated 2025) without moving the mean
    cols_q = cols + (["def_expl_sus"] if pos == "RB" and "def_expl_sus" not in cols else [])
    Xa_q, Xb_q = fill(tr, te, cols_q)
    sc_q = StandardScaler().fit(Xa_q)
    Xa_qs, Xb_qs = sc_q.transform(Xa_q), sc_q.transform(Xb_q)
    q10 = QuantileRegressor(quantile=0.10, alpha=0.01, solver="highs").fit(Xa_s, tr.fp)
    q50 = QuantileRegressor(quantile=0.50, alpha=0.01, solver="highs").fit(Xa_s, tr.fp)
    q90 = QuantileRegressor(quantile=0.90, alpha=0.01, solver="highs").fit(Xa_qs, tr.fp)
    te["mean"] = med.predict(Xb_s)
    te["p10"], te["p50"], te["p90"] = q10.predict(Xb_s), q50.predict(Xb_s), q90.predict(Xb_qs)
    # group contributions: sum of (standardized feature * coef) by group, relative to an average player
    contrib = pd.DataFrame(Xb_s * med.coef_, columns=cols, index=te.index)
    for gname, gcols in grp.items():
        gc = [c for c in gcols if c in cols]
        te["contrib_" + gname] = contrib[gc].sum(axis=1) if gc else 0.0
    te["baseline"] = med.intercept_
    te["model_used"] = best
    projections.append(te)

P = pd.concat(projections)
P.to_parquet(ROOT / "results" / f"projections_{SEASON}_wk{WEEK}.parquet", index=False)
print("Walk-forward validation on", TRAIN, "(MAE in PPR points / within-week rank correlation):")
for pos, n, sc, best in report:
    print(f"  {pos} (n={n}): " + " | ".join(f"{k}: {v['MAE']} / {v['rank']}" for k, v in sc.items()) + f"   -> using '{best}'")
pd.set_option("display.width", 250)
P["ceiling_flag"] = np.where((P.position == "RB") & (P.def_expl_tercile == "soft"), "soft run D", "")
cols = ["player", "position", "team", "opp", "p10", "p50", "p90", "mean", "ceiling_flag", "contrib_usage", "contrib_environment", "contrib_matchup", "status", "practice"]
if NAMES:
    sel = P[P.player.isin(NAMES)]
else:
    sel = P.sort_values("mean", ascending=False).groupby("position").head(8)
print(sel[cols].round(1).to_string(index=False))
