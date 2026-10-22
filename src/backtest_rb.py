"""Walk-forward backtest of the RB model ladder.

For each week w (7..18): fit each model on rows from weeks < w, predict week w.
Models are ridge regressions (point) plus logistic models for P(smash) and
P(bust). Scored on MAE, within-week rank correlation, and smash/bust
discrimination (AUC) and calibration.

Usage: python src/backtest_rb.py 2025
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
ROOT = Path(__file__).resolve().parents[1]
f = pd.read_parquet(ROOT / "results" / f"features_rb_{SEASON}.parquet")

# universe: fantasy-relevant backs who were not ruled out
f = f[(f.touches_l3 >= 6) & (f.status != "Out") & (f.weeks_since_last <= 3)].copy()
f["smash"] = (f.fp >= 20).astype(int)
f["bust"] = (f.fp < 8).astype(int)
f = f.fillna({"off_light_box_ypc_gain": 0, "box_x_light_gain": 0, "all_share": f.all_share.median(),
              "trail_share": f.neutral_share.median(), "neutral_share": f.neutral_share.median()})

LADDER = {
    "1_naive": ["fp_l3"],
    "2_vegas": ["fp_l3", "touches_l3", "implied_total", "spread"],
    "3_usage": ["fp_l3", "touches_l3", "targets_l3", "touch_share_l3", "gl_share", "is_lead_back", "lead_back_out",
                "implied_total", "spread", "script_sens", "script_risk", "dome",
                "off_plays_pg", "off_neutral_run_rate", "off_proe", "off_plays_per_drive",
                "opp_off_plays_pg", "opp_off_run_rate", "opp_off_plays_per_drive",
                "opp_def_three_out_rate", "opp_def_plays_allowed_pg", "opp_def_blitz_rate", "blitz_x_targets",
                "pace_mismatch"],
}
LADDER["4_full"] = LADDER["3_usage"] + [
    "off_line_yds", "off_stuff", "opp_def_line_yds_allowed", "opp_def_stuff_rate", "opp_def_rush_epa_allowed",
    "opp_def_explosive_run_allowed", "opp_def_box_vs_one_back", "opp_def_light_box_rate",
    "off_light_box_ypc_gain", "box_x_light_gain", "line_edge", "opp_def_rb_target_share", "opp_def_man_rate",
    "opp_def_pressure_rate"]

TEST_WEEKS = range(7, 19)
preds = []
for w in TEST_WEEKS:
    tr = f[(f.week < w) & f.played.eq(1)]
    te = f[(f.week == w) & f.played.eq(1)].copy()
    if te.empty:
        continue
    for name, cols in LADDER.items():
        X, Xt = tr[cols].values, te[cols].values
        if name == "1_naive":
            te[f"pred_{name}"] = te.fp_l3
            # naive smash/bust probs: empirical rate among players with similar fp_l3 (bins)
            bins = pd.qcut(tr.fp_l3, 5, duplicates="drop")
            sm = tr.groupby(bins, observed=True).smash.mean()
            bu = tr.groupby(bins, observed=True).bust.mean()
            cut = pd.cut(te.fp_l3, bins.cat.categories)
            te[f"psmash_{name}"] = cut.map(sm).astype(float).fillna(tr.smash.mean())
            te[f"pbust_{name}"] = cut.map(bu).astype(float).fillna(tr.bust.mean())
            continue
        reg = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(X, tr.fp)
        te[f"pred_{name}"] = reg.predict(Xt)
        for tgt in ["smash", "bust"]:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(X, tr[tgt])
            te[f"p{tgt}_{name}"] = clf.predict_proba(Xt)[:, 1]
    preds.append(te)
P = pd.concat(preds)
P.to_parquet(ROOT / "results" / f"backtest_rb_preds_{SEASON}.parquet", index=False)


def score(P, name):
    p = P[f"pred_{name}"]
    mae = (p - P.fp).abs().mean()
    rank = P.groupby("week").apply(lambda g: spearmanr(g[f"pred_{name}"], g.fp).correlation, include_groups=False).mean()
    auc_s = roc_auc_score(P.smash, P[f"psmash_{name}"])
    auc_b = roc_auc_score(P.bust, P[f"pbust_{name}"])
    # calibration: predicted vs actual smash rate in top-decile of predicted smash prob
    top = P[P[f"psmash_{name}"] >= P[f"psmash_{name}"].quantile(0.9)]
    topb = P[P[f"pbust_{name}"] >= P[f"pbust_{name}"].quantile(0.9)]
    return dict(model=name, MAE=round(mae, 2), rank_corr=round(rank, 3), smash_AUC=round(auc_s, 3),
                bust_AUC=round(auc_b, 3), top10pct_smash_pred=round(top[f"psmash_{name}"].mean(), 2),
                top10pct_smash_actual=round(top.smash.mean(), 2), top10pct_bust_pred=round(topb[f"pbust_{name}"].mean(), 2),
                top10pct_bust_actual=round(topb.bust.mean(), 2))


res = pd.DataFrame([score(P, n) for n in LADDER])
res.to_csv(ROOT / "results" / f"backtest_rb_scores_{SEASON}.csv", index=False)
print(f"test rows {len(P)}, weeks {P.week.min()}-{P.week.max()}, base rates: smash {P.smash.mean():.2f} bust {P.bust.mean():.2f}")
print(res.to_string(index=False))

# lead backs only (the decisions people actually make)
L = P[P.is_lead_back == 1]
print(f"\nLead backs only ({len(L)} rows, smash {L.smash.mean():.2f} bust {L.bust.mean():.2f}):")
print(pd.DataFrame([score(L, n) for n in LADDER]).to_string(index=False))

# which features carry weight in the full model (final fit on all data)
tr = f[f.played.eq(1)]
cols = LADDER["4_full"]
reg = make_pipeline(StandardScaler(), Ridge(alpha=5.0)).fit(tr[cols], tr.fp)
coef = pd.Series(reg[-1].coef_, index=cols).sort_values(key=abs, ascending=False)
print("\nFull-model standardized coefficients (points per 1 SD):")
print(coef.round(2).to_string())
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(tr[cols], tr.bust)
print("\nBust-model coefficients (log-odds per 1 SD):")
print(pd.Series(clf[-1].coef_[0], index=cols).sort_values(key=abs, ascending=False).round(2).head(12).to_string())
