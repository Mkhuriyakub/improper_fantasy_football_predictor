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


def add_qb_components(df, season):
    """Trailing passing-only and rushing-only fantasy points for quarterbacks (walk-forward)."""
    st = pd.read_parquet(ROOT / "data" / f"player_stats_{season}.parquet")
    st = st[st.season_type == "REG"].sort_values(["player_id", "week"]).copy()
    st["pass_fp_g"] = st.passing_yards.fillna(0) * 0.04 + st.passing_tds.fillna(0) * 4 - st.passing_interceptions.fillna(0) * 2 - st.sack_fumbles_lost.fillna(0) * 2
    st["rush_fp_g"] = st.rushing_yards.fillna(0) * 0.1 + st.rushing_tds.fillna(0) * 6 - st.rushing_fumbles_lost.fillna(0) * 2
    g = st.groupby("player_id")
    st["pass_fp_l3"] = g.pass_fp_g.transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    st["pass_fp_season"] = g.pass_fp_g.transform(lambda s: s.shift().expanding().mean())
    st["rush_fp_season"] = g.rush_fp_g.transform(lambda s: s.shift().expanding().mean())
    # for a target week beyond the data, carry the latest values forward
    latest = st.groupby("player_id")[["pass_fp_g", "rush_fp_g"]].apply(lambda x: pd.Series({
        "pass_fp_l3_latest": x.pass_fp_g.tail(3).mean(), "pass_fp_season_latest": x.pass_fp_g.mean(), "rush_fp_season_latest": x.rush_fp_g.mean()}))
    df = df.merge(st[["player_id", "week", "pass_fp_l3", "pass_fp_season", "rush_fp_season"]], on=["player_id", "week"], how="left")
    df = df.merge(latest, left_on="player_id", right_index=True, how="left")
    for c in ["pass_fp_l3", "pass_fp_season", "rush_fp_season"]:
        df[c] = df[c].fillna(df[c + "_latest"])
    df["pass_fp"] = df.fp - df.rush_fp - df.rec_fp
    return df


tr_all, te_all = add_qb_components(tr_all, TRAIN), add_qb_components(te_all, SEASON)


def add_def_tiers(df, season):
    """Opponent pass-defense rating on the absolute, era-adjusted scale (src/def_tiers.py)."""
    t = pd.read_parquet(ROOT / "results" / f"def_tiers_{season}.parquet").rename(columns={"defteam": "opp"})
    df = df.merge(t[["opp", "week", "def_rel_epa", "def_tier"]], on=["opp", "week"], how="left")
    df["def_rel_epa"] = df.def_rel_epa.fillna(0)
    df["def_tier"] = df.def_tier.fillna("avg or below")
    return df


tr_all, te_all = add_def_tiers(tr_all, TRAIN), add_def_tiers(te_all, SEASON)
_QB_PASS_BASE = ["pass_fp_l3", "pass_fp_season", "attempts_l3", "pass_yds_l3", "pass_td_l3", "qb_epa"]
_QB_PASS_TAIL = [
    "off_dropbacks_pg", "off_pass_sr", "off_rz_plays_pg", "def_rel_epa", "opp_def_ypa_allowed", "opp_def_pressure_rate",
    "opp_def_sack_rate", "opp_def_blitz_rate"]  # def_rel_epa: absolute-scale pass D rating (validated 2025: better rank, less bias vs strong D)
QB_RUSH = ["rush_yds_l3", "carries_l3", "rush_fp_l3", "rush_fp_season", "spread", "implied_total", "off_rz_plays_pg",
           "opp_def_rush_epa_allowed", "opp_def_line_yds_allowed"]

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
    if pos == "QB":
        # split model: passing and rushing projected separately and summed (validated 2025: better rank, lower tier bias)
        QB_PASS = _QB_PASS_BASE + COMMON_ENV + _QB_PASS_TAIL
        Xp_a, Xp_b = fill(tr, te, QB_PASS); Xr_a, Xr_b = fill(tr, te, QB_RUSH)
        scp, scr = StandardScaler().fit(Xp_a), StandardScaler().fit(Xr_a)
        mp = Ridge(alpha=60).fit(scp.transform(Xp_a), tr.pass_fp)
        mr = Ridge(alpha=20).fit(scr.transform(Xr_a), tr.rush_fp)
        te["mean_pass"], te["mean_rush"] = mp.predict(scp.transform(Xp_b)), mr.predict(scr.transform(Xr_b))
    # ceiling model: for RBs the opponent's explosive-run susceptibility improves p90 (validated 2025) without moving the mean
    cols_q = cols + (["def_expl_sus"] if pos == "RB" and "def_expl_sus" not in cols else [])
    Xa_q, Xb_q = fill(tr, te, cols_q)
    sc_q = StandardScaler().fit(Xa_q)
    Xa_qs, Xb_qs = sc_q.transform(Xa_q), sc_q.transform(Xb_q)
    q10 = QuantileRegressor(quantile=0.10, alpha=0.01, solver="highs").fit(Xa_s, tr.fp)
    q50 = QuantileRegressor(quantile=0.50, alpha=0.01, solver="highs").fit(Xa_s, tr.fp)
    q90 = QuantileRegressor(quantile=0.90, alpha=0.01, solver="highs").fit(Xa_qs, tr.fp)
    te["mean"] = (te.mean_pass + te.mean_rush) if pos == "QB" else med.predict(Xb_s)
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


def shootout_flag(P):
    """Game-level label: both offenses top-third (pass EPA), spread <= 3.5, neither defense top-7 against the pass.
    2023-25: ~40% of such games reached 63+ combined points vs an 11% base rate (small n)."""
    t = P.drop_duplicates(["team", "week"])[["team", "opp", "week", "off_pass_epa", "opp_def_pass_epa_allowed", "spread"]].copy()
    t["off_rank"] = t.groupby("week").off_pass_epa.rank(ascending=False)
    t["def_rank"] = t.groupby("week").opp_def_pass_epa_allowed.rank(ascending=True)  # opp's defense; 1 = best
    n = t.groupby("week").team.transform("size")
    t["off_top3rd"] = t.off_rank <= n / 3
    t["opp_def_elite"] = t.def_rank <= 7
    o = t[["team", "week", "off_top3rd", "opp_def_elite"]].rename(columns={"team": "opp", "off_top3rd": "opp_off_top3rd", "opp_def_elite": "own_def_elite"})
    t = t.merge(o, on=["opp", "week"], how="left")
    t["shootout"] = (t.off_top3rd & t.opp_off_top3rd.fillna(False) & (t.spread.abs() <= 3.5) & ~t.opp_def_elite & ~t.own_def_elite.fillna(False)).astype(int)
    return P.merge(t[["team", "week", "shootout"]], on=["team", "week"], how="left")


P = shootout_flag(P)
P.to_parquet(ROOT / "results" / f"projections_{SEASON}_wk{WEEK}.parquet", index=False)
print("Walk-forward validation on", TRAIN, "(MAE in PPR points / within-week rank correlation):")
for pos, n, sc, best in report:
    print(f"  {pos} (n={n}): " + " | ".join(f"{k}: {v['MAE']} / {v['rank']}" for k, v in sc.items()) + f"   -> using '{best}'")
pd.set_option("display.width", 250)
P["ceiling_flag"] = np.where((P.position == "RB") & (P.def_expl_tercile == "soft"), "soft run D", "")
cols = ["player", "position", "team", "opp", "p10", "p50", "p90", "mean", "ceiling_flag", "shootout", "def_tier", "contrib_usage", "contrib_environment", "contrib_matchup", "status", "practice"]
if NAMES:
    sel = P[P.player.isin(NAMES)]
else:
    sel = P.sort_values("mean", ascending=False).groupby("position").head(8)
print(sel[cols].round(1).to_string(index=False))
