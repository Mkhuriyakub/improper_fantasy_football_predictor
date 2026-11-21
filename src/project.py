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
import nflreadpy as nfl
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
    if "qb_override" in df:  # manual: a new starter named this week (data/overrides_{season}.csv)
        df["new_starter"] = np.maximum(df.new_starter, df.qb_override.fillna(0).astype(int))
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


def add_pedigree(df, season):
    """Draft capital, experience and prior-season efficiency (src/pedigree.py). Validated 2024+2025 for RB/WR/TE:
    lowers MAE on top+emerging players and removes most of the over-projection of backs thrust into volume."""
    p = pd.read_parquet(ROOT / "results" / f"pedigree_{season}.parquet")
    return df.merge(p, on="player_id", how="left")


tr_all, te_all = add_pedigree(tr_all, TRAIN), add_pedigree(te_all, SEASON)


def add_def_effect(df, season):
    """Each defense's individual, walk-forward effect on opposing QBs vs their own averages (src/def_effect.py),
    plus its blitz and pressure rates. Applied structurally (not learned): DEF_C points per EPA/dropback of effect."""
    d = pd.read_parquet(ROOT / "results" / f"def_effect_{season}.parquet").rename(columns={"defteam": "opp"})
    df = df.merge(d[["opp", "week", "def_qb_delta", "def_blitz", "def_pressure"]], on=["opp", "week"], how="left")
    df["def_qb_delta"] = df.def_qb_delta.fillna(0)
    return df


tr_all, te_all = add_def_effect(tr_all, TRAIN), add_def_effect(te_all, SEASON)


def add_cpoe(df, season):
    """Walk-forward season-to-date completion % over expected (Next Gen Stats), attempt-weighted, prior weeks only.
    Tested 2024+2025 vs ESPN QBR, passer rating, NGS aggressiveness/time-to-throw, success rate: CPOE is the only rating
    that improves the QB passing model beyond the QB's own EPA (MAE 5.95->5.93, 5.99->5.94 on top+emerging)."""
    f = ROOT / "data" / "ngs_passing.parquet"
    n = pd.read_parquet(f) if f.exists() else nfl.load_nextgen_stats([season], stat_type="passing").to_pandas()
    n = n[(n.season == season) & (n.week > 0)].sort_values("week").copy()
    n["x"] = n.completion_percentage_above_expectation * n.attempts
    cum_x = n.groupby("player_gsis_id").x.cumsum() - n.x
    cum_a = n.groupby("player_gsis_id").attempts.cumsum() - n.attempts
    n["cpoe_std"] = cum_x / (cum_a + 40)  # 40-attempt prior at 0
    n = n[["player_gsis_id", "week", "cpoe_std"]].rename(columns={"player_gsis_id": "player_id"})
    # a QB's value for week W is the row for his most recent game before W; carry forward through byes/misses
    full = pd.MultiIndex.from_product([n.player_id.unique(), range(1, 19)], names=["player_id", "week"]).to_frame(index=False)
    n = full.merge(n, how="left").sort_values(["player_id", "week"])
    n["cpoe_std"] = n.groupby("player_id").cpoe_std.ffill()
    out = df.merge(n, on=["player_id", "week"], how="left")
    out["cpoe_std"] = out.cpoe_std.fillna(0)
    return out


tr_all, te_all = add_cpoe(tr_all, TRAIN), add_cpoe(te_all, SEASON)
# Validated 2024+2025 on top+emerging: c=10 lowers MAE for WR/TE/RB in 5 of 6 season-position cells and removes most of
# the -1 to -2.5 under-performance of pass catchers against the toughest 15% of defenses. Ridge learns it at half strength.
DEF_C = {"WR": 10.0, "TE": 10.0, "RB": 10.0, "QB": 10.0}

# Effect of a listed injury on an active top-N player, points vs projection, 2024-25 (n in comments), shrunk n/(n+30).
INJURY_ADJ = {"illness": -1.6, "calf": -1.0, "wrist": -1.2, "hamstring": -1.4, "shoulder": -1.2, "concussion": -0.7, "back": -0.7,
              "toe": -0.6, "knee": -0.4, "groin": -0.1, "ankle": 0.0}
INJURY_ADJ_POS = {("RB", "ankle"): -0.7, ("RB", "knee"): -0.8, ("QB", "knee"): -0.9, ("QB", "shoulder"): -1.0, ("WR", "hamstring"): -1.4}


def injury_adjust(injury, pos):
    if not isinstance(injury, str) or not injury:
        return 0.0
    key = injury.lower()
    for k in ("illness", "calf", "wrist", "hamstring", "shoulder", "concussion", "back", "toe", "knee", "groin", "ankle"):
        if k in key:
            return INJURY_ADJ_POS.get((pos, k), INJURY_ADJ[k])
    return 0.0
PEDIGREE = ["top64", "log_pick", "exp", "prior_fp_per_opp", "prior_opp_pg"]
_QB_PASS_BASE = ["pass_fp_l3", "pass_fp_season", "attempts_l3", "pass_yds_l3", "pass_td_l3", "qb_epa", "cpoe_std"]
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
                  "is_lead_back", "lead_back_out", "script_sens", "script_risk"] + PEDIGREE,
        "environment": COMMON_ENV + ["off_neutral_run_rate", "opp_off_run_rate", "opp_def_blitz_rate", "blitz_x_targets"],
        "matchup": ["def_expl_sus"],
    },
    "WR": {
        "usage": ["fp_l3", "fp_season", "targets_l3", "rec_l3", "rec_yds_l3", "target_share_l3", "air_share_l3", "adot_l3", "rz_share",
                  "is_lead_receiver", "lead_receiver_out", "carries_l3", "new_starter"] + PEDIGREE,
        "environment": COMMON_ENV + ["off_dropbacks_pg", "off_pass_sr"],
        "matchup": ["opp_def_pass_epa_allowed", "opp_def_ypa_allowed", "opp_def_pressure_rate", "opp_def_blitz_rate", "pass_edge"],
    },
    "QB": {
        "usage": ["fp_l3", "fp_season", "attempts_l3", "pass_yds_l3", "pass_td_l3", "rush_yds_l3", "carries_l3"],
        "environment": COMMON_ENV + ["off_dropbacks_pg", "off_pass_sr", "off_rz_plays_pg"],
        "matchup": ["opp_def_pass_epa_allowed", "opp_def_ypa_allowed", "opp_def_pressure_rate", "opp_def_sack_rate", "opp_def_blitz_rate", "pass_edge"],
    },
}
GROUPS["TE"] = {**GROUPS["WR"], "usage": GROUPS["WR"]["usage"] + ["snap_last"],  # snap share helps TEs only (2024+25)
                "matchup": GROUPS["WR"]["matchup"] + ["opp_def_te_target_share"]}
MIN_USAGE = {"RB": ("touches_l3", 6), "WR": ("targets_l3", 3), "TE": ("targets_l3", 3), "QB": ("attempts_l3", 15)}
TOPN = {"RB": 24, "WR": 36, "TE": 12, "QB": 16}  # validation is scored on these plus emerging players


def prep(df, pos, returning=False):
    """Training rows: players who played within the last 3 weeks. Prediction rows (returning=True) also keep players back
    from a 3-10 week absence; they are projected from their pre-absence usage and discounted (RETURN_DISCOUNT)."""
    col, mn = MIN_USAGE[pos]
    d = df[(df.position == pos) & (df[col] >= mn) & (df.status != "Out") & (df.weeks_since_last <= (10 if returning else 3))].copy()
    return d


# First game back after missing 3+ weeks, 2019-25 (n=218, prior 4-game avg >= 8 PPR): points vs prior average.
# RB 0.80, WR 0.85, TE 0.75, QB 0.65 for 3-4 missed weeks; 0.65 for 5+ at every position. Opportunities fall similarly.
RETURN_DISCOUNT = {"RB": 0.80, "WR": 0.85, "TE": 0.75, "QB": 0.65}


def return_factor(pos, missed):
    if missed < 3:
        return 1.0
    return RETURN_DISCOUNT[pos] if missed <= 4 else 0.65


def fill(train, test, cols):
    med = train[cols].median()
    return train[cols].fillna(med), test[cols].fillna(med)


report = []
projections = []
for pos, grp in GROUPS.items():
    tr = prep(tr_all, pos)
    tr = tr[tr.played == 1]
    all_cols = grp["usage"] + grp["environment"] + grp["matchup"]
    ladder = {"naive": ["fp_l3"], "season": ["fp_season"], "usage": grp["usage"], "usage+env": grp["usage"] + grp["environment"], "full": all_cols}
    # ---- walk-forward validation on the training season
    scores = {}
    for name, cols in ladder.items():
        out = []
        for w in range(7, 19):
            a, b = tr[tr.week < w], tr[tr.week == w].copy()
            if b.empty:
                continue
            Xa, Xb = fill(a, b, cols)
            b["pred"] = (b.fp_l3 if name == "naive" else b.fp_season if name == "season"
                         else make_pipeline(StandardScaler(), Ridge(alpha=20)).fit(Xa, a.fp).predict(Xb) + DEF_C[pos] * b.def_qb_delta)
            out.append(b)
        P = pd.concat(out)
        # score on the players that matter: top-N by season average plus "emerging" (hot and now ranked top-N on trailing)
        topn = TOPN[pos]
        top = P.groupby("week").fp_season.rank(ascending=False) <= topn
        emerging = ~top & ((P.fp_l3 - P.fp_season) >= 4) & (P.groupby("week").fp_l3.rank(ascending=False) <= topn)
        P = P[top | emerging]
        scores[name] = dict(MAE=round((P.pred - P.fp).abs().mean(), 2),
                            rank=round(P.groupby("week").apply(lambda g: spearmanr(g.pred, g.fp).correlation, include_groups=False).mean(), 3))
    # pick the simplest feature set unless a richer one beats it by a clear margin (0.02 MAE); avoids tie flips between runs
    best = None
    for k in [k for k in scores if k not in ("naive", "season")]:
        if best is None or scores[k]["MAE"] < scores[best]["MAE"] - 0.02:
            best = k
    # fall back to the season average only if it clearly beats the model on the top/emerging population
    # (ties go to the model: it is unbiased on top players where the season average over-projects by ~1 pt)
    use_season = pos != "QB" and scores[best]["MAE"] > scores["season"]["MAE"] + 0.10
    if use_season:
        print(f"  !! {pos}: best model set '{best}' ({scores[best]['MAE']}) does not beat the season average "
              f"({scores['season']['MAE']}); mean = season average, ranges from '{best}'")
    report.append((pos, len(tr), scores, best))
    # ---- final fit on full training season with the best feature set, plus quantiles
    cols = ladder[best]
    te = prep(te_all, pos, returning=True)
    te["missed_weeks"] = te.games_missed if "games_missed" in te else (te.weeks_since_last - 1).clip(lower=0)
    te["return_factor"] = te.missed_weeks.apply(lambda m: return_factor(pos, m))
    # Sole substitute: the lead back is Out, already missed the last game, and this back took 65%+ of the snaps in it.
    # 2019-25 (n=20): such backs got 17.6 touches / 14.8 PPR the next week (last game 20.6 touches; r=0.21 with the
    # next week). Use last game's usage x0.85 as the usage input instead of the 3-game blend that averages in backup weeks.
    te["sole_sub"] = 0
    if pos == "RB" and {"lead_back_out", "lead_missed_last", "snap_last", "touches_last"} <= set(te.columns):
        sole = (te.lead_back_out == 1) & (te.lead_missed_last == 1) & (te.snap_last >= 0.65) & te.touches_last.notna()
        te.loc[sole, "sole_sub"] = 1
        for c, lastc in (("touches_l3", "touches_last"), ("carries_l3", "carries_last"), ("targets_l3", "targets_last")):
            te.loc[sole, c] = np.maximum(te.loc[sole, c], 0.85 * te.loc[sole, lastc])
        te.loc[sole, "touch_share_l3"] = np.maximum(te.loc[sole, "touch_share_l3"], 0.7)
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
    # structural defense adjustment: the opponent's measured effect on quarterbacks, applied to the whole passing game
    # plus wind (2019-25 outdoor games, net of the Vegas total: -0.095 QB passing pts per mph; pass catchers share
    # ~2.4x that in PPR, split by target share). Cold has no effect beyond the total and is not used.
    wind = te.get("wind", pd.Series(0.0, index=te.index)).fillna(0)
    if pos == "QB":
        te["wind_adj"] = -0.095 * wind
    else:
        te["wind_adj"] = -0.23 * wind * te.get("target_share_l3", pd.Series(0.0, index=te.index)).fillna(0)
    # extreme weather (manual flag: heavy rain or 25+ mph sustained at kickoff): the passing game may not exist.
    # Size is an assumption, not an estimate (no precipitation in free historical data); every instance is logged.
    ext = te.get("extreme", pd.Series(0, index=te.index)).fillna(0).astype(int)
    if pos == "QB":
        te["weather_flag"] = np.where(ext == 1, "extreme: passing at risk", "")
        te["wind_adj"] = te.wind_adj - 4.0 * ext
    elif pos in ("WR", "TE"):
        te["weather_flag"] = np.where(ext == 1, "extreme: passing at risk", "")
        base_mean = pd.Series(te.fp_season if use_season else med.predict(Xb_s), index=te.index).clip(lower=0)
        te["wind_adj"] = te.wind_adj - 0.35 * ext * base_mean
    else:
        te["weather_flag"] = np.where(ext == 1, "extreme: run-heavy script", "")
    # listed injury, for players who are active (2024-25, top-N players who played, bias vs players with nothing listed,
    # shrunk toward 0 by n/(n+30)); the designation itself carries no effect once he plays. Not learned; applied.
    te["injury_adj"] = te.apply(lambda r: injury_adjust(r.get("injury", ""), pos), axis=1) if "injury" in te else 0.0
    te["def_adj"] = DEF_C[pos] * te.def_qb_delta + te.wind_adj + te.injury_adj
    te["mean"] = ((te.mean_pass + te.mean_rush) if pos == "QB" else (te.fp_season if use_season else med.predict(Xb_s))) * te.return_factor + te.def_adj
    if pos == "QB":
        te["mean_pass"] = te.mean_pass * te.return_factor + te.def_adj
    te["p10"], te["p50"], te["p90"] = q10.predict(Xb_s), q50.predict(Xb_s), q90.predict(Xb_qs)
    for c in ("p10", "p50", "p90"):
        te[c] = te[c] * te.return_factor + te.def_adj
    # group contributions: sum of (standardized feature * coef) by group, relative to an average player
    contrib = pd.DataFrame(Xb_s * med.coef_, columns=cols, index=te.index)
    for gname, gcols in {**grp, "pedigree": PEDIGREE}.items():
        gc = [c for c in gcols if c in cols and (gname == "pedigree" or c not in PEDIGREE)]
        te["contrib_" + gname] = contrib[gc].sum(axis=1) if gc else 0.0
    te["baseline"] = med.intercept_
    te["model_used"] = best
    if use_season:
        te["contrib_usage"], te["contrib_environment"], te["contrib_matchup"] = te.fp_season - tr.fp.mean(), 0.0, 0.0
        te["baseline"], te["model_used"] = tr.fp.mean(), "season"
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
P["ceiling_flag"] = np.where((P.position == "RB") & (P.def_expl_tercile == "soft"), "soft run D", "")
P.to_parquet(ROOT / "results" / f"projections_{SEASON}_wk{WEEK}.parquet", index=False)
print("Walk-forward validation on", TRAIN, "(top-N + emerging players only; MAE in PPR points / within-week rank correlation):")
for pos, n, sc, best in report:
    print(f"  {pos} (n={n}): " + " | ".join(f"{k}: {v['MAE']} / {v['rank']}" for k, v in sc.items()) + f"   -> using '{best}'")
import json as _json
_json.dump({pos: dict(n=int(n), chosen=best, **{k: dict(MAE=float(v["MAE"]), rank=float(v["rank"])) for k, v in sc.items()}) for pos, n, sc, best in report},
           open(ROOT / "results" / f"validation_{TRAIN}_latest.json", "w"), indent=1)
pd.set_option("display.width", 250)
cols = ["player", "position", "team", "opp", "p10", "p50", "p90", "mean", "ceiling_flag", "shootout", "def_tier", "def_qb_delta", "def_adj", "wind_adj", "injury_adj", "injury", "sole_sub", "missed_weeks", "return_factor", "weather_flag", "def_blitz", "contrib_usage", "contrib_environment", "contrib_matchup", "status", "practice"]
if NAMES:
    sel = P[P.player.isin(NAMES)]
else:
    sel = P.sort_values("mean", ascending=False).groupby("position").head(8)
print(sel[cols].round(1).to_string(index=False))
