"""QB model experiments: does crediting passing efficiency (EPA/dropback, success rate, ANY/A, points per dropback)
fix the under-projection of elite passers? Scored on 2025 walk-forward (weeks 7-18) and 2026 live (weeks 7-11),
top-16 by season average + emerging, with bias by efficiency tercile.

Usage: python notebooks/qb_model_experiments.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
pd.set_option("display.width", 250)

COMMON_ENV = ["spread", "implied_total", "total_line", "dome", "off_plays_pg", "off_proe", "off_plays_per_drive", "off_pass_epa",
              "opp_off_plays_pg", "opp_off_plays_per_drive", "opp_def_three_out_rate", "opp_def_plays_allowed_pg", "pace_mismatch"]
QB_USAGE = ["fp_l3", "fp_season", "attempts_l3", "pass_yds_l3", "pass_td_l3", "rush_yds_l3", "carries_l3"]
QB_ENV = COMMON_ENV + ["off_dropbacks_pg", "off_pass_sr", "off_rz_plays_pg"]
QB_MATCH = ["opp_def_pass_epa_allowed", "opp_def_ypa_allowed", "opp_def_pressure_rate", "opp_def_sack_rate", "opp_def_blitz_rate", "pass_edge"]
QB_FULL = QB_USAGE + QB_ENV + QB_MATCH
QB_PASS = ["pass_fp_l3", "pass_fp_season", "attempts_l3", "pass_yds_l3", "pass_td_l3", "qb_epa", "cpoe_std"] + COMMON_ENV + [
    "off_dropbacks_pg", "off_pass_sr", "off_rz_plays_pg", "def_rel_epa", "opp_def_ypa_allowed", "opp_def_pressure_rate",
    "opp_def_sack_rate", "opp_def_blitz_rate"]
QB_RUSH = ["rush_yds_l3", "carries_l3", "rush_fp_l3", "rush_fp_season", "spread", "implied_total", "off_rz_plays_pg",
           "opp_def_rush_epa_allowed", "opp_def_line_yds_allowed"]
EFF = ["eff_epa", "eff_sr", "eff_anya", "eff_fp_db", "db_pg", "exp_pass_fp"]


def add_all(df, season):
    d = pd.read_parquet(ROOT / "results" / f"expl_def_{season}.parquet").rename(columns={"defteam": "opp"})
    df = df.merge(d, on=["opp", "week"], how="left")
    q = pd.read_parquet(ROOT / "results" / f"qb_traits_{season}.parquet")
    df = df.merge(q[["team", "week", "qb_id", "qb_name", "qb_epa", "qb_expl", "qb_changed"]], on=["team", "week"], how="left")
    df["new_starter"] = df.qb_changed.fillna(0).astype(int)
    st = pd.read_parquet(ROOT / "data" / f"player_stats_{season}.parquet")
    st = st[st.season_type == "REG"].sort_values(["player_id", "week"]).copy()
    st["pass_fp_g"] = st.passing_yards.fillna(0) * 0.04 + st.passing_tds.fillna(0) * 4 - st.passing_interceptions.fillna(0) * 2 - st.sack_fumbles_lost.fillna(0) * 2
    st["rush_fp_g"] = st.rushing_yards.fillna(0) * 0.1 + st.rushing_tds.fillna(0) * 6 - st.rushing_fumbles_lost.fillna(0) * 2
    g = st.groupby("player_id")
    st["pass_fp_l3"] = g.pass_fp_g.transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    st["pass_fp_season"] = g.pass_fp_g.transform(lambda s: s.shift().expanding().mean())
    st["rush_fp_season"] = g.rush_fp_g.transform(lambda s: s.shift().expanding().mean())
    latest = st.groupby("player_id")[["pass_fp_g", "rush_fp_g"]].apply(lambda x: pd.Series({
        "pass_fp_l3_latest": x.pass_fp_g.tail(3).mean(), "pass_fp_season_latest": x.pass_fp_g.mean(), "rush_fp_season_latest": x.rush_fp_g.mean()}))
    df = df.merge(st[["player_id", "week", "pass_fp_l3", "pass_fp_season", "rush_fp_season"]], on=["player_id", "week"], how="left")
    df = df.merge(latest, left_on="player_id", right_index=True, how="left")
    for c in ["pass_fp_l3", "pass_fp_season", "rush_fp_season"]:
        df[c] = df[c].fillna(df[c + "_latest"])
    df["pass_fp"] = df.fp - df.rush_fp - df.rec_fp
    t = pd.read_parquet(ROOT / "results" / f"def_tiers_{season}.parquet").rename(columns={"defteam": "opp"})
    df = df.merge(t[["opp", "week", "def_rel_epa", "def_tier"]], on=["opp", "week"], how="left")
    df["def_rel_epa"] = df.def_rel_epa.fillna(0)
    d = pd.read_parquet(ROOT / "results" / f"def_effect_{season}.parquet").rename(columns={"defteam": "opp"})
    df = df.merge(d[["opp", "week", "def_qb_delta", "def_blitz", "def_pressure"]], on=["opp", "week"], how="left")
    df["def_qb_delta"] = df.def_qb_delta.fillna(0)
    n = pd.read_parquet(ROOT / "data" / "ngs_passing.parquet")
    n = n[(n.season == season) & (n.week > 0)].sort_values("week").copy()
    n["x"] = n.completion_percentage_above_expectation * n.attempts
    cum_x = n.groupby("player_gsis_id").x.cumsum() - n.x
    cum_a = n.groupby("player_gsis_id").attempts.cumsum() - n.attempts
    n["cpoe_std"] = cum_x / (cum_a + 40)
    n = n[["player_gsis_id", "week", "cpoe_std"]].rename(columns={"player_gsis_id": "player_id"})
    full = pd.MultiIndex.from_product([n.player_id.unique(), range(1, 19)], names=["player_id", "week"]).to_frame(index=False)
    n = full.merge(n, how="left").sort_values(["player_id", "week"])
    n["cpoe_std"] = n.groupby("player_id").cpoe_std.ffill()
    df = df.merge(n, on=["player_id", "week"], how="left")
    df["cpoe_std"] = df.cpoe_std.fillna(0)
    return df.merge(efficiency(season), on=["player_id", "week"], how="left")


def efficiency(season, prior_db=100):
    """Walk-forward passing efficiency per QB per week: EPA/dropback, success rate, ANY/A, passing fantasy points per
    dropback (shrunk toward league average with a prior_db-dropback prior; prior season at half weight), dropbacks per
    game, and their product exp_pass_fp = points per dropback x dropbacks per game (what efficiency and volume imply)."""
    def load(yr):
        p = pd.read_parquet(ROOT / "data" / f"pbp_{yr}.parquet")
        p = p[(p.season_type == "REG") & (p.qb_dropback == 1) & p.passer_player_id.notna()].copy()
        p["succ"] = (p.epa > 0).astype(int)
        p["att"] = (p.pass_attempt == 1).astype(int)
        p["sk"] = p.sack.fillna(0)
        p["anya_num"] = p.passing_yards.fillna(0) + 20 * p.pass_touchdown.fillna(0) - 45 * p.interception.fillna(0) + np.where(p.sk == 1, p.yards_gained.fillna(0), 0)
        p["fp"] = p.passing_yards.fillna(0) * 0.04 + p.pass_touchdown.fillna(0) * 4 - p.interception.fillna(0) * 2
        return p[["game_id", "week", "passer_player_id", "epa", "succ", "att", "sk", "anya_num", "fp"]].rename(columns={"passer_player_id": "player_id"})
    prev, cur = load(season - 1), load(season)
    lg = dict(epa=prev.epa.mean(), sr=prev.succ.mean(), anya=prev.anya_num.sum() / (prev.att.sum() + prev.sk.sum()), fp_db=prev.fp.sum() / len(prev))
    pg = prev.groupby("player_id").agg(n=("epa", "size"), epa=("epa", "sum"), succ=("succ", "sum"), att=("att", "sum"), sk=("sk", "sum"), anya=("anya_num", "sum"), fp=("fp", "sum"))
    cg = cur.groupby(["player_id", "week"]).agg(n=("epa", "size"), epa=("epa", "sum"), succ=("succ", "sum"), att=("att", "sum"), sk=("sk", "sum"), anya=("anya_num", "sum"), fp=("fp", "sum")).reset_index()
    rows = []
    for pid, h in cg.groupby("player_id"):
        h = h.sort_values("week")
        p = pg.loc[pid] if pid in pg.index else pd.Series(dict(n=0, epa=0, succ=0, att=0, sk=0, anya=0, fp=0))
        cum = h[["n", "epa", "succ", "att", "sk", "anya", "fp"]].cumsum()
        games = np.arange(1, len(h) + 1)
        for i, w in enumerate(h.week.values):
            # values for week w use games before w; row i's cumulative includes week w, so shift by one
            c = cum.iloc[i - 1] if i > 0 else pd.Series(dict(n=0, epa=0, succ=0, att=0, sk=0, anya=0, fp=0))
            n = c.n + 0.5 * p.n
            denom = n + prior_db
            rows.append(dict(player_id=pid, week=int(w),
                             eff_epa=(c.epa + 0.5 * p.epa + lg["epa"] * prior_db) / denom,
                             eff_sr=(c.succ + 0.5 * p.succ + lg["sr"] * prior_db) / denom,
                             eff_anya=(c.anya + 0.5 * p.anya + lg["anya"] * prior_db) / (c.att + c.sk + 0.5 * (p.att + p.sk) + prior_db),
                             eff_fp_db=(c.fp + 0.5 * p.fp + lg["fp_db"] * prior_db) / denom,
                             db_pg=(c.n / (games[i] - 1)) if i > 0 else np.nan))
        # one more row for the week after the last game (target weeks beyond the data), carried as the latest state
        c = cum.iloc[-1]; n = c.n + 0.5 * p.n; denom = n + prior_db
        rows.append(dict(player_id=pid, week=int(h.week.values[-1]) + 1,
                         eff_epa=(c.epa + 0.5 * p.epa + lg["epa"] * prior_db) / denom, eff_sr=(c.succ + 0.5 * p.succ + lg["sr"] * prior_db) / denom,
                         eff_anya=(c.anya + 0.5 * p.anya + lg["anya"] * prior_db) / (c.att + c.sk + 0.5 * (p.att + p.sk) + prior_db),
                         eff_fp_db=(c.fp + 0.5 * p.fp + lg["fp_db"] * prior_db) / denom, db_pg=c.n / len(h)))
    e = pd.DataFrame(rows)
    # carry forward through byes / missed weeks
    full = pd.MultiIndex.from_product([e.player_id.unique(), range(1, 19)], names=["player_id", "week"]).to_frame(index=False)
    e = full.merge(e, how="left").sort_values(["player_id", "week"])
    e[["eff_epa", "eff_sr", "eff_anya", "eff_fp_db", "db_pg"]] = e.groupby("player_id")[["eff_epa", "eff_sr", "eff_anya", "eff_fp_db", "db_pg"]].ffill()
    e["exp_pass_fp"] = e.eff_fp_db * e.db_pg
    return e


def prep(df):
    d = df[(df.position == "QB") & (df.attempts_l3 >= 15) & (df.status != "Out") & (df.weeks_since_last <= 3)].copy()
    return d[d.played == 1] if "played" in d else d


def fill(train, test, cols):
    med = train[cols].median()
    return train[cols].fillna(med), test[cols].fillna(med)


def ridge(a, b, cols, y, alpha):
    Xa, Xb = fill(a, b, cols)
    sc = StandardScaler().fit(Xa)
    return Ridge(alpha=alpha).fit(sc.transform(Xa), a[y]).predict(sc.transform(Xb))


def quant(a, b, cols, y, q=0.5):
    Xa, Xb = fill(a, b, cols)
    sc = StandardScaler().fit(Xa)
    return QuantileRegressor(quantile=q, alpha=0.01, solver="highs").fit(sc.transform(Xa), a[y]).predict(sc.transform(Xb))


_ST = {}
def actual_db(season):
    if season not in _ST:
        st = pd.read_parquet(ROOT / "data" / f"player_stats_{season}.parquet"); st = st[st.season_type == "REG"]
        _ST[season] = st.assign(db_act=st.attempts.fillna(0) + st.sacks_suffered.fillna(0))[["player_id", "week", "db_act"]]
    return _ST[season]


def vol_x_eff(a, b, alpha=20):
    """Volume x efficiency: project the week's dropbacks and passing points per dropback separately and multiply."""
    vol_cols = ["db_pg", "attempts_l3", "off_dropbacks_pg", "off_proe", "off_plays_pg", "spread", "implied_total", "total_line", "opp_off_plays_pg", "opp_def_three_out_rate", "pace_mismatch", "dome"]
    eff_cols = ["eff_epa", "eff_sr", "eff_anya", "eff_fp_db", "cpoe_std", "qb_epa", "off_pass_sr", "off_pass_epa", "def_rel_epa", "opp_def_ypa_allowed", "opp_def_pressure_rate", "opp_def_sack_rate", "opp_def_blitz_rate", "implied_total", "spread"]
    a = a.merge(pd.concat([actual_db(s) for s in a.season.unique()]), on=["player_id", "week"], how="left")
    a = a[a.db_act.fillna(0) >= 10].copy()
    a["fp_per_db"] = a.pass_fp / a.db_act
    vol = ridge(a, b, vol_cols, "db_act", alpha)
    eff = ridge(a, b, eff_cols, "fp_per_db", alpha)
    return vol * eff + ridge(a, b, QB_RUSH, "rush_fp", 20)


def hgb(a, b, cols, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    Xa, Xb = fill(a, b, cols)
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(Xa, a[y])
    return m.predict(Xb)


VARIANTS = {
    "season avg": lambda a, b: b.fp_season.values,
    "board median (usage+env, q50)": lambda a, b: quant(a, b, QB_USAGE + QB_ENV, "fp"),
    "ladder ridge usage+env a20": lambda a, b: ridge(a, b, QB_USAGE + QB_ENV, "fp", 20),
    "q50 usage+env+eff": lambda a, b: quant(a, b, QB_USAGE + QB_ENV + EFF, "fp"),
    "ridge usage+env+eff a20": lambda a, b: ridge(a, b, QB_USAGE + QB_ENV + EFF, "fp", 20),
    "current mean (split a60/20)": lambda a, b: ridge(a, b, QB_PASS, "pass_fp", 60) + ridge(a, b, QB_RUSH, "rush_fp", 20),
    "split+eff a60": lambda a, b: ridge(a, b, QB_PASS + EFF, "pass_fp", 60) + ridge(a, b, QB_RUSH, "rush_fp", 20),
    "split+eff a20": lambda a, b: ridge(a, b, QB_PASS + EFF, "pass_fp", 20) + ridge(a, b, QB_RUSH, "rush_fp", 20),
    "eff-only pass a20": lambda a, b: ridge(a, b, EFF + ["pass_fp_season", "attempts_l3"] + COMMON_ENV + ["def_rel_epa", "opp_def_pressure_rate"], "pass_fp", 20) + ridge(a, b, QB_RUSH, "rush_fp", 20),
    "median on split+eff (q50)": lambda a, b: quant(a, b, list(dict.fromkeys(QB_PASS + EFF + QB_RUSH)), "fp"),
}


def score(P, label):
    top = P.groupby("week").fp_season.rank(ascending=False) <= 16
    emerging = ~top & ((P.fp_l3 - P.fp_season) >= 4) & (P.groupby("week").fp_l3.rank(ascending=False) <= 16)
    S = P[top | emerging].copy()
    S["tercile"] = pd.qcut(S.eff_epa.fillna(S.eff_epa.median()), 3, labels=["low", "mid", "high"])
    out = []
    for name in VARIANTS:
        err = S[name] - S.fp
        bias = (-err).groupby(S.tercile, observed=True).mean()
        out.append(dict(variant=name, MAE=err.abs().mean(), rank=S.groupby("week").apply(lambda g: spearmanr(g[name], g.fp).correlation, include_groups=False).mean(),
                        bias_low=bias["low"], bias_mid=bias["mid"], bias_high=bias["high"],
                        MAE_high=err[S.tercile == "high"].abs().mean()))
    r = pd.DataFrame(out).set_index("variant").round(3)
    print(f"\n== {label}: n={len(S)} QB-weeks (top-16 + emerging); bias = actual - projection by efficiency tercile")
    print(r.to_string())
    return S


def run_walkforward(train_df, extra_train=None):
    out = []
    for w in range(7, 19):
        a, b = train_df[train_df.week < w], train_df[train_df.week == w].copy()
        if extra_train is not None:
            a = pd.concat([extra_train, a])
        if b.empty:
            continue
        for name, f in VARIANTS.items():
            b[name] = f(a, b) + (0 if name == "season avg" else 10 * b.def_qb_delta.values)
        out.append(b)
    return pd.concat(out)


def run_live(train_df, test_df):
    b = test_df.copy()
    for name, f in VARIANTS.items():
        b[name] = f(train_df, b) + (0 if name == "season avg" else 10 * b.def_qb_delta.values)
    return b


if __name__ == "__main__":
    f24 = prep(add_all(pd.read_parquet(ROOT / "results" / "features_2024.parquet"), 2024))
    f25 = prep(add_all(pd.read_parquet(ROOT / "results" / "features_2025.parquet"), 2025))
    import os
    live_dir = Path(os.environ.get("LIVE_DIR", ROOT / "results"))  # rebuilt week files with actuals filled in
    live = []
    for w in range(7, 12):
        f = live_dir / f"features_2026_wk{w}.parquet"
        if not f.exists():
            f = ROOT / "results" / f"features_2026_wk{w}.parquet"
        t = add_all(pd.read_parquet(f), 2026)
        live.append(prep(t))
    f26 = pd.concat(live)
    f26 = f26[f26.fp.notna()]
    print("training rows: 2024", len(f24), "2025", len(f25), "| live 2026 wks 7-11:", len(f26))
    print("efficiency feature coverage 2025:", f25[EFF].notna().mean().round(2).to_dict())

    S1 = score(run_walkforward(f25), "2025 walk-forward, train 2025 weeks < w")
    S2 = score(run_walkforward(f25, extra_train=f24), "2025 walk-forward, train 2024 + 2025 weeks < w")
    S3 = score(run_live(f25, f26), "2026 live weeks 7-11, train 2025")
    S4 = score(run_live(pd.concat([f24, f25]), f26), "2026 live weeks 7-11, train 2024 + 2025")
    names = ["Brock Purdy", "Joe Burrow", "Trevor Lawrence", "Patrick Mahomes", "Dak Prescott", "Jared Goff", "C.J. Stroud", "Josh Allen", "Kirk Cousins"]
    cols = ["player", "week", "opp", "fp", "fp_season", "eff_epa", "eff_anya", "exp_pass_fp", "season avg", "board median (usage+env, q50)", "q50 usage+env+eff", "current mean (split a60/20)", "split+eff a20"]
    print("\n== 2026 live, selected QBs (train 2024+2025)")
    print(S4[S4.player.isin(names)].sort_values(["player", "week"])[cols].round(1).to_string(index=False))
    print("\n== per-QB mean over 2026 weeks 7-11 (train 2024+2025)")
    g = S4[S4.player.isin(names)].groupby("player")[["fp", "season avg", "board median (usage+env, q50)", "q50 usage+env+eff", "current mean (split a60/20)", "split+eff a20"]].mean().round(1)
    g["n"] = S4[S4.player.isin(names)].groupby("player").size()
    print(g.sort_values("fp", ascending=False).to_string())


    # ---- where is the bias? finer efficiency bins, pooled 2024 wf + 2025 wf + 2026 live; what explains the residual
    f24wf = run_walkforward(f24)
    pool = pd.concat([f24wf.assign(src="2024wf"), run_walkforward(f25).assign(src="2025wf"), run_live(pd.concat([f24, f25]), f26).assign(src="2026live")])
    top = pool.groupby(["src", "week"]).fp_season.rank(ascending=False) <= 16
    pool = pool[top].copy()
    pool["resid_mean"] = pool.fp - pool["current mean (split a60/20)"]
    pool["resid_season"] = pool.fp - pool["season avg"]
    pool["resid_pass"] = pool.pass_fp - pool.pass_fp_season
    pool["resid_rush"] = pool.rush_fp - pool.rush_fp_season
    pool["q"] = pd.qcut(pool.eff_epa, 5, labels=["q1 low", "q2", "q3", "q4", "q5 high"])
    pool["top5"] = pool.groupby(["src", "week"]).eff_epa.rank(ascending=False) <= 5
    print("\n== pooled top-16 QB-weeks by efficiency quintile (season-to-date EPA/dropback, shrunk): bias of current mean and of the season average")
    print(pool.groupby("q", observed=True).agg(n=("fp", "size"), eff_epa=("eff_epa", "mean"), actual=("fp", "mean"), proj_mean=("current mean (split a60/20)", "mean"),
                                                 bias_mean=("resid_mean", "mean"), bias_season=("resid_season", "mean"), pass_vs_season=("resid_pass", "mean"), rush_vs_season=("resid_rush", "mean")).round(2).to_string())
    print(pool.groupby(["src", "top5"]).agg(n=("fp", "size"), bias_mean=("resid_mean", "mean"), bias_season=("resid_season", "mean")).round(2).to_string())
    import statsmodels.api as sm
    for src in ["2024wf", "2025wf", "2026live"]:
        d = pool[pool.src == src]
        X = sm.add_constant(pd.DataFrame({"eff_epa_c": d.eff_epa - d.eff_epa.mean(), "gap": d.exp_pass_fp - d.pass_fp_season}))
        m = sm.OLS(d.resid_mean, X).fit()
        print(f"\n{src}: residual (actual - current mean) ~ eff_epa (centered) + (efficiency-implied pass pts - realized season pass pts)")
        print(m.params.round(2).to_dict(), "t:", m.tvalues.round(1).to_dict(), "n", len(d))
        X1 = sm.add_constant(d.eff_epa - d.eff_epa.mean())
        m1 = sm.OLS(d.resid_mean, X1).fit(); print("  slope on eff_epa alone: %.1f pts per EPA/db (t=%.1f)" % (m1.params.iloc[1], m1.tvalues.iloc[1]))
