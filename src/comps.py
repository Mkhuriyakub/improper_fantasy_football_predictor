"""Historical comps for a back who has just moved into volume.

Finds 2019-2025 running backs who, like the target, had < 8 touches/g over
their previous 8 games and then three straight 12+ touch weeks, and matches
on draft capital, experience, 3-week touches and 3-week efficiency. Reports
what those comps did over their next four games.

Usage: python src/comps.py 2026 "Woody Marks" ["Kaelon Black" ...]
       python src/comps.py 2026 --wr "Keon Coleman" ...   (receivers: growth model + comps)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASON = int(sys.argv[1]); NAMES = sys.argv[2:]
E = pd.read_csv(ROOT / "results" / "rb_emergence_events_2019_25.csv")
dr = pd.read_parquet(ROOT / "data" / "draft_picks.parquet")[["gsis_id", "season", "pick"]].dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
S = pd.read_parquet(ROOT / "data" / f"player_stats_{SEASON}.parquet")
S = S[S.season_type == "REG"].copy()
S["touches"] = S.carries.fillna(0) + S.receptions.fillna(0); S["fp"] = S.fantasy_points_ppr.fillna(0)
prev = pd.concat([pd.read_parquet(ROOT / "data" / f"player_stats_{y}.parquet") for y in range(2018, SEASON)])
prev = prev[prev.season_type == "REG"]; prev["touches"] = prev.carries.fillna(0) + prev.receptions.fillna(0); prev["fp"] = prev.fantasy_points_ppr.fillna(0)

for name in ([] if "--wr" in NAMES else NAMES):
    g = S[S.player_display_name == name].sort_values("week")
    if g.empty:
        print(f"!! {name} not found"); continue
    pid = g.player_id.iloc[0]
    hist = pd.concat([prev[prev.player_id == pid], g]).sort_values(["season", "week"])
    last3 = hist.tail(3); before = hist.iloc[:-3].tail(8)
    d = dr[dr.gsis_id == pid]
    pick = float(d.pick.iloc[0]) if len(d) else 300.0
    dyear = int(d.season.iloc[0]) if len(d) else int(hist.season.min())
    t3_tpg, t3_fppt = last3.touches.mean(), last3.fp.sum() / max(last3.touches.sum(), 1)
    prev_tpg = before.touches.mean() if len(before) else 0.0
    exp = SEASON - dyear
    print(f"\n=== {name}: pick {int(pick) if pick < 300 else 'UDFA'}, year {exp + 1}; last 3 games {t3_tpg:.1f} touches/g, "
          f"{t3_fppt:.2f} pts/touch ({last3.fp.mean():.1f} pts/g); previous 8 games {prev_tpg:.1f} touches/g")
    if prev_tpg >= 8:
        print("  (not a 'thrust into volume' case; comps below are still matched on pedigree and 3-week usage)")
    c = E.copy()
    pick_grp = lambda p: 0 if p <= 64 else 1 if p <= 150 else 2
    c["dist"] = (2.0 * (c.pick.map(pick_grp) != pick_grp(pick)) + 0.5 * (c.exp - exp).abs().clip(upper=4)
                 + (c.t3_tpg - t3_tpg).abs() / 4 + (c.t3_fppt - t3_fppt).abs() / 0.15)
    top = c.nsmallest(10, "dist")
    print(f"  10 closest comps: during their 3 weeks {top.t3_ppg.mean():.1f} pts/g on {top.t3_tpg.mean():.1f} touches; "
          f"next 4 games {top.nxt_ppg.mean():.1f} pts/g on {top.nxt_tpg.mean():.1f} touches ({top.nxt_fppt.mean():.2f} pts/touch)")
    print(f"  range of next-4 pts/g: {top.nxt_ppg.quantile(.2):.1f} – {top.nxt_ppg.quantile(.8):.1f}; "
          f"kept 12+ touches/g: {(top.nxt_tpg >= 12).mean():.0%}")
    show = top[["player", "season", "pick", "exp", "t3_tpg", "t3_ppg", "nxt_tpg", "nxt_ppg"]].round(1)
    show["pick"] = show.pick.apply(lambda p: "UDFA" if p >= 300 else int(p))
    print(show.to_string(index=False))


# ---------------- receivers (WR/TE) ----------------
def receiver_weekly(season):
    cols = ["season", "week", "season_type", "posteam", "pass_attempt", "sack", "receiver_player_id", "air_yards", "yardline_100",
            "yards_after_catch", "complete_pass", "yards_gained", "pass_touchdown", "xyac_mean_yardage", "epa"]
    T = pd.read_parquet(ROOT / "data" / f"pbp_{season}.parquet", columns=cols)
    T = T[(T.season_type == "REG") & (T.pass_attempt == 1) & (T.sack != 1) & T.receiver_player_id.notna()].copy()
    T["ez"] = (T.air_yards >= T.yardline_100).astype(int); T["yac"] = T.yards_after_catch.fillna(0); T["comp"] = T.complete_pass.fillna(0)
    T["yds"] = np.where(T.comp == 1, T.yards_gained, 0); T["td"] = T.pass_touchdown.fillna(0); T["ay"] = T.air_yards.fillna(0)
    T["xyac"] = T.xyac_mean_yardage.fillna(0) * T.comp; T["one"] = 1
    g = T.groupby(["week", "posteam", "receiver_player_id"]).agg(tg=("one", "sum"), ez=("ez", "sum"), ay=("ay", "sum"), yac=("yac", "sum"), xyac=("xyac", "sum"),
                                                                 rec=("comp", "sum"), yds=("yds", "sum"), td=("td", "sum"), epa=("epa", "sum")).reset_index()
    tm = T.groupby(["week", "posteam"]).agg(team_tg=("one", "sum"), team_ay=("ay", "sum")).reset_index()
    g = g.merge(tm, on=["week", "posteam"]); g["fp"] = g.rec + g.yds / 10 + 6 * g.td
    g["tg_share"] = g.tg / g.team_tg; g["ay_share"] = g.ay / g.team_ay.replace(0, np.nan)
    return g.rename(columns={"receiver_player_id": "player_id"})


def receiver_comps(names, season):
    import statsmodels.api as sm
    E = pd.read_csv(ROOT / "results" / "receiver_emergence_events_2019_25.csv")
    X = E[["t3_adot", "t3_ay_share", "t3_ez", "t3_yac_oe", "exp"]].copy(); X["top64"] = (E.pick <= 64).astype(int); X = sm.add_constant(X)
    m_share, m_fp = sm.OLS(E.nxt_share, X).fit(), sm.OLS(E.nxt_fp, X).fit()
    W = receiver_weekly(season)
    names_map = pd.read_parquet(ROOT / "data" / f"player_stats_{season}.parquet", columns=["player_id", "player_display_name", "position"]).drop_duplicates("player_id")
    W = W.merge(names_map, on="player_id")
    for name in names:
        h = W[W.player_display_name == name].sort_values("week")
        if len(h) < 3:
            print(f"!! {name}: fewer than 3 games"); continue
        pid = h.player_id.iloc[0]; w = h.tail(3); before = h.iloc[:-3]
        d = dr[dr.gsis_id == pid]; pick = float(d.pick.iloc[0]) if len(d) else 300.0
        dyear = int(d.season.iloc[0]) if len(d) else season; exp = season - dyear
        prof = dict(t3_adot=w.ay.sum() / w.tg.sum(), t3_ay_share=w.ay_share.mean(), t3_ez=w.ez.sum() / 3, t3_yac_oe=(w.yac.sum() - w.xyac.sum()) / max(1, w.rec.sum()), exp=exp, top64=int(pick <= 64))
        x = pd.DataFrame([dict(const=1.0, **prof)])[X.columns]
        ps, pf = float(m_share.predict(x)[0]), float(m_fp.predict(x)[0])
        print(f"\n=== {name} ({h.position.iloc[0]}, pick {int(pick) if pick < 300 else 'UDFA'}, year {exp + 1}): last 3 games target share {w.tg_share.mean():.0%}, "
              f"air-yards share {w.ay_share.mean():.0%}, aDOT {prof['t3_adot']:.1f}, EZ tgts/g {prof['t3_ez']:.2f}, YAC o/e {prof['t3_yac_oe']:+.1f}, {w.fp.mean():.1f} PPR/g"
              f"{'' if before.empty else f'; before that {before.tg_share.mean():.0%} share'}")
        print(f"  growth model (184 emergences 2019-25): next-4 target share {ps:.0%}, {pf:.1f} PPR/g")
        pg = lambda p: 0 if p <= 64 else 1 if p <= 150 else 2
        c = E.copy(); c["dist"] = (1.5 * (c.pick.map(pg) != pg(pick)) + 0.4 * (c.exp - exp).abs().clip(upper=4) + (c.t3_ay_share - prof["t3_ay_share"]).abs() / 0.08
                                   + (c.t3_adot - prof["t3_adot"]).abs() / 3 + (c.t3_share - w.tg_share.mean()).abs() / 0.05)
        top = c.nsmallest(10, "dist")
        print(f"  10 closest comps: during {top.t3_share.mean():.0%} share, {top.t3_fp.mean():.1f} PPR/g -> next 4: {top.nxt_share.mean():.0%} share, {top.nxt_fp.mean():.1f} PPR/g; grew further {top.grew.mean():.0%}")
        show = top[["player", "season", "pick", "exp", "t3_share", "t3_ay_share", "t3_adot", "t3_fp", "nxt_share", "nxt_fp"]].copy()
        show["pick"] = show.pick.apply(lambda p: "UDFA" if p >= 300 else int(p))
        for c_ in ["t3_share", "t3_ay_share", "nxt_share"]: show[c_] = (show[c_] * 100).round(0).astype(int).astype(str) + "%"
        print(show.round(1).to_string(index=False))


if __name__ == "__main__" and "--wr" in sys.argv:
    receiver_comps([n for n in NAMES if n != "--wr"], SEASON)
