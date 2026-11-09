"""Player pedigree and prior-season efficiency, built from seasons before the target season.

Per player: draft pick (300 = undrafted), years of experience, prior-season
opportunities per game, and prior-season fantasy points per opportunity shrunk
toward the position's league rate with a 150-opportunity prior. Opportunities
are carries + targets (RB/WR/TE) or dropbacks ~ attempts + carries (QB).

Usage: python src/pedigree.py 2025  -> results/pedigree_2025.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
K = 150
FIRST = 2018


def build(season):
    frames = []
    for y in range(FIRST, season):
        p = D / f"player_stats_{y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    S = pd.concat(frames)
    S = S[(S.season_type == "REG") & S.position.isin(["RB", "WR", "TE", "QB"])].copy()
    S["opps"] = S.carries.fillna(0) + np.where(S.position == "QB", S.attempts.fillna(0), S.targets.fillna(0))
    S["fp"] = S.fantasy_points_ppr.fillna(0)
    lg = S.groupby("position").apply(lambda x: x.fp.sum() / max(x.opps.sum(), 1))
    pr = S.groupby(["player_id", "position"]).agg(prior_opps=("opps", "sum"), prior_fp=("fp", "sum"), prior_games=("week", "count"),
                                                  first_season=("season", "min")).reset_index()
    pr["lg"] = pr.position.map(lg)
    pr["prior_fp_per_opp"] = (pr.prior_fp + K * pr.lg) / (pr.prior_opps + K)
    pr["prior_opp_pg"] = pr.prior_opps / pr.prior_games.clip(lower=1)
    dr = pd.read_parquet(D / "draft_picks.parquet")[["gsis_id", "season", "pick"]].rename(columns={"season": "draft_year"})
    dr = dr.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id")
    # all players who appear in the target season too (rookies have no prior rows)
    cur = D / f"player_stats_{season}.parquet"
    ids = pd.read_parquet(cur)[["player_id", "position"]].drop_duplicates() if cur.exists() else pr[["player_id", "position"]]
    out = ids.merge(pr.drop(columns=["position"]), on="player_id", how="left").merge(dr, left_on="player_id", right_on="gsis_id", how="left")
    out["pick"] = out.pick.fillna(300)
    out["draft_year"] = out.draft_year.fillna(out.first_season).fillna(season)
    out["exp"] = (season - out.draft_year).clip(lower=0)
    out["top64"] = (out.pick <= 64).astype(int)
    out["log_pick"] = np.log(out.pick)
    out["prior_fp_per_opp"] = out.prior_fp_per_opp.fillna(out.position.map(lg))
    out["prior_opp_pg"] = out.prior_opp_pg.fillna(0)
    out["prior_games"] = out.prior_games.fillna(0)
    return out[["player_id", "pick", "exp", "top64", "log_pick", "prior_fp_per_opp", "prior_opp_pg", "prior_games"]]


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    out = build(season)
    out.to_parquet(ROOT / "results" / f"pedigree_{season}.parquet", index=False)
    print(f"{season}: {len(out)} players; drafted top-64: {out.top64.sum()}; with prior data: {(out.prior_games > 0).sum()}")
