"""Walk-forward explosive-run traits with a prior-season prior.

For each defense and week: share of RB carries allowed that went 10+ yards,
computed from prior weeks of the season plus half-weight on the previous
season, shrunk toward league average (100-carry prior). Also the same for
each rusher (150-carry prior), used only as a descriptive trait.

Usage: python src/expl_traits.py 2025   -> results/expl_traits_2025.parquet
"""
import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
ROOT = Path(__file__).resolve().parents[1]


def runs(yr):
    p = nfl.load_pbp(yr).to_pandas()
    p = p[(p.season_type == "REG") & (p.play_type == "run") & (p.qb_scramble != 1) & p.rusher_player_id.notna()]
    passers = set(p.passer_player_id.dropna())
    p = p[~p.rusher_player_id.isin(passers)]
    return p[["week", "defteam", "rusher_player_id", "yards_gained"]].assign(expl=lambda d: (d.yards_gained >= 10).astype(int))


prev, cur = runs(SEASON - 1), runs(SEASON)
lg = prev.expl.mean()
MAX_WEEK = 18

dprev = prev.groupby("defteam").agg(dex=("expl", "sum"), dc=("expl", "size"))
dw = cur.groupby(["defteam", "week"]).agg(ex=("expl", "sum"), car=("expl", "size")).reset_index()
grid = pd.MultiIndex.from_product([sorted(dprev.index), range(1, MAX_WEEK + 1)], names=["defteam", "week"]).to_frame(index=False)
dw = grid.merge(dw, how="left").fillna({"ex": 0, "car": 0}).sort_values(["defteam", "week"])
dw["pex"] = dw.groupby("defteam").ex.transform(lambda s: s.shift().cumsum()).fillna(0)
dw["pcar"] = dw.groupby("defteam").car.transform(lambda s: s.shift().cumsum()).fillna(0)
dw = dw.merge(dprev, left_on="defteam", right_index=True)
dw["def_expl_sus"] = (dw.pex + 0.5 * dw.dex + lg * 100) / (dw.pcar + 0.5 * dw.dc + 100)
# tercile label by week (soft = most susceptible third)
dw["def_expl_tercile"] = dw.groupby("week").def_expl_sus.transform(lambda s: pd.qcut(s, 3, labels=["tough", "mid", "soft"]).astype(str))

pprev = prev.groupby("rusher_player_id").agg(pex_prev=("expl", "sum"), pc_prev=("expl", "size"))
pw = cur.groupby(["rusher_player_id", "week"]).agg(ex=("expl", "sum"), car=("expl", "size")).reset_index().sort_values("week")
pw["pex"] = pw.groupby("rusher_player_id").ex.transform(lambda s: s.shift().cumsum()).fillna(0)
pw["pcar"] = pw.groupby("rusher_player_id").car.transform(lambda s: s.shift().cumsum()).fillna(0)
pw = pw.merge(pprev, left_on="rusher_player_id", right_index=True, how="left").fillna({"pex_prev": 0, "pc_prev": 0})
pw["rb_expl_trait"] = (pw.pex + pw.pex_prev + lg * 150) / (pw.pcar + pw.pc_prev + 150)

out = ROOT / "results"
dw[["defteam", "week", "def_expl_sus", "def_expl_tercile"]].to_parquet(out / f"expl_def_{SEASON}.parquet", index=False)
pw[["rusher_player_id", "week", "rb_expl_trait"]].to_parquet(out / f"expl_rb_{SEASON}.parquet", index=False)
print(f"{SEASON}: league explosive rate (prev season) {lg:.3f}; defenses {dw.defteam.nunique()}; rushers {pw.rusher_player_id.nunique()}")
