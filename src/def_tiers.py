"""Walk-forward defensive pass tiers on an absolute, era-adjusted scale.

For each defense and week W: pass EPA per dropback allowed relative to that
season's league average, from weeks < W, shrunk toward 0 with a 150-dropback
prior and the previous season (relative) at half weight. Tiers use fixed
cutoffs from the 2000-2025 distribution of full-season relative EPA:
  historic <= -0.205 (top 2%), elite <= -0.160 (top 5%), very good <= -0.119
  (top 10%), good <= -0.079 (top 20%), above avg <= -0.048 (top 30%).

Usage: python src/def_tiers.py 2025   -> results/def_tiers_2025.parquet
"""
import sys
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
CUTS = [(-0.205, "historic"), (-0.160, "elite"), (-0.119, "very good"), (-0.079, "good"), (-0.048, "above avg")]
PRIOR = 150


def dropbacks(yr):
    p = D / f"pbp_{yr}.parquet"
    p = pd.read_parquet(p) if p.exists() else nfl.load_pbp(yr).to_pandas()
    p = p[(p.season_type == "REG") & (p.qb_dropback == 1) & p.defteam.notna() & p.epa.notna()]
    return p[["week", "defteam", "epa"]]


prev, cur = dropbacks(SEASON - 1), dropbacks(SEASON)
prev_lg = prev.epa.mean()
pv = prev.groupby("defteam").agg(s=("epa", "sum"), n=("epa", "size"))
pv["rel_sum"] = pv.s - prev_lg * pv.n  # sum of (epa - league avg)
wk = cur.groupby(["defteam", "week"]).agg(s=("epa", "sum"), n=("epa", "size")).reset_index()
lg_by_week = cur.groupby("week").epa.agg(["sum", "size"]).cumsum()  # league running totals
grid = pd.MultiIndex.from_product([sorted(pv.index), range(1, 19)], names=["defteam", "week"]).to_frame(index=False)
wk = grid.merge(wk, how="left").fillna(0).sort_values(["defteam", "week"])
wk["ps"] = wk.groupby("defteam").s.transform(lambda x: x.shift().cumsum()).fillna(0)
wk["pn"] = wk.groupby("defteam").n.transform(lambda x: x.shift().cumsum()).fillna(0)
rows = []
for r in wk.itertuples():
    prior_weeks = lg_by_week[lg_by_week.index < r.week]
    lg = prior_weeks["sum"].iloc[-1] / prior_weeks["size"].iloc[-1] if len(prior_weeks) else prev_lg
    rel_sum = (r.ps - lg * r.pn) + 0.5 * pv.loc[r.defteam, "rel_sum"]
    n = r.pn + 0.5 * pv.loc[r.defteam, "n"]
    rel = rel_sum / (n + PRIOR)
    tier = next((t for c, t in CUTS if rel <= c), "avg or below")
    rows.append(dict(defteam=r.defteam, week=r.week, def_rel_epa=rel, def_tier=tier, def_sample=n))
out = pd.DataFrame(rows)
out["def_elite"] = out.def_tier.isin(["historic", "elite"]).astype(int)
out["def_verygood"] = (out.def_tier == "very good").astype(int)
out.to_parquet(ROOT / "results" / f"def_tiers_{SEASON}.parquet", index=False)
latest = out[out.week == out.week.max()]
print(f"{SEASON}: tiers at week {out.week.max()}: {latest.def_tier.value_counts().to_dict()}")
