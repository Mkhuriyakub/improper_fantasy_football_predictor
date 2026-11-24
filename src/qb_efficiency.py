"""Walk-forward passing efficiency per quarterback and week.

For each passer and week W, from his dropbacks before W this season plus the prior season at half weight, shrunk
toward league average with a 100-dropback prior: EPA per dropback, success rate (EPA > 0), adjusted net yards per
attempt, passing fantasy points per dropback, dropbacks per game, and their product (points per dropback x
dropbacks per game: what efficiency and volume together imply). Values carry forward through byes and missed weeks,
and one week past the last game so a target week beyond the data is covered.

Tested 2024+2025 walk-forward and 2026 live (notebooks/qb_model_experiments.py): as inputs to the QB passing model and
the QB median they lower MAE and raise rank on 2026 weeks 7-11 (6.47 -> 6.42 mean, 0.33 -> 0.36 rank) and are neutral
on 2025. There is no stable "elite passer premium" beyond them: the model's bias by efficiency quintile is zero pooled
over 2024-26, positive in 2024 and 2026 and negative in 2025.

Usage: python src/qb_efficiency.py 2025   -> results/qb_efficiency_2025.parquet
"""
import sys
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
PRIOR_DB = 100
COLS = ["eff_epa", "eff_sr", "eff_anya", "eff_fp_db", "db_pg"]


def load(yr):
    f = D / f"pbp_{yr}.parquet"
    if f.exists():
        p = pd.read_parquet(f)
    else:
        p = nfl.load_pbp(yr).to_pandas()
        p.to_parquet(f, index=False)
    p = p[(p.season_type == "REG") & (p.qb_dropback == 1) & p.passer_player_id.notna()].copy()
    p["succ"] = (p.epa > 0).astype(int)
    p["att"] = (p.pass_attempt == 1).astype(int)
    p["sk"] = p.sack.fillna(0)
    p["anya_num"] = p.passing_yards.fillna(0) + 20 * p.pass_touchdown.fillna(0) - 45 * p.interception.fillna(0) + np.where(p.sk == 1, p.yards_gained.fillna(0), 0)
    p["fp"] = p.passing_yards.fillna(0) * 0.04 + p.pass_touchdown.fillna(0) * 4 - p.interception.fillna(0) * 2
    return p[["game_id", "week", "passer_player_id", "epa", "succ", "att", "sk", "anya_num", "fp"]].rename(columns={"passer_player_id": "player_id"})


def state(c, p, lg):
    n = c.n + 0.5 * p.n
    denom = n + PRIOR_DB
    return dict(eff_epa=(c.epa + 0.5 * p.epa + lg["epa"] * PRIOR_DB) / denom,
                eff_sr=(c.succ + 0.5 * p.succ + lg["sr"] * PRIOR_DB) / denom,
                eff_anya=(c.anya + 0.5 * p.anya + lg["anya"] * PRIOR_DB) / (c.att + c.sk + 0.5 * (p.att + p.sk) + PRIOR_DB),
                eff_fp_db=(c.fp + 0.5 * p.fp + lg["fp_db"] * PRIOR_DB) / denom)


prev, cur = load(SEASON - 1), load(SEASON)
lg = dict(epa=prev.epa.mean(), sr=prev.succ.mean(), anya=prev.anya_num.sum() / (prev.att.sum() + prev.sk.sum()), fp_db=prev.fp.sum() / len(prev))
agg = dict(n=("epa", "size"), epa=("epa", "sum"), succ=("succ", "sum"), att=("att", "sum"), sk=("sk", "sum"), anya=("anya_num", "sum"), fp=("fp", "sum"))
pg = prev.groupby("player_id").agg(**agg)
cg = cur.groupby(["player_id", "week"]).agg(**agg).reset_index()
zero = pd.Series(dict(n=0, epa=0, succ=0, att=0, sk=0, anya=0, fp=0))
rows = []
for pid, h in cg.groupby("player_id"):
    h = h.sort_values("week")
    p = pg.loc[pid] if pid in pg.index else zero
    cum = h[["n", "epa", "succ", "att", "sk", "anya", "fp"]].cumsum()
    for i, w in enumerate(h.week.values):
        c = cum.iloc[i - 1] if i > 0 else zero  # games before week w
        rows.append(dict(player_id=pid, week=int(w), **state(c, p, lg), db_pg=(c.n / i) if i > 0 else np.nan))
    c = cum.iloc[-1]
    rows.append(dict(player_id=pid, week=int(h.week.values[-1]) + 1, **state(c, p, lg), db_pg=c.n / len(h)))
e = pd.DataFrame(rows)
full = pd.MultiIndex.from_product([e.player_id.unique(), range(1, 19)], names=["player_id", "week"]).to_frame(index=False)
e = full.merge(e, how="left").sort_values(["player_id", "week"])
e[COLS] = e.groupby("player_id")[COLS].ffill()
e["exp_pass_fp"] = e.eff_fp_db * e.db_pg
e["season"] = SEASON
e.to_parquet(ROOT / "results" / f"qb_efficiency_{SEASON}.parquet", index=False)
print(f"{SEASON}: {e.player_id.nunique()} passers; league prior EPA/db {lg['epa']:.3f}, success {lg['sr']:.3f}, ANY/A {lg['anya']:.2f}, pts/db {lg['fp_db']:.3f}")
