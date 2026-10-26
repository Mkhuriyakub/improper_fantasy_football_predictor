"""Walk-forward quarterback traits keyed to each team's expected starter.

For each team and week W: the expected starter is the passer with the most
attempts in the team's most recent game before W (if he is listed Out on the
week-W injury report, the next passer from that game, else league average with
a flag). His traits come from his own dropbacks before W this season plus the
prior season at half weight, shrunk toward league average.

Usage: python src/qb_traits.py 2025   -> results/qb_traits_2025.parquet
"""
import sys
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
PRIOR_DB = 150


def dropbacks(yr):
    p = D / f"pbp_{yr}.parquet"
    p = pd.read_parquet(p) if p.exists() else nfl.load_pbp(yr).to_pandas()
    p = p[(p.season_type == "REG") & (p.qb_dropback == 1) & p.passer_player_id.notna() & p.posteam.notna()].copy()
    p["expl"] = ((p.complete_pass == 1) & (p.yards_gained >= 20) & (p.pass_attempt == 1)).astype(int)
    p["att"] = (p.pass_attempt == 1).astype(int)
    return p[["game_id", "week", "posteam", "passer_player_id", "passer_player_name", "epa", "expl", "att", "sack", "cpoe"]]


prev, cur = dropbacks(SEASON - 1), dropbacks(SEASON)
lg = dict(epa=prev.epa.mean(), expl=prev.expl.sum() / prev.att.sum(), sack=prev.sack.mean(), cpoe=prev.cpoe.mean())
pq = prev.groupby("passer_player_id").agg(p_n=("epa", "size"), p_epa=("epa", "sum"), p_ex=("expl", "sum"), p_att=("att", "sum"),
                                          p_sack=("sack", "sum"), p_cpoe=("cpoe", "sum"), p_cpoe_n=("cpoe", "count"))
inj = pd.read_parquet(D / f"injuries_{SEASON}.parquet") if (D / f"injuries_{SEASON}.parquet").exists() else None
sched = pd.read_parquet(D / f"schedules_{SEASON}.parquet")
sched = sched[sched.game_type == "REG"]
teams = sorted(cur.posteam.unique())
max_week = int(sched.week.max())
rows = []
for team in teams:
    tg = cur[cur.posteam == team]
    for w in range(1, max_week + 1):
        before = tg[tg.week < w]
        starter = None
        if len(before):
            last_game = before[before.week == before.week.max()]
            order = last_game.groupby("passer_player_id").att.sum().sort_values(ascending=False)
            for pid in order.index:
                out = inj is not None and ((inj.week == w) & (inj.gsis_id == pid) & (inj.report_status == "Out")).any()
                if not out:
                    starter = pid
                    break
        if starter is None:
            rows.append(dict(team=team, week=w, qb_id=None, qb_name=None, qb_epa=lg["epa"], qb_expl=lg["expl"], qb_sack=lg["sack"],
                             qb_cpoe=lg["cpoe"], qb_att_pg=np.nan, qb_sample=0, qb_unknown=1))
            continue
        h = cur[(cur.passer_player_id == starter) & (cur.week < w)]
        p = pq.loc[starter] if starter in pq.index else pd.Series(dict(p_n=0, p_epa=0, p_ex=0, p_att=0, p_sack=0, p_cpoe=0, p_cpoe_n=0))
        n = len(h) + 0.5 * p.p_n
        epa = (h.epa.sum() + 0.5 * p.p_epa + lg["epa"] * PRIOR_DB) / (n + PRIOR_DB)
        att_n = h.att.sum() + 0.5 * p.p_att
        expl = (h.expl.sum() + 0.5 * p.p_ex + lg["expl"] * PRIOR_DB) / (att_n + PRIOR_DB)
        sack = (h.sack.sum() + 0.5 * p.p_sack + lg["sack"] * PRIOR_DB) / (n + PRIOR_DB)
        cn = h.cpoe.count() + 0.5 * p.p_cpoe_n
        cpoe = (h.cpoe.sum() + 0.5 * p.p_cpoe + lg["cpoe"] * PRIOR_DB) / (cn + PRIOR_DB)
        games = h.game_id.nunique()
        att_pg = h.att.sum() / games if games else np.nan
        name = cur.loc[cur.passer_player_id == starter, "passer_player_name"].iloc[0]
        rows.append(dict(team=team, week=w, qb_id=starter, qb_name=name, qb_epa=epa, qb_expl=expl, qb_sack=sack, qb_cpoe=cpoe,
                         qb_att_pg=att_pg, qb_sample=n, qb_unknown=0))
out = pd.DataFrame(rows)
# did the starter change vs. 3 weeks earlier (what a player's trailing window reflects)?
out = out.sort_values(["team", "week"])
out["qb_3wk_ago"] = out.groupby("team").qb_id.shift(3)
out["qb_changed"] = ((out.qb_id != out.qb_3wk_ago) & out.qb_3wk_ago.notna() & out.qb_id.notna()).astype(int)
out.to_parquet(ROOT / "results" / f"qb_traits_{SEASON}.parquet", index=False)
print(f"{SEASON}: {len(out)} team-weeks; starter changes (vs 3 wks earlier): {out.qb_changed.sum()}; unknown starter rows: {out.qb_unknown.sum()}")
