"""Each defense's individual effect on opposing quarterbacks, walk-forward.

For defense D at week W: over its games in weeks < W, each opposing QB's EPA
per dropback in that game minus that QB's EPA per dropback in his OTHER games
of the season (weeks < W, excluding games vs D). Dropback-weighted, shrunk
toward 0 with a 200-dropback prior, previous season at a weight that fades
with the current sample (zero for a defense with a new coordinator).

Also records the defense's blitz rate and pressure rate (same walk-forward)
so the profile that drives the number is visible.

Usage: python src/def_effect.py 2026  -> results/def_effect_2026.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
PRIOR = 200


def dropbacks(yr):
    cols = ["season_type", "week", "posteam", "defteam", "qb_dropback", "epa", "passer_player_id", "qb_hit", "sack", "game_id", "play_id"]
    p = pd.read_parquet(ROOT / "data" / f"pbp_{yr}.parquet", columns=cols)
    p = p[(p.season_type == "REG") & (p.qb_dropback == 1) & p.epa.notna() & p.passer_player_id.notna()].copy()
    p["pressure"] = ((p.qb_hit == 1) | (p.sack == 1)).astype(int)
    f = ROOT / "data" / f"ftn_{yr}.parquet"
    if f.exists():
        ftn = pd.read_parquet(f).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        p = p.merge(ftn[["game_id", "play_id", "n_pass_rushers"]].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left")
        p["blitz"] = (p.n_pass_rushers >= 5).astype(float)
    else:
        p["blitz"] = np.nan
    return p


def qb_delta_table(p, max_week):
    """Per defense: dropback-weighted sum of (QB game EPA - QB's EPA elsewhere), using weeks < max_week."""
    q = p[p.week < max_week]
    g = q.groupby(["defteam", "passer_player_id", "week"]).agg(epa_sum=("epa", "sum"), n=("epa", "size")).reset_index()
    tot = q.groupby("passer_player_id").agg(t_sum=("epa", "sum"), t_n=("epa", "size"))
    g = g.merge(tot, left_on="passer_player_id", right_index=True)
    g["other_n"] = g.t_n - g.n
    g["other_epa"] = (g.t_sum - g.epa_sum) / g.other_n.replace(0, np.nan)
    g = g[g.other_n >= 30]  # the QB needs a baseline from other games
    g["delta_sum"] = g.epa_sum - g.other_epa * g.n
    return g.groupby("defteam").agg(delta_sum=("delta_sum", "sum"), n=("n", "sum"))


cur, prev = dropbacks(SEASON), dropbacks(SEASON - 1)
coach = ROOT / "data" / "coaching_changes.csv"
new_dc = set(pd.read_csv(coach).query("season == @SEASON and new_dc == 1").team) if coach.exists() else set()
prev_tab = qb_delta_table(prev, 99)
rows = []
for w in range(1, 19):
    cur_tab = qb_delta_table(cur, w)
    prof = cur[cur.week < w].groupby("defteam").agg(blitz=("blitz", "mean"), pressure=("pressure", "mean"), db=("epa", "size"))
    for d in sorted(set(cur.defteam)):
        cs, cn = (cur_tab.loc[d] if d in cur_tab.index else (0.0, 0))
        ps, pn = (prev_tab.loc[d] if d in prev_tab.index else (0.0, 0))
        w_prev = 0.0 if d in new_dc else 0.5 * (100 / (100 + cn))
        delta = (cs + w_prev * ps) / (cn + w_prev * pn + PRIOR)
        raw = cs / cn if cn else np.nan
        rows.append(dict(defteam=d, week=w, def_qb_delta=delta, def_qb_delta_raw=raw, def_delta_n=cn,
                         def_blitz=prof.blitz.get(d, np.nan), def_pressure=prof.pressure.get(d, np.nan)))
out = pd.DataFrame(rows)
out.to_parquet(ROOT / "results" / f"def_effect_{SEASON}.parquet", index=False)
last = out[out.week == out.week.max()].sort_values("def_qb_delta")
pd.set_option("display.width", 200)
print(f"{SEASON} through week {out.week.max() - 1}: effect on opposing QBs (EPA/dropback vs their own average), shrunk")
print(last.head(8).round(3).to_string(index=False)); print("..."); print(last.tail(4).round(3).to_string(index=False))
