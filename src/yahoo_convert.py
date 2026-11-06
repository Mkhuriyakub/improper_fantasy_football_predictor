"""Convert model projections to a custom (Yahoo) scoring format.

Method: each player's season-to-date per-game stat line (yards, TDs, INTs,
fumbles) is scaled so that its PPR value matches the model's projected mean,
then re-scored under the league's rules. Yardage bonuses are priced with a
normal approximation using the player's own game-to-game spread.

Usage: python src/yahoo_convert.py 2026 9 "Drake Maye" "Jonathan Taylor" ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])
NAMES = sys.argv[3:]

# league rules
R = dict(pass_yd=1 / 50, pass_td=6, int=-2, rush_yd=1 / 20, rush_td=6, rec_yd=1 / 20, rec_td=6, rec=0,
         fum_lost=-2, two_pt=2, bonus_pass=(300, 3), bonus_rush=(100, 3), bonus_rec=(125, 3))

st = pd.read_parquet(ROOT / "data" / f"player_stats_{SEASON}.parquet")
st = st[(st.season_type == "REG") & (st.week < WEEK)]
proj = pd.read_parquet(ROOT / "results" / f"projections_{SEASON}_wk{WEEK}.parquet")

COMP = ["passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds", "receiving_yards",
        "receiving_tds", "receptions", "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost",
        "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"]


def ppr(c):
    return (c["passing_yards"] * 0.04 + c["passing_tds"] * 4 - c["passing_interceptions"] * 2 + c["rushing_yards"] * 0.1
            + c["rushing_tds"] * 6 + c["receiving_yards"] * 0.1 + c["receiving_tds"] * 6 + c["receptions"]
            - 2 * (c["rushing_fumbles_lost"] + c["receiving_fumbles_lost"] + c["sack_fumbles_lost"])
            + 2 * (c["passing_2pt_conversions"] + c["rushing_2pt_conversions"] + c["receiving_2pt_conversions"]))


def yahoo(c):
    return (c["passing_yards"] * R["pass_yd"] + c["passing_tds"] * R["pass_td"] + c["passing_interceptions"] * R["int"]
            + c["rushing_yards"] * R["rush_yd"] + c["rushing_tds"] * R["rush_td"] + c["receiving_yards"] * R["rec_yd"]
            + c["receiving_tds"] * R["rec_td"]
            + R["fum_lost"] * (c["rushing_fumbles_lost"] + c["receiving_fumbles_lost"] + c["sack_fumbles_lost"])
            + R["two_pt"] * (c["passing_2pt_conversions"] + c["rushing_2pt_conversions"] + c["receiving_2pt_conversions"]))


def p_over(mean, sd, thr):
    return 1 - norm.cdf(thr, mean, max(sd, 1e-6)) if mean > 0 else 0.0


rows = []
for name in NAMES:
    pr = proj[proj.player == name]
    if pr.empty:
        print(f"!! {name} not in projections"); continue
    pr = pr.iloc[0]
    g = st[st.player_display_name == name].sort_values("week")
    if g.empty:
        print(f"!! {name} no stats"); continue
    g = g[g.week >= g.week.max() - 5]  # last ~6 games for the shape of the stat line
    avg = g[COMP].fillna(0).mean()
    # shrink noisy TD/INT rates halfway toward league rate-per-yard for the position
    pos = pr.position
    lg = st[st.position == pos][COMP].fillna(0).sum()
    for td, yd in [("passing_tds", "passing_yards"), ("rushing_tds", "rushing_yards"), ("receiving_tds", "receiving_yards"),
                   ("passing_interceptions", "passing_yards")]:
        rate = lg[td] / lg[yd] if lg[yd] > 0 else 0
        avg[td] = 0.5 * avg[td] + 0.5 * avg[yd] * rate
    k = pr["mean"] / max(ppr(avg), 1e-6)
    sc = avg * k
    base = yahoo(sc)
    # bonuses, normal approx with player's own game spread scaled
    sd = (g[COMP].fillna(0).std() * np.sqrt(k)).fillna(0)
    bonus = 0.0
    bp = {}
    for col, key in [("passing_yards", "bonus_pass"), ("rushing_yards", "bonus_rush"), ("receiving_yards", "bonus_rec")]:
        thr, pts = R[key]
        p = p_over(sc[col], sd[col] if sd[col] > 0 else sc[col] * 0.5, thr) if sc[col] > 5 else 0.0
        bp[key] = p
        bonus += p * pts
    rows.append(dict(player=name, pos=pr.position, team=pr.team, opp=pr.opp, model_ppr=pr["mean"], scale=k,
                     pass_yd=sc.passing_yards, pass_td=sc.passing_tds, ints=sc.passing_interceptions,
                     rush_yd=sc.rushing_yards, rush_td=sc.rushing_tds, rec_yd=sc.receiving_yards, rec_td=sc.receiving_tds,
                     p300=bp["bonus_pass"], p100=bp["bonus_rush"], p125=bp["bonus_rec"],
                     yahoo_base=base, yahoo_bonus=bonus, yahoo=base + bonus,
                     yahoo_p10=base * pr.p10 / pr["mean"], yahoo_p90=(base + bonus) * pr.p90 / pr["mean"]))

out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.round(2).to_string(index=False))
out.to_csv(ROOT / "results" / f"yahoo_{SEASON}_wk{WEEK}.csv", index=False)
