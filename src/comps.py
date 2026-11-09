"""Historical comps for a back who has just moved into volume.

Finds 2019-2025 running backs who, like the target, had < 8 touches/g over
their previous 8 games and then three straight 12+ touch weeks, and matches
on draft capital, experience, 3-week touches and 3-week efficiency. Reports
what those comps did over their next four games.

Usage: python src/comps.py 2026 "Woody Marks" ["Kaelon Black" ...]
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

for name in NAMES:
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
