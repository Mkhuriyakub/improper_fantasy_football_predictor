"""Weekly board: top 20 QB / 30 RB / 30 WR / 20 TE with projection tiers and utilization labels.

Tiers are rank buckets of the projected mean (RB1 = 1-12, RB2 = 13-24, ...) with
sub-tiers (a, b, c...) broken wherever the gap to the next player exceeds 1.5
points. Utilization is a usage label from trailing snap share and carry/target
share (elite / high / mid / low), independent of the projection.

Usage: python src/board.py 2026 10  -> results/board_{season}_wk{week}.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])
BUCKET, GAP = 12, 1.5
N = {"QB": 20, "RB": 30, "WR": 30, "TE": 20}

p = pd.read_parquet(ROOT / "results" / f"projections_{SEASON}_wk{WEEK}.parquet")
sch = pd.read_parquet(ROOT / "data" / f"schedules_{SEASON}.parquet")
w = sch[sch.week == WEEK]
p = p[p.team.isin(set(w.home_team) | set(w.away_team)) & (p.status != "Out")].copy()
for c in ["snap_l3", "carry_share_l3", "target_share_l3", "rz_share"]:
    p[c] = p[c].fillna(0)


def tiers(b, pos):
    b = b.sort_values("mean", ascending=False).reset_index(drop=True)
    b["rank"] = b.index + 1
    b["bucket"] = pos + ((b["rank"] - 1) // BUCKET + 1).astype(str)
    gap = b["mean"].diff(-1).fillna(0)
    sub, k = [], 1
    for i in range(len(b)):
        sub.append(k)
        if (i + 1) % BUCKET == 0:
            k = 1
        elif gap[i] > GAP:
            k += 1
    b["tier"] = b.bucket + "-" + pd.Series(sub).map(lambda s: "abcde"[min(s, 5) - 1])
    return b


def util(r):
    if r.position == "RB":
        score = 0.4 * r.snap_l3 + 0.4 * r.carry_share_l3 + 0.2 * min(r.target_share_l3 / 0.15, 1)
    elif r.position in ("WR", "TE"):
        score = 0.35 * r.snap_l3 + 0.65 * min(r.target_share_l3 / 0.28, 1)
    else:
        return ""
    return "elite" if score >= 0.8 else "high" if score >= 0.65 else "mid" if score >= 0.5 else "low"


p["utilization"] = p.apply(util, axis=1)
p["emerging"] = ((p.fp_l3 - p.fp_season) >= 4).astype(int)
out = []
for pos, n in N.items():
    b = tiers(p[p.position == pos].copy(), pos)
    out.append(b[b["rank"] <= n])
B = pd.concat(out)
cols = ["rank", "tier", "utilization", "player", "position", "team", "opp", "spread", "p10", "p50", "p90", "mean", "fp_season", "fp_l3",
        "snap_l3", "carry_share_l3", "target_share_l3", "rz_share", "def_tier", "shootout", "emerging", "status", "practice"]
B = B[[c for c in cols if c in B]].round(2)
B["espn"], B["actual"] = None, None
B.to_csv(ROOT / "results" / f"board_{SEASON}_wk{WEEK}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
for pos in N:
    share = "carry_share_l3" if pos == "RB" else "target_share_l3"
    print(f"\n{pos}\n", B[B.position == pos][["rank", "tier", "utilization", "player", "team", "opp", "p10", "p50", "p90", "mean", "snap_l3", share, "rz_share", "practice"]].to_string(index=False))
