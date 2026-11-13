"""Situational quarterback profile from play-by-play.

For each QB (min dropbacks), EPA/dropback, success rate, sack rate, interception
rate and low-percentage-throw rate split by situation:
  after_opp_score   possession that starts right after the opponent scored
  after_own_score   possession after the QB's own team scored (got the ball back)
  after_turnover    possession that starts after the team's own turnover (defense gave it back)
  own_territory     drive starts inside own 25
  trailing_7        trailing by 7+ when the play starts
  leading_7         leading by 7+
  third_long        3rd/4th down, 7+ to go
  pressure          FTN-charted pressure (if available) or qb_hit
  low_pct_throw     attempts with completion probability < 0.35 (share of attempts)
A QB's "composure delta" is his EPA/dropback in the adverse situations
(after_opp_score, after_turnover, trailing_7, third_long) minus his EPA/dropback elsewhere.

Usage: python src/qb_situations.py 2026 [min_dropbacks] ["J.McCarthy" ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASON = int(sys.argv[1]); MIN_DB = int(sys.argv[2]) if len(sys.argv) > 2 else 100
NAMES = sys.argv[3:]

p = pd.read_parquet(ROOT / "data" / f"pbp_{SEASON}.parquet")
p = p[(p.season_type == "REG") & (p.qb_dropback == 1) & p.passer_player_name.notna() & p.epa.notna()].copy()
p["qb"] = p.passer_player_name
# previous drive in the same game, from the drive table
drives = p.drop_duplicates(["game_id", "fixed_drive"])[["game_id", "fixed_drive", "posteam", "fixed_drive_result", "drive_start_transition"]].sort_values(["game_id", "fixed_drive"])
drives["prev_team"] = drives.groupby("game_id").posteam.shift(); drives["prev_result"] = drives.groupby("game_id").fixed_drive_result.shift()
drives["prev_prev_team"] = drives.groupby("game_id").posteam.shift(2); drives["prev_prev_result"] = drives.groupby("game_id").fixed_drive_result.shift(2)
p = p.merge(drives[["game_id", "fixed_drive", "prev_team", "prev_result", "prev_prev_team", "prev_prev_result"]], on=["game_id", "fixed_drive"], how="left")
score = lambda r: r.isin(["Touchdown", "Field goal"])
p["after_opp_score"] = (p.prev_team != p.posteam) & score(p.prev_result)
p["after_own_score"] = (p.prev_team != p.posteam) & (p.prev_prev_team == p.posteam) & score(p.prev_prev_result)
p["after_turnover"] = (p.prev_team == p.posteam) & p.prev_result.isin(["Interception", "Fumble", "Turnover on downs"]) | \
                      ((p.prev_team != p.posteam) & (p.prev_prev_team == p.posteam) & p.prev_prev_result.isin(["Interception", "Fumble"]))
p["own_territory"] = p.drive_start_yard_line.astype(str).str.contains(" ") & p.apply(
    lambda r: (str(r.drive_start_yard_line).split(" ")[0] == r.posteam) and int(str(r.drive_start_yard_line).split(" ")[-1]) <= 25 if " " in str(r.drive_start_yard_line) else False, axis=1)
p["trailing_7"] = p.score_differential <= -7
p["leading_7"] = p.score_differential >= 7
p["third_long"] = p.down.isin([3, 4]) & (p.ydstogo >= 7)
p["pressure"] = (p.qb_hit == 1) | (p.sack == 1)
p["success"] = (p.epa > 0).astype(int)
p["low_pct"] = (p.pass_attempt == 1) & (p.cp < 0.35)
p["adverse"] = p.after_opp_score | p.after_turnover | p.trailing_7 | p.third_long
SITS = ["after_opp_score", "after_own_score", "after_turnover", "own_territory", "trailing_7", "leading_7", "third_long", "pressure"]


def prof(g):
    base = dict(dropbacks=len(g), epa=g.epa.mean(), success=g.success.mean(), sack_rate=g.sack.mean(), int_rate=g.interception.mean(),
                low_pct_rate=g.low_pct.sum() / max(1, g.pass_attempt.sum()), cpoe=g.cpoe.mean())
    for s in SITS:
        x = g[g[s]]
        base[f"{s}_n"] = len(x); base[f"{s}_epa"] = x.epa.mean() if len(x) >= 8 else np.nan
    adv, calm = g[g.adverse], g[~g.adverse]
    base["adverse_epa"] = adv.epa.mean(); base["calm_epa"] = calm.epa.mean(); base["composure_delta"] = adv.epa.mean() - calm.epa.mean()
    base["adverse_low_pct"] = adv.low_pct.sum() / max(1, adv.pass_attempt.sum()); base["adverse_sack"] = adv.sack.mean(); base["adverse_int"] = adv.interception.mean()
    return pd.Series(base)


out = p.groupby("qb").apply(prof)
out = out[out.dropbacks >= MIN_DB].sort_values("epa", ascending=False)
lg = prof(p); out.loc["LEAGUE"] = lg
out.to_csv(ROOT / "results" / f"qb_situations_{SEASON}.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
cols = ["dropbacks", "epa", "success", "cpoe", "sack_rate", "int_rate", "low_pct_rate", "adverse_epa", "calm_epa", "composure_delta", "adverse_low_pct", "adverse_sack", "adverse_int"]
show = out[cols].copy(); show["rank_epa"] = out.epa.rank(ascending=False); show["rank_composure"] = out.composure_delta.rank(ascending=False)
print(show.round(3).to_string())
if NAMES:
    for n in NAMES:
        if n in out.index:
            r = out.loc[n]
            print(f"\n{n}: by situation (EPA/dropback, n)")
            for s in SITS:
                print(f"  {s:16s} {r[f'{s}_epa']:+.3f} (n={int(r[f'{s}_n'])})   league {lg[f'{s}_epa']:+.3f}")
