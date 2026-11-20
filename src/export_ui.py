"""Export the week's board, picks, defense profiles and metadata as JSON for the web UI.

Usage: python src/export_ui.py 2026 11  -> ui/data/{board,picks,defenses,meta}.json
"""
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEASON, WEEK = int(sys.argv[1]), int(sys.argv[2])
OUT = ROOT / "ui" / "data"; OUT.mkdir(parents=True, exist_ok=True)

board = pd.read_csv(ROOT / "results" / f"board_{SEASON}_wk{WEEK}.csv")
proj = pd.read_parquet(ROOT / "results" / f"projections_{SEASON}_wk{WEEK}.parquet")
feat = pd.read_parquet(ROOT / "results" / f"features_{SEASON}_wk{WEEK}.parquet")
ped = pd.read_parquet(ROOT / "results" / f"pedigree_{SEASON}.parquet")
fcols = [c for c in ["player_id", "wind", "precip", "extreme", "games_prior", "touches_l3", "carries_l3", "targets_l3", "adot_l3", "air_share_l3", "gl_share", "lead_back_out", "lead_receiver_out", "new_starter", "qb_changed"] if c in feat]
extra = proj.drop(columns=[c for c in proj.columns if c in feat.columns and c not in ("player_id",)], errors="ignore").merge(feat[fcols], on="player_id", how="left").merge(ped, on="player_id", how="left")
for c in ("player", "position", "team", "opp", "spread", "implied_total", "status", "practice"):
    if c not in extra and c in proj:
        extra[c] = proj.set_index("player_id").loc[extra.player_id, c].values
cols = ["player", "position", "team", "opp", "p10", "p50", "p90", "mean", "fp_season", "fp_l3", "def_tier", "def_qb_delta", "def_adj", "wind_adj", "weather_flag", "def_blitz", "ceiling_flag", "shootout", "contrib_usage", "contrib_environment", "contrib_matchup", "contrib_pedigree", "spread", "implied_total", "wind", "precip", "extreme", "games_prior", "touches_l3", "carries_l3", "targets_l3", "adot_l3", "air_share_l3", "gl_share", "lead_back_out", "lead_receiver_out", "new_starter", "pick", "exp", "prior_fp_per_opp", "prior_opp_pg", "status", "practice"]
extra = extra[[c for c in cols if c in extra]]
b = board.merge(extra.drop(columns=[c for c in extra if c in board and c not in ("player", "position")]), on=["player", "position"], how="left")
b = b.replace({np.nan: None})
rows = []
for r in b.to_dict("records"):
    rows.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(rows, open(OUT / "board.json", "w"))

pk = ROOT / "results" / f"picks_{SEASON}_wk{WEEK}.csv"
picks = pd.read_csv(pk).replace({np.nan: None}).to_dict("records") if pk.exists() else []
sched = pd.read_parquet(ROOT / "data" / f"schedules_{SEASON}.parquet"); w = sched[sched.week == WEEK]
wx = ROOT / "data" / f"weather_{SEASON}_wk{WEEK}.csv"; wxd = pd.read_csv(wx).set_index("home_team").to_dict("index") if wx.exists() else {}
games = []
for g in w.itertuples():
    p = next((x for x in picks if x["away"] == g.away_team and x["home"] == g.home_team), {})
    wd = wxd.get(g.home_team, {})
    games.append(dict(away=g.away_team, home=g.home_team, gameday=str(g.gameday), gametime=str(g.gametime), spread=None if pd.isna(g.spread_line) else float(g.spread_line), total=None if pd.isna(g.total_line) else float(g.total_line),
                      roof=str(g.roof), stadium=str(g.stadium), away_score=None if pd.isna(g.away_score) else float(g.away_score), home_score=None if pd.isna(g.home_score) else float(g.home_score),
                      pick=p.get("pick"), prob=p.get("prob"), note=p.get("note"), wind=None if pd.isna(wd.get("wind", np.nan)) else float(wd.get("wind")), precip=None if pd.isna(wd.get("precip", np.nan)) else float(wd.get("precip")), extreme=int(wd.get("extreme", 0) or 0)))
json.dump(games, open(OUT / "games.json", "w"))

de = pd.read_parquet(ROOT / "results" / f"def_effect_{SEASON}.parquet"); de = de[de.week == WEEK]
dt = pd.read_parquet(ROOT / "results" / f"def_tiers_{SEASON}.parquet"); dt = dt[dt.week == WEEK]
ex = pd.read_parquet(ROOT / "results" / f"expl_def_{SEASON}.parquet"); ex = ex[ex.week == WEEK] if "week" in ex else ex
d = de.merge(dt[["defteam", "def_rel_epa", "def_tier"]], on="defteam", how="left").merge(ex[["defteam", "def_expl_sus", "def_expl_tercile"]], on="defteam", how="left")
json.dump(d.replace({np.nan: None}).round(4).to_dict("records"), open(OUT / "defenses.json", "w"))

meta = dict(season=SEASON, week=WEEK, built=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), n_players=len(rows), n_games=len(games))
json.dump(meta, open(OUT / "meta.json", "w"))
print(meta)
