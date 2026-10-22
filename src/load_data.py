"""Download a season of free nflverse data into data/.

Usage:  python src/load_data.py 2025

Pulls play-by-play, FTN charting, play-level participation (available after a
season ends; includes box counts, personnel, pressure, routes and coverage),
snap counts, injury reports, schedules (with Vegas lines), weekly player stats
and weekly rosters. All data is CC-BY 4.0 (FTN charting CC-BY-SA 4.0).
"""
import sys
from pathlib import Path

import nflreadpy as nfl

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

LOADERS = {
    "pbp": lambda s: nfl.load_pbp(s),
    "ftn": lambda s: nfl.load_ftn_charting(s),
    "participation": lambda s: nfl.load_participation(s),
    "snaps": lambda s: nfl.load_snap_counts(s),
    "injuries": lambda s: nfl.load_injuries(s),
    "schedules": lambda s: nfl.load_schedules(s),
    "player_stats": lambda s: nfl.load_player_stats(s),
    "rosters": lambda s: nfl.load_rosters_weekly(s),
}

for name, fn in LOADERS.items():
    try:
        df = fn(SEASON).to_pandas()
        df.to_parquet(OUT / f"{name}_{SEASON}.parquet")
        print(f"{name:14s} {df.shape}")
    except Exception as e:  # participation lags a season; keep going
        print(f"{name:14s} unavailable: {e}")
