# Improper Fantasy Football Predictor

A fantasy football projection model built on free data, with a focus on the
things most public projections skip: game shape (pace, possession, script),
player-specific usage sensitivity, offensive and defensive line play, and
defensive scheme tendencies. Every projection is a distribution, not a point,
and every layer of the model is scored against outcomes and against consensus.

## Approach

Projections are built in layers, each tested for whether it adds accuracy:

1. **Naive baseline** – player's trailing 3-week average
2. **Vegas baseline** – adjusted by implied team total and spread
3. **Usage model** – pace, possession, script sensitivity, injuries, depth chart
4. **Full model** – adds line matchups and defensive scheme tendencies

The first test is a walk-forward backtest on the 2025 season: for each week
from 7 on, project every player using only prior weeks, then score the result.

## Data

All from [nflverse](https://github.com/nflverse) via `nflreadpy`:
play-by-play, FTN charting, play-level participation (prior seasons only),
snap counts, injury reports, schedules with Vegas lines, weekly player stats.

```
pip install nflreadpy pandas pyarrow
python src/load_data.py 2025
```

## Layout

- `src/` – data loading, feature building, models, scoring
- `data/` – downloaded parquet files (not committed; regenerate with `load_data.py`)
- `results/` – metric tables and backtest output
- `notebooks/` – exploratory analysis
