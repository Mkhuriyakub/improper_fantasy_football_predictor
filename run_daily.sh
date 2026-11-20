#!/usr/bin/env bash
# Daily refresh: data, traits, features, projections, board, for the target week.
# Usage: ./run_daily.sh 2026 11
set -e
S=${1:-2026}; W=${2}
cd "$(dirname "$0")"
pip install -q --break-system-packages nflreadpy pandas pyarrow scikit-learn statsmodels scipy >/dev/null 2>&1 || true
python src/load_data.py 2025; python src/load_data.py 2024; python src/load_data.py $S
python - <<'PY'
import nflreadpy as nfl, os
for y in range(2015, 2024):
    p=f'data/player_stats_{y}.parquet'
    if not os.path.exists(p): nfl.load_player_stats(y).to_pandas().to_parquet(p,index=False)
if not os.path.exists('data/draft_picks.parquet'): nfl.load_draft_picks().to_pandas().to_parquet('data/draft_picks.parquet',index=False)
PY
for y in 2025 $S; do python src/expl_traits.py $y; python src/qb_traits.py $y; python src/def_tiers.py $y; python src/def_effect.py $y; python src/pedigree.py $y; done
[ -f results/features_2025.parquet ] || python src/features.py 2025
python src/features.py $S $W
python src/project.py 2025 $S $W
python src/board.py $S $W
python src/export_ui.py $S $W
./run_checks.sh
