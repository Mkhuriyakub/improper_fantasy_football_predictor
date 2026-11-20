import glob, re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def latest_week(season=2026):
    weeks = [int(re.search(r"wk(\d+)", p).group(1)) for p in glob.glob(str(ROOT / "results" / f"projections_{season}_wk*.parquet"))]
    if not weeks:
        pytest.skip("no projections built yet")
    return max(weeks)


@pytest.fixture(scope="session")
def season():
    return 2026


@pytest.fixture(scope="session")
def week(season):
    return latest_week(season)


@pytest.fixture(scope="session")
def proj(season, week):
    return pd.read_parquet(ROOT / "results" / f"projections_{season}_wk{week}.parquet")


@pytest.fixture(scope="session")
def feat(season, week):
    return pd.read_parquet(ROOT / "results" / f"features_{season}_wk{week}.parquet")


@pytest.fixture(scope="session")
def board(season, week):
    p = ROOT / "results" / f"board_{season}_wk{week}.csv"
    if not p.exists():
        pytest.skip("no board for latest week")
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def sched(season):
    return pd.read_parquet(ROOT / "data" / f"schedules_{season}.parquet")


@pytest.fixture(scope="session")
def stats(season):
    return pd.read_parquet(ROOT / "data" / f"player_stats_{season}.parquet")
