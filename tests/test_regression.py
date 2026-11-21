"""Model regression: the 2025 walk-forward validation (top-N + emerging players) must not get worse than the frozen baseline.

The projection script writes results/validation_2025_latest.json on every run. tests/validation_baseline.json is the
accepted state. A code change that makes any position's chosen model worse than the baseline by more than the tolerance
fails here. To accept a change deliberately: python tests/freeze_baseline.py
"""
import json
import pytest

from conftest import ROOT

TOL_MAE = 0.03    # points
TOL_RANK = 0.015  # Spearman


@pytest.fixture(scope="module")
def latest():
    p = ROOT / "results" / "validation_2025_latest.json"
    if not p.exists():
        pytest.skip("no validation output; run project.py")
    return json.load(open(p))


@pytest.fixture(scope="module")
def baseline():
    return json.load(open(ROOT / "tests" / "validation_baseline.json"))


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
def test_chosen_model_not_worse(latest, baseline, pos):
    cur, base = latest[pos], baseline[pos]
    c, b = cur[cur["chosen"]], base[base["chosen"]]
    assert c["MAE"] <= b["MAE"] + TOL_MAE, f"{pos}: MAE {c['MAE']} vs baseline {b['MAE']} ({base['chosen']})"
    assert c["rank"] >= b["rank"] - TOL_RANK, f"{pos}: rank {c['rank']} vs baseline {b['rank']}"


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
def test_chosen_model_beats_season_average(latest, pos):
    cur = latest[pos]
    assert cur[cur["chosen"]]["MAE"] <= cur["season"]["MAE"] + 0.10, f"{pos}: chosen model does not beat the season-average baseline"


@pytest.mark.parametrize("pos", ["QB", "RB", "WR", "TE"])
def test_training_sample_stable(latest, baseline, pos):
    """A big change in n means the training population changed (filters, data), which makes the MAE comparison meaningless."""
    assert abs(latest[pos]["n"] - baseline[pos]["n"]) <= 0.05 * baseline[pos]["n"], f"{pos}: training n {latest[pos]['n']} vs {baseline[pos]['n']}"
