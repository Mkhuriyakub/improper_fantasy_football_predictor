"""Contract tests on the week's outputs: the columns the board and UI depend on, and the invariants every row must satisfy."""
import json
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

PROJ_REQUIRED = ["player", "player_id", "position", "team", "opp", "p10", "p50", "p90", "mean", "fp_season", "fp_l3", "status", "practice",
                 "ceiling_flag", "shootout", "def_tier", "def_qb_delta", "def_adj", "wind_adj", "weather_flag", "def_blitz",
                 "contrib_usage", "contrib_environment", "contrib_matchup", "contrib_pedigree", "def_expl_tercile", "spread", "implied_total"]
BOARD_REQUIRED = ["rank", "tier", "utilization", "player", "position", "team", "opp", "p10", "p50", "p90", "mean", "fp_season", "fp_l3",
                  "snap_l3", "carry_share_l3", "target_share_l3", "rz_share", "def_tier", "shootout", "emerging", "status", "practice", "espn", "actual"]


def test_projection_columns(proj):
    missing = [c for c in PROJ_REQUIRED if c not in proj]
    assert not missing, f"projections missing {missing}"


def test_board_columns(board):
    missing = [c for c in BOARD_REQUIRED if c not in board]
    assert not missing, f"board missing {missing}"


def test_quantiles_ordered(proj):
    bad = proj[(proj.p10 > proj.p50 + 1e-6) | (proj.p50 > proj.p90 + 1e-6)]
    assert bad.empty, f"p10<=p50<=p90 violated for {bad.player.tolist()[:10]}"


def test_numbers_finite(proj):
    for c in ["p10", "p50", "p90", "mean", "def_adj", "wind_adj"]:
        assert np.isfinite(proj[c]).all(), f"{c} has NaN/inf"


def test_mean_in_plausible_range(proj):
    assert proj["mean"].between(-5, 45).all(), "mean outside -5..45"


def test_no_out_players_on_board(board):
    assert not (board.status == "Out").any(), board[board.status == "Out"].player.tolist()


def test_no_bye_teams_on_board(board, sched, week):
    w = sched[sched.week == week]
    playing = set(w.home_team) | set(w.away_team)
    off = board[~board.team.isin(playing)]
    assert off.empty, f"players on bye on the board: {off.player.tolist()}"


def test_board_ranks_and_counts(board):
    limits = {"QB": 32, "RB": 40, "WR": 60, "TE": 24}
    for pos, n in limits.items():
        b = board[board.position == pos]
        assert len(b) <= n
        assert list(b["rank"]) == list(range(1, len(b) + 1)), f"{pos} ranks not consecutive"
    assert not board.duplicated(["player", "position"]).any()


def test_board_sorted_by_projection(board):
    for pos, b in board.groupby("position"):
        assert (b["p50"].diff().dropna() <= 1e-9).all(), f"{pos} board not sorted by projection (p50)"


def test_tier_buckets_match_rank(board):
    for r in board.itertuples():
        bucket = int(r.tier.split("-")[0][len(r.position):])
        assert bucket == (r.rank - 1) // 12 + 1, f"{r.player}: tier {r.tier} vs rank {r.rank}"


def test_soft_run_d_flag_matches_defense_table(proj, season, week):
    ex = pd.read_parquet(ROOT / "results" / f"expl_def_{season}.parquet")
    soft = set(ex[(ex.week == week) & (ex.def_expl_tercile == "soft")].defteam)
    rb = proj[proj.position == "RB"]
    should = rb.opp.isin(soft)
    has = rb.ceiling_flag.fillna("") == "soft run D"
    mism = rb[should != has]
    assert mism.empty, f"soft-run-D flag mismatch for {mism[['player', 'opp']].values.tolist()[:10]}"


def test_emerging_excludes_established(board):
    for pos, b in board.groupby("position"):
        rank = b.fp_season.rank(ascending=False)
        bad = b[(b.emerging == 1) & ((rank <= 12) | ((b.fp_l3 - b.fp_season) < 4))]
        assert bad.empty, f"{pos}: emerging flag on {bad.player.tolist()}"


def test_defense_adjustment_identity(proj):
    diff = (proj.def_adj - (10 * proj.def_qb_delta + proj.wind_adj)).abs()
    assert (diff < 1e-6).all(), "def_adj != 10*def_qb_delta + wind_adj"


def test_defense_adjustment_equal_within_team(proj):
    """The opponent effect is applied equally to every pass catcher on an offense (wind differs by target share)."""
    pc = proj[proj.position.isin(["WR", "TE", "RB"])]
    for team, g in pc.groupby("team"):
        assert g.def_qb_delta.nunique() == 1, f"{team}: different def_qb_delta within one offense"


def test_wind_adjustment_sign_and_domes(proj, feat):
    assert (proj.wind_adj <= 1e-9).all(), "wind adjustment should never add points"
    f = feat.set_index("player_id")
    dome = proj[proj.player_id.map(f.dome) == 1]
    assert (dome.wind_adj.abs() < 1e-9).all(), "wind adjustment applied to a dome game"


def test_extreme_weather_applied(proj, feat):
    f = feat.set_index("player_id")
    ext = proj[proj.player_id.map(f.get("extreme", pd.Series(dtype=float))).fillna(0) == 1]
    if ext.empty:
        pytest.skip("no extreme-weather games this week")
    assert (ext.weather_flag != "").all()
    qb = ext[ext.position == "QB"]
    assert (qb.wind_adj <= -4 + 1e-6).all()


def test_ui_exports_match_board(board, season, week):
    d = ROOT / "ui" / "data"
    if not (d / "meta.json").exists():
        pytest.skip("no UI export")
    meta = json.load(open(d / "meta.json"))
    assert meta["week"] == week and meta["season"] == season
    b = json.load(open(d / "board.json"))
    assert len(b) == len(board)
    assert {r["player"] for r in b} == set(board.player)
    for key in ["p10", "p50", "p90", "mean", "tier", "utilization", "ceiling_flag", "def_adj", "wind_adj", "emerging", "contrib_pedigree", "pick", "prior_fp_per_opp"]:
        assert key in b[0], f"UI board.json missing {key}"
    games = json.load(open(d / "games.json"))
    sched = pd.read_parquet(ROOT / "data" / f"schedules_{season}.parquet")
    assert len(games) == (sched.week == week).sum()
    defs = json.load(open(d / "defenses.json"))
    assert len(defs) == 32


def test_one_quarterback_per_team(board):
    q = board[board.position == "QB"]
    dup = q[q.team.duplicated(keep=False)]
    assert dup.empty, f"more than one QB listed for {sorted(dup.team.unique())}"
