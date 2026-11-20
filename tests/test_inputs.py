"""Input hygiene: overrides land, injuries map, weather maps, nothing from the target week leaks into features."""
import numpy as np
import pandas as pd
import pytest

from conftest import ROOT


def test_overrides_applied(feat, season, week):
    p = ROOT / "data" / f"overrides_{season}.csv"
    if not p.exists():
        pytest.skip("no overrides file")
    ov = pd.read_csv(p)
    ov = ov[(ov.week == week) | (ov.week == 0)]
    outs = ov[ov.status.fillna("") == "Out"]
    f = feat.set_index("player")
    for r in outs.itertuples():
        if r.player in f.index:
            assert (f.loc[[r.player], "status"] == "Out").all(), f"override Out not applied: {r.player}"
    if "new_qb" in ov:
        proj = pd.read_parquet(ROOT / "results" / f"projections_{season}_wk{week}.parquet")
        for team in ov[ov.new_qb == 1].team.dropna():
            pc = feat[(feat.team == team) & feat.position.isin(["WR", "TE", "RB"])]
            assert (pc.qb_override == 1).all(), f"new_qb override not in features for {team}"
            pp = proj[(proj.team == team) & proj.position.isin(["WR", "TE", "RB"])]
            assert (pp.new_starter == 1).all(), f"new_qb override not applied in projections for {team}"


def test_override_names_resolve(stats, season, week):
    """Every player named in the overrides must exist in the stats file (catches misspellings that silently do nothing)."""
    p = ROOT / "data" / f"overrides_{season}.csv"
    if not p.exists():
        pytest.skip("no overrides file")
    ov = pd.read_csv(p)
    names = set(stats.player_display_name)
    bad = [n for n in ov.player.dropna() if n not in names]
    assert not bad, f"override names not found in stats: {bad}"


def test_doubtful_counts_as_out(feat, season, week):
    inj = pd.read_parquet(ROOT / "data" / f"injuries_{season}.parquet")
    d = inj[(inj.week == week) & (inj.report_status == "Doubtful")]
    f = feat.set_index("player_id")
    for pid in d.gsis_id.dropna():
        if pid in f.index:
            assert (f.loc[[pid], "status"] == "Out").all(), f"Doubtful player not treated as Out: {pid}"


def test_lead_back_out_flag(feat):
    """When a team's lead back is Out, his teammates carry lead_back_out=1 and he does not."""
    for team, g in feat[feat.position == "RB"].groupby("team"):
        lead = g[g.is_lead_back == 1]
        if lead.empty:
            continue
        if (lead.status == "Out").any():
            others = g[g.is_lead_back == 0]
            assert (others.lead_back_out == 1).all(), f"{team}: lead back Out but teammates not flagged"
            assert (lead.lead_back_out == 0).all()


def test_wind_from_weather_file(feat, sched, season, week):
    p = ROOT / "data" / f"weather_{season}_wk{week}.csv"
    if not p.exists():
        pytest.skip("no weather file")
    wx = pd.read_csv(p).dropna(subset=["wind"]).set_index("home_team")
    w = sched[sched.week == week]
    for g in w.itertuples():
        for t in (g.home_team, g.away_team):
            rows = feat[feat.team == t]
            if rows.empty:
                continue
            if str(g.roof) in ("dome", "closed"):
                assert (rows.wind == 0).all(), f"{t}: wind set for a dome game"
            elif g.home_team in wx.index:
                assert np.allclose(rows.wind, wx.loc[g.home_team, "wind"]), f"{t}: feature wind != forecast"


def test_no_target_week_leakage(feat, stats, week):
    """fp_season in the feature table must be computable from weeks < target week only."""
    s = stats[(stats.season_type == "REG") & (stats.week < week)]
    season_avg = s.groupby("player_id").fantasy_points_ppr.mean()
    f = feat[feat.games_prior >= 3].set_index("player_id")
    common = f.index.intersection(season_avg.index)
    diff = (f.loc[common, "fp_season"] - season_avg.loc[common]).abs()
    # allow for partial-game weighting differences; a leak shows as a systematic large gap
    assert (diff < 2.0).mean() > 0.95, f"fp_season disagrees with prior-week average for {(diff >= 2.0).sum()} players"
    played_target = set(stats[(stats.week == week)].player_id)
    # a feature row may exist for a player who later plays the target week; what must not happen is the target week's points inside fp_l3
    wk_pts = stats[stats.week == week].set_index("player_id").fantasy_points_ppr
    exact = f.loc[f.index.intersection(wk_pts.index)]
    nz = exact[(exact.fp_l3.abs() > 0.5) & (wk_pts.loc[exact.index].abs() > 0.5)]
    assert not ((nz.fp_l3 - wk_pts.loc[nz.index]).abs() < 1e-9).any(), "fp_l3 equals the target week's score for some player (leak)"


def test_defense_effect_walk_forward(season, week):
    de = pd.read_parquet(ROOT / "results" / f"def_effect_{season}.parquet")
    row = de[de.week == week]
    assert len(row) == 32
    assert (row.def_delta_n >= 0).all()
    # week-1 values must be prior-season only (no current dropbacks)
    assert (de[de.week == 1].def_delta_n == 0).all()


def test_pedigree_sanity(season):
    p = pd.read_parquet(ROOT / "results" / f"pedigree_{season}.parquet")
    assert p.player_id.is_unique
    assert ((p.pick <= 64) == (p.top64 == 1)).all()
    assert (p.exp >= 0).all()
    assert (p.loc[p.prior_games == 0, "prior_opp_pg"] == 0).all()


def test_season_long_outs_not_on_board(board, season):
    p = ROOT / "data" / f"overrides_{season}.csv"
    if not p.exists():
        pytest.skip("no overrides file")
    ov = pd.read_csv(p)
    ros = set(ov[(ov.week == 0) & (ov.status.fillna("") == "Out")].player)
    on = board[board.player.isin(ros)]
    assert on.empty, f"season-long Out players on the board: {on.player.tolist()}"
