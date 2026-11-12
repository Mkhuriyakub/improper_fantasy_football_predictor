"""Walk-forward feature builder for all skill positions (RB, WR, TE, QB).

Works on any season using only free nflverse data. Box counts and pass-rusher
counts come from FTN charting (available in-season); play-level participation
(prior seasons only) is used when present for pressure and coverage.

For target week W (optional), builds rows for W using weeks < W, including
players who have a game scheduled in W even if they haven't played yet that
week. Without a target week, builds rows for every week >= 4 (backtesting).

Usage:
  python src/features.py 2025            -> results/features_2025.parquet
  python src/features.py 2026 7          -> results/features_2026_wk7.parquet
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
TARGET_WEEK = int(sys.argv[2]) if len(sys.argv) > 2 else None
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load(name):
    p = D / f"{name}_{SEASON}.parquet"
    return pd.read_parquet(p) if p.exists() else None


pbp = load("pbp")
pbp = pbp[pbp.season_type == "REG"].copy()
ftn = load("ftn").rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
part = load("participation")
stats = load("player_stats")
stats = stats[stats.season_type == "REG"].copy()
sched = load("schedules")
sched = sched[sched.game_type == "REG"].copy()
inj = load("injuries")
rost = load("rosters")
snaps = load("snaps")
SNAP_WEIGHT = os.environ.get("SNAP_WEIGHT", "1") == "1"
_idmap = rost.drop_duplicates("gsis_id")[["gsis_id", "pfr_id"]].dropna()
_sn = snaps.merge(_idmap, left_on="pfr_player_id", right_on="pfr_id") if snaps is not None else None
SNAP = {(r.gsis_id, int(r.week)): float(r.offense_pct) for r in _sn.itertuples()} if _sn is not None else {}


pbp["play_id"] = pbp.play_id.astype(float)
ftn["play_id"] = ftn.play_id.astype(float)
pbp = pbp.merge(ftn[["game_id", "play_id", "n_defense_box", "n_pass_rushers", "n_offense_backfield",
                     "is_qb_fault_sack", "is_screen_pass"]], on=["game_id", "play_id"], how="left")
if part is not None:
    part["play_id"] = part.play_id.astype(float)
    pbp = pbp.merge(part[["nflverse_game_id", "play_id", "was_pressure", "defense_man_zone_type"]],
                    left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left")
else:
    pbp["was_pressure"] = np.nan
    pbp["defense_man_zone_type"] = np.nan

plays = pbp[pbp.play_type.isin(["run", "pass"]) & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1)
            & pbp.posteam.notna()].copy()
plays["is_run"] = (plays.play_type == "run").astype(int)
plays["trailing"] = plays.score_differential <= -8
plays["leading"] = plays.score_differential >= 8
plays["neutral"] = plays.score_differential.abs() < 8
plays["light_box"] = plays.n_defense_box <= 6
plays["one_back"] = plays.n_offense_backfield <= 1
plays["blitz"] = plays.n_pass_rushers >= 5
plays["rz"] = plays.yardline_100 <= 20
passers = set(plays.loc[plays.pass_attempt == 1, "passer_player_id"].dropna())
runs = plays[(plays.is_run == 1) & (plays.qb_scramble != 1) & (~plays.rusher_player_id.isin(passers))].copy()
runs["line_yds"] = runs.yards_gained.clip(upper=10).where(runs.yards_gained >= 0, runs.yards_gained * 1.2)
runs["line_yds"] = np.where(runs.yards_gained > 4, 4 + 0.5 * (runs.yards_gained.clip(upper=10) - 4), runs.line_yds)
runs["stuff"] = (runs.yards_gained <= 0).astype(int)
runs["explosive"] = (runs.yards_gained >= 10).astype(int)
dropbacks = plays[plays.qb_dropback == 1].copy()
targets = plays[(plays.pass_attempt == 1) & plays.receiver_player_id.notna()].copy()
drives = plays.groupby(["game_id", "posteam", "defteam", "week", "drive"]).agg(n=("play_id", "size")).reset_index()
drives["three_out"] = drives.n <= 3
drives["long"] = drives.n >= 8

pos_map = rost.drop_duplicates("gsis_id").set_index("gsis_id").position
RB_IDS = set(stats.loc[stats.position == "RB", "player_id"])

# ---------------------------------------------------------------- team profiles


def team_profile(w):
    p, r, db, dr, tg = (x[x.week < w] for x in (plays, runs, dropbacks, drives, targets))
    out = {}
    for t in p.posteam.unique():
        o, orun, odb, odr = p[p.posteam == t], r[r.posteam == t], db[db.posteam == t], dr[dr.posteam == t]
        d, drun, ddb, ddr, dtg = p[p.defteam == t], r[r.defteam == t], db[db.defteam == t], dr[dr.defteam == t], tg[tg.defteam == t]
        gm = max(1, o.game_id.nunique())
        neu = o[o.neutral & (o.down <= 2)]
        out[t] = dict(
            off_plays_pg=len(o) / gm, off_neutral_run_rate=neu.is_run.mean(), off_proe=o.pass_oe.mean() / 100,
            off_plays_per_drive=odr.n.mean(), off_long_drive_rate=odr.long.mean(), off_epa=o.epa.mean(),
            off_pass_epa=odb.epa.mean(), off_pass_sr=odb.success.mean(), off_dropbacks_pg=len(odb) / gm,
            off_rz_plays_pg=o.rz.sum() / gm, off_line_yds=orun.line_yds.mean(), off_stuff=orun.stuff.mean(),
            off_light_box_ypc_gain=orun.loc[orun.light_box, "yards_gained"].mean() - orun.loc[~orun.light_box, "yards_gained"].mean(),
            def_plays_allowed_pg=len(d) / gm, def_three_out_rate=ddr.three_out.mean(), def_long_drive_allowed=ddr.long.mean(),
            def_epa_allowed=d.epa.mean(), def_pass_epa_allowed=ddb.epa.mean(), def_pass_sr_allowed=ddb.success.mean(),
            def_rush_epa_allowed=drun.epa.mean(), def_line_yds_allowed=drun.line_yds.mean(), def_stuff_rate=drun.stuff.mean(),
            def_explosive_run_allowed=drun.explosive.mean(), def_blitz_rate=ddb.blitz.mean(), def_pressure_rate=ddb.was_pressure.mean(),
            def_sack_rate=ddb.sack.mean(), def_box_vs_one_back=drun.loc[drun.one_back, "n_defense_box"].mean(),
            def_light_box_rate=drun.light_box.mean(), def_rb_target_share=dtg.receiver_player_id.isin(RB_IDS).mean(),
            def_te_target_share=dtg.receiver_player_id.map(pos_map).eq("TE").mean(),
            def_man_rate=(ddb.defense_man_zone_type == "MAN_COVERAGE").mean() if ddb.defense_man_zone_type.notna().any() else np.nan,
            def_ypa_allowed=ddb.loc[ddb.pass_attempt == 1, "yards_gained"].mean(),
            opp_off_plays_pg=len(o) / gm, opp_off_run_rate=o.is_run.mean(), opp_off_plays_per_drive=odr.n.mean(), opp_off_epa=o.epa.mean(),
        )
    return pd.DataFrame(out).T


# ---------------------------------------------------------------- player weekly
sk = stats[stats.position.isin(["RB", "WR", "TE", "QB"])].copy()
sk["touches"] = sk.carries.fillna(0) + sk.receptions.fillna(0)
sk["fp"] = sk.fantasy_points_ppr.fillna(0)
sk["fp_std"] = sk.fantasy_points.fillna(0)
sk["rush_fp"] = sk.rushing_yards.fillna(0) * 0.1 + sk.rushing_tds.fillna(0) * 6 - sk.rushing_fumbles_lost.fillna(0) * 2
sk["rec_fp"] = sk.receiving_yards.fillna(0) * 0.1 + sk.receiving_tds.fillna(0) * 6 + sk.receptions.fillna(0)
sk = sk.sort_values(["player_id", "week"])

# script-state usage (touches/targets by state) from pbp
plays["ev_id"] = np.where(plays.is_run == 1, plays.rusher_player_id, plays.receiver_player_id)
ev = plays[plays.ev_id.notna()]
team_state = plays.groupby(["posteam", "week"]).agg(plays=("play_id", "size"), trail_plays=("trailing", "sum"),
                                                     neutral_plays=("neutral", "sum")).reset_index()
pl_state = ev.groupby(["ev_id", "posteam", "week"]).agg(
    touches=("play_id", "size"), trail_touches=("trailing", "sum"), neutral_touches=("neutral", "sum"),
    rz_touches=("rz", "sum"), gl=("yardline_100", lambda s: (s <= 10).sum()),
    air=("air_yards", "sum")).reset_index().merge(team_state, on=["posteam", "week"])
team_tg = targets.groupby(["posteam", "week"]).agg(team_targets=("play_id", "size"), team_air=("air_yards", "sum")).reset_index()

# vegas
h = sched[["week", "home_team", "away_team", "spread_line", "total_line", "roof"]].rename(columns={"home_team": "team", "away_team": "opp"})
h["spread"] = -h.spread_line
a = sched[["week", "home_team", "away_team", "spread_line", "total_line", "roof"]].rename(columns={"away_team": "team", "home_team": "opp"})
a["spread"] = a.spread_line
vegas = pd.concat([h, a])
vegas["implied_total"] = vegas.total_line / 2 - vegas.spread / 2
inj_w = inj[["week", "team", "gsis_id", "report_status", "practice_status"]].rename(columns={"gsis_id": "player_id"}) if inj is not None else None

weeks = [TARGET_WEEK] if TARGET_WEEK else [w for w in sorted(plays.week.unique()) if w >= 4]
rows = []
for w in weeks:
    prof = team_profile(w)
    hist = sk[sk.week < w]
    hs = pl_state[pl_state.week < w]
    cur = vegas[vegas.week == w]
    last_team = hist.groupby("player_id").team.last()
    for pid, team in last_team.items():
        g = cur[cur.team == team]
        if g.empty:
            continue
        opp = g.opp.iat[0]
        ph = hist[hist.player_id == pid]
        pos = ph.position.iat[-1]
        pe = hs[hs.ev_id == pid]
        l3 = ph.tail(3)
        wts = np.array([SNAP.get((pid, int(wk)), 1.0) if SNAP_WEIGHT else 1.0 for wk in l3.week]) if len(l3) else np.array([])
        wts = np.where(np.isnan(wts), 1.0, wts); wts = np.clip(wts, 0.02, 1.0)
        wm = (lambda col: float(np.average(col.fillna(0).values, weights=wts))) if len(l3) and wts.sum() > 0 else (lambda col: np.nan)
        tm = hist[hist.team == team]
        tt = team_tg[(team_tg.posteam == team) & team_tg.week.isin(l3.week)]
        team_touch = tm[tm.week.isin(l3.week)].groupby("week").touches.sum()
        share3 = float(np.average((l3.set_index("week").touches / team_touch).fillna(0).values, weights=wts)) if len(l3) else np.nan
        team_carries = tm[tm.week.isin(l3.week) & (tm.position != "QB")].groupby("week").carries.sum()
        carry_share3 = float(np.average((l3.set_index("week").carries / team_carries.replace(0, np.nan)).fillna(0).values, weights=wts)) if len(l3) else np.nan
        snaps3 = [SNAP.get((pid, int(wk)), np.nan) for wk in l3.week]
        snap_last = snaps3[-1] if len(snaps3) else np.nan
        snap_l3 = float(np.nanmean(snaps3)) if len(snaps3) and not all(np.isnan(snaps3)) else np.nan
        snap_trend = (snaps3[-1] - np.nanmean(snaps3[:-1])) if len(snaps3) >= 2 and not np.isnan(snaps3[-1]) and not all(np.isnan(snaps3[:-1])) else 0.0
        _tt = tt.set_index("week").reindex(l3.week); tgt_share3 = float(np.average((l3.targets.fillna(0).values / _tt.team_targets.replace(0, np.nan).fillna(1).values), weights=wts)) if len(l3) and tt.team_targets.sum() else np.nan
        air_share3 = float(np.average((l3.receiving_air_yards.fillna(0).values / _tt.team_air.replace(0, np.nan).fillna(1).values), weights=wts)) if len(l3) and tt.team_air.sum() else np.nan
        trail_share = pe.trail_touches.sum() / pe.trail_plays.sum() if pe.trail_plays.sum() >= 30 else np.nan
        neutral_share = pe.neutral_touches.sum() / pe.neutral_plays.sum() if pe.neutral_plays.sum() >= 30 else np.nan
        recent = tm[tm.week >= w - 3]
        lead_rb = recent[recent.position == "RB"].groupby("player_id").touches.sum().idxmax() if len(recent[recent.position == "RB"]) else None
        lead_wr = recent[recent.position.isin(["WR", "TE"])].groupby("player_id").targets.sum().idxmax() if len(recent[recent.position.isin(["WR", "TE"])]) else None
        status = inj_w[(inj_w.week == w) & (inj_w.player_id == pid)] if inj_w is not None else pd.DataFrame()
        ls = inj_w[(inj_w.week == w) & (inj_w.player_id == lead_rb)] if inj_w is not None else pd.DataFrame()
        lws = inj_w[(inj_w.week == w) & (inj_w.player_id == lead_wr)] if inj_w is not None else pd.DataFrame()
        actual = sk[(sk.week == w) & (sk.player_id == pid)]
        prof_o = prof.loc[team] if team in prof.index else pd.Series(dtype=float)
        prof_d = prof.loc[opp] if opp in prof.index else pd.Series(dtype=float)
        rows.append(dict(
            season=SEASON, week=w, player_id=pid, player=ph.player_display_name.iat[-1], position=pos, team=team, opp=opp,
            games_prior=len(ph), weeks_since_last=w - ph.week.iat[-1],
            fp_l3=wm(l3.fp), fp_season=ph.fp.mean(), fp_std_l3=wm(l3.fp_std), rush_fp_l3=wm(l3.rush_fp), rec_fp_l3=wm(l3.rec_fp),
            touches_l3=wm(l3.touches), carries_l3=wm(l3.carries), targets_l3=wm(l3.targets),
            attempts_l3=wm(l3.attempts), pass_yds_l3=wm(l3.passing_yards), pass_td_l3=wm(l3.passing_tds),
            rush_yds_l3=wm(l3.rushing_yards), rec_yds_l3=wm(l3.receiving_yards), rec_l3=wm(l3.receptions),
            adot_l3=(l3.receiving_air_yards.sum() / max(1, l3.targets.sum())) if l3.targets.sum() else np.nan,
            touch_share_l3=share3, target_share_l3=tgt_share3, air_share_l3=air_share3, carry_share_l3=carry_share3,
            snap_last=snap_last, snap_l3=snap_l3, snap_trend=snap_trend,
            gl_share=pe.gl.sum() / max(1, pe.touches.sum()), rz_share=pe.rz_touches.sum() / max(1, pe.touches.sum()),
            trail_share=trail_share, neutral_share=neutral_share,
            script_sens=(trail_share - neutral_share) if pd.notna(trail_share) and pd.notna(neutral_share) else 0.0,
            status=status.report_status.iat[0] if len(status) and pd.notna(status.report_status.iat[0]) else "none",
            practice=status.practice_status.iat[0] if len(status) and pd.notna(status.practice_status.iat[0]) else "none",
            is_lead_back=int(lead_rb == pid), lead_back_out=int(len(ls) and ls.report_status.iat[0] == "Out" and lead_rb != pid),
            is_lead_receiver=int(lead_wr == pid), lead_receiver_out=int(len(lws) and lws.report_status.iat[0] == "Out" and lead_wr != pid),
            spread=g.spread.iat[0], total_line=g.total_line.iat[0], implied_total=g.implied_total.iat[0],
            dome=int(str(g.roof.iat[0]) in ("dome", "closed")),
            **{k: prof_o.get(k, np.nan) for k in prof.columns if k.startswith("off_")},
            **{"opp_" + k: prof_d.get(k, np.nan) for k in prof.columns if k.startswith("def_")},
            **{k: prof_d.get(k, np.nan) for k in prof.columns if k.startswith("opp_off_")},
            fp=actual.fp.iat[0] if len(actual) else np.nan, fp_std=actual.fp_std.iat[0] if len(actual) else np.nan,
            rush_fp=actual.rush_fp.iat[0] if len(actual) else np.nan, rec_fp=actual.rec_fp.iat[0] if len(actual) else np.nan,
            touches=actual.touches.iat[0] if len(actual) else np.nan, carries=actual.carries.iat[0] if len(actual) else np.nan,
            targets=actual.targets.iat[0] if len(actual) else np.nan, played=int(len(actual) > 0),
        ))

feat = pd.DataFrame(rows)

# manual overrides for news the injury file doesn't carry yet: data/overrides_{season}.csv
# columns: week, player, status (Out) | week, team, new_qb (1) -> flags pass catchers as new-starter weeks
_ov = ROOT / "data" / f"overrides_{SEASON}.csv"
if _ov.exists():
    ov = pd.read_csv(_ov)
    for r in ov[ov.get("status", pd.Series(dtype=str)).notna()].itertuples():
        m = (feat.week == r.week) & (feat.player == r.player)
        feat.loc[m, "status"] = r.status
        if r.status == "Out" and m.any():
            team, pos = feat.loc[m, "team"].iat[0], feat.loc[m, "position"].iat[0]
            tm = (feat.week == r.week) & (feat.team == team) & ~m
            if pos == "RB" and feat.loc[m, "is_lead_back"].iat[0] == 1:
                feat.loc[tm, "lead_back_out"] = 1
            if pos in ("WR", "TE") and feat.loc[m, "is_lead_receiver"].iat[0] == 1:
                feat.loc[tm, "lead_receiver_out"] = 1
    if "new_qb" in ov:
        for r in ov[ov.new_qb == 1].itertuples():
            tm = (feat.week == r.week) & (feat.team == r.team) & (feat.position != "QB")
            for c in ("new_starter", "qb_changed"):
                if c in feat:
                    feat.loc[tm, c] = 1
    print(f"overrides applied: {len(ov)} rows")
feat["p_trail_proxy"] = 1 / (1 + np.exp(-feat.spread / 6))
feat["script_risk"] = feat.p_trail_proxy * feat.script_sens
feat["pace_mismatch"] = feat.opp_off_plays_per_drive - feat.off_plays_per_drive
feat["blitz_x_targets"] = feat.opp_def_blitz_rate * feat.targets_l3
feat["line_edge"] = feat.off_line_yds - feat.opp_def_line_yds_allowed
feat["box_x_light_gain"] = feat.opp_def_light_box_rate * feat.off_light_box_ypc_gain
feat["pass_edge"] = feat.off_pass_epa - (-feat.opp_def_pass_epa_allowed)  # offense strength + defense weakness
name = f"features_{SEASON}" + ("" if SNAP_WEIGHT else "_unw") + (f"_wk{TARGET_WEEK}" if TARGET_WEEK else "")
feat.to_parquet(OUT / f"{name}.parquet", index=False)
print(name, feat.shape, feat.position.value_counts().to_dict())
