"""Walk-forward feature table for running backs.

One row per RB per regular-season week. Every feature uses ONLY games played
before that week, so a row for week w is what you could have known on the
Wednesday before it.

Usage: python src/features_rb.py 2025   -> results/features_rb_2025.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
pbp = pd.read_parquet(D / f"pbp_{SEASON}.parquet")
pbp = pbp[pbp.season_type == "REG"].copy()
part = pd.read_parquet(D / f"participation_{SEASON}.parquet")
ftn = pd.read_parquet(D / f"ftn_{SEASON}.parquet").rename(
    columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
stats = pd.read_parquet(D / f"player_stats_{SEASON}.parquet")
stats = stats[stats.season_type == "REG"].copy()
sched = pd.read_parquet(D / f"schedules_{SEASON}.parquet")
sched = sched[sched.game_type == "REG"].copy()
inj = pd.read_parquet(D / f"injuries_{SEASON}.parquet")
snaps = pd.read_parquet(D / f"snaps_{SEASON}.parquet")

pbp["play_id"] = pbp.play_id.astype(float)
part["play_id"] = part.play_id.astype(float)
ftn["play_id"] = ftn.play_id.astype(float)
pbp = pbp.merge(part[["nflverse_game_id", "play_id", "defenders_in_box", "offense_personnel",
                      "number_of_pass_rushers", "was_pressure", "defense_man_zone_type"]],
                left_on=["game_id", "play_id"], right_on=["nflverse_game_id", "play_id"], how="left")
pbp = pbp.merge(ftn[["game_id", "play_id", "n_offense_backfield", "is_qb_fault_sack"]],
                on=["game_id", "play_id"], how="left")

plays = pbp[pbp.play_type.isin(["run", "pass"]) & (pbp.qb_kneel != 1) & (pbp.qb_spike != 1)
            & pbp.posteam.notna()].copy()
plays["is_run"] = (plays.play_type == "run").astype(int)
plays["trailing"] = plays.score_differential <= -8
plays["leading"] = plays.score_differential >= 8
plays["neutral"] = plays.score_differential.abs() < 8
plays["light_box"] = plays.defenders_in_box <= 6
plays["one_back"] = plays.n_offense_backfield <= 1
plays["blitz"] = plays.number_of_pass_rushers >= 5
passers = set(plays.loc[plays.pass_attempt == 1, "passer_player_id"].dropna())
runs = plays[(plays.is_run == 1) & (plays.qb_scramble != 1) & (~plays.rusher_player_id.isin(passers))].copy()


def aly(y):
    if y < 0:
        return 1.2 * y
    if y <= 4:
        return y
    if y <= 10:
        return 4 + 0.5 * (y - 4)
    return 7


runs["line_yds"] = runs.yards_gained.apply(aly)
runs["stuff"] = (runs.yards_gained <= 0).astype(int)
runs["explosive"] = (runs.yards_gained >= 10).astype(int)
dropbacks = plays[plays.qb_dropback == 1].copy()

# drives
drives = plays.groupby(["game_id", "posteam", "defteam", "week", "drive"]).agg(
    n=("play_id", "size")).reset_index()
drives["three_out"] = drives.n <= 3
drives["long"] = drives.n >= 8

WEEKS = sorted(plays.week.unique())
MAX_WEEK = max(WEEKS)


# ----------------------------------------------------------------------------
# team-level tendencies, computed from weeks < w
# ----------------------------------------------------------------------------
def team_profile(w):
    p = plays[plays.week < w]
    r = runs[runs.week < w]
    db = dropbacks[dropbacks.week < w]
    dr = drives[drives.week < w]
    out = {}
    for t in p.posteam.unique():
        o = p[p.posteam == t]
        gm = o.game_id.nunique()
        neu = o[o.neutral & (o.down <= 2)]
        orun = r[r.posteam == t]
        odb = db[db.posteam == t]
        odr = dr[dr.posteam == t]
        d = p[p.defteam == t]
        drun = r[r.defteam == t]
        ddb = db[db.defteam == t]
        ddr = dr[dr.defteam == t]
        rb_tgts = ddb[ddb.pass_attempt == 1]
        out[t] = dict(
            # own offense
            off_plays_pg=len(o) / gm,
            off_neutral_run_rate=neu.is_run.mean(),
            off_proe=o.pass_oe.mean() / 100 if o.pass_oe.notna().any() else np.nan,
            off_plays_per_drive=odr.n.mean(),
            off_long_drive_rate=odr.long.mean(),
            off_epa=o.epa.mean(),
            off_line_yds=orun.line_yds.mean(),
            off_stuff=orun.stuff.mean(),
            off_light_box_ypc_gain=(orun.loc[orun.light_box, "yards_gained"].mean()
                                    - orun.loc[~orun.light_box, "yards_gained"].mean()),
            # own defense (what it allows)
            def_plays_allowed_pg=len(d) / gm,
            def_three_out_rate=ddr.three_out.mean(),
            def_long_drive_allowed=ddr.long.mean(),
            def_epa_allowed=d.epa.mean(),
            def_rush_epa_allowed=drun.epa.mean(),
            def_line_yds_allowed=drun.line_yds.mean(),
            def_stuff_rate=drun.stuff.mean(),
            def_explosive_run_allowed=drun.explosive.mean(),
            def_blitz_rate=ddb.blitz.mean(),
            def_pressure_rate=ddb.was_pressure.mean(),
            def_box_vs_one_back=drun.loc[drun.one_back, "defenders_in_box"].mean(),
            def_light_box_rate=drun.light_box.mean(),
            def_rb_target_share=(rb_tgts.receiver_player_id.isin(RB_IDS)).mean(),
            def_man_rate=(ddb.defense_man_zone_type == "MAN_COVERAGE").mean(),
            # opp offense tendencies (used when this team is the opponent)
            opp_off_plays_pg=len(o) / gm,
            opp_off_run_rate=o.is_run.mean(),
            opp_off_plays_per_drive=odr.n.mean(),
            opp_off_epa=o.epa.mean(),
        )
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------
# player-level: RBs
# ----------------------------------------------------------------------------
rb = stats[stats.position == "RB"].copy()
RB_IDS = set(rb.player_id)
rb["touches"] = rb.carries.fillna(0) + rb.receptions.fillna(0)
rb["fp"] = rb.fantasy_points_ppr.fillna(0)
rb = rb.sort_values(["player_id", "week"])

# per-player per-week touch share by score state (from pbp)
plays["rb_id"] = np.where(plays.is_run == 1, plays.rusher_player_id, plays.receiver_player_id)
touch_ev = plays[plays.rb_id.isin(RB_IDS)]

# team plays per state per week
team_state = plays.groupby(["posteam", "week"]).agg(
    plays=("play_id", "size"), trail_plays=("trailing", "sum"), neutral_plays=("neutral", "sum"),
    lead_plays=("leading", "sum")).reset_index()
pl_state = touch_ev.groupby(["rb_id", "posteam", "week"]).agg(
    touches=("play_id", "size"), trail_touches=("trailing", "sum"), neutral_touches=("neutral", "sum"),
    lead_touches=("leading", "sum"),
    gl_carries=("yardline_100", lambda s: (s <= 10).sum())).reset_index()
pl_state = pl_state.merge(team_state, on=["posteam", "week"])

# Vegas, per team-week
h = sched[["week", "home_team", "away_team", "spread_line", "total_line", "roof", "gameday"]].rename(
    columns={"home_team": "team", "away_team": "opp"})
h["spread"] = -h.spread_line  # spread_line is home margin; negative spread = favored
a = sched[["week", "home_team", "away_team", "spread_line", "total_line", "roof", "gameday"]].rename(
    columns={"away_team": "team", "home_team": "opp"})
a["spread"] = a.spread_line
vegas = pd.concat([h, a])
vegas["implied_total"] = vegas.total_line / 2 - vegas.spread / 2
vegas["home"] = vegas.team.isin(sched.home_team) & vegas.index.isin(h.index)
vegas = vegas[["week", "team", "opp", "spread", "total_line", "implied_total", "roof"]]

# injuries: status for this week
inj_w = inj[inj.position == "RB"][["week", "team", "gsis_id", "report_status", "practice_status"]].rename(
    columns={"gsis_id": "player_id"})

rows = []
for w in WEEKS:
    if w < 4:
        continue  # need some history
    prof = team_profile(w)
    hist = rb[rb.week < w]
    hist_ev = pl_state[pl_state.week < w]
    cur = vegas[vegas.week == w]
    # players on teams playing this week: use all RBs who've appeared, plus their current team
    last_team = hist.groupby("player_id").team.last()
    for pid, team in last_team.items():
        g = cur[cur.team == team]
        if g.empty:
            continue  # bye
        opp = g.opp.iat[0]
        ph = hist[hist.player_id == pid]
        pe = hist_ev[hist_ev.rb_id == pid]
        l3 = ph.tail(3)
        tm_hist = hist[hist.team == team]
        # team touch share, trailing 3 team games
        team_touch = tm_hist[tm_hist.week.isin(l3.week)].groupby("week").touches.sum()
        share3 = (l3.set_index("week").touches / team_touch).mean() if len(l3) else np.nan
        # script sensitivity: share of team plays touched when trailing vs neutral
        trail_share = pe.trail_touches.sum() / pe.trail_plays.sum() if pe.trail_plays.sum() >= 30 else np.nan
        neutral_share = pe.neutral_touches.sum() / pe.neutral_plays.sum() if pe.neutral_plays.sum() >= 30 else np.nan
        all_share = pe.touches.sum() / pe.plays.sum() if pe.plays.sum() else np.nan
        gl_share = pe.gl_carries.sum() / max(1, pe.touches.sum())
        # is the team's lead back (by prior touches) out this week?
        lead_back = tm_hist[tm_hist.week >= w - 3].groupby("player_id").touches.sum().idxmax() \
            if len(tm_hist[tm_hist.week >= w - 3]) else None
        status = inj_w[(inj_w.week == w) & (inj_w.player_id == pid)]
        lead_status = inj_w[(inj_w.week == w) & (inj_w.player_id == lead_back)]
        actual = rb[(rb.week == w) & (rb.player_id == pid)]
        snap_pct = snaps[(snaps.pfr_player_id.notna()) & (snaps.week == w)]  # placeholder, not joined
        rows.append(dict(
            season=SEASON, week=w, player_id=pid, player=ph.player_display_name.iat[-1], team=team, opp=opp,
            games_prior=len(ph),
            # usage
            fp_l3=l3.fp.mean(), touches_l3=l3.touches.mean(), carries_l3=l3.carries.mean(),
            targets_l3=l3.targets.mean(), touch_share_l3=share3, fp_season=ph.fp.mean(),
            touches_last=l3.touches.iat[-1] if len(l3) else np.nan,
            gl_share=gl_share, all_share=all_share,
            trail_share=trail_share, neutral_share=neutral_share,
            script_sens=(trail_share - neutral_share) if pd.notna(trail_share) and pd.notna(neutral_share) else 0.0,
            weeks_since_last=w - ph.week.iat[-1],
            status=status.report_status.iat[0] if len(status) else "none",
            practice=status.practice_status.iat[0] if len(status) else "none",
            lead_back_out=int(len(lead_status) and lead_status.report_status.iat[0] == "Out" and lead_back != pid),
            is_lead_back=int(lead_back == pid),
            # vegas
            spread=g.spread.iat[0], total_line=g.total_line.iat[0], implied_total=g.implied_total.iat[0],
            dome=int(g.roof.iat[0] in ("dome", "closed")),
            # team & opponent profiles
            **{k: prof.loc[team, k] if team in prof.index else np.nan for k in prof.columns if k.startswith("off_")},
            **{("opp_" + k): prof.loc[opp, k] if opp in prof.index else np.nan for k in prof.columns if k.startswith("def_")},
            **{k: prof.loc[opp, k] if opp in prof.index else np.nan for k in prof.columns if k.startswith("opp_off_")},
            # outcome
            fp=actual.fp.iat[0] if len(actual) else np.nan,
            touches=actual.touches.iat[0] if len(actual) else np.nan,
            carries=actual.carries.iat[0] if len(actual) else np.nan,
            rush_yds=actual.rushing_yards.iat[0] if len(actual) else np.nan,
            rec_yds=actual.receiving_yards.iat[0] if len(actual) else np.nan,
            played=int(len(actual) > 0),
        ))

feat = pd.DataFrame(rows)
# interaction features the "full" model uses
feat["box_x_light_gain"] = feat.opp_def_light_box_rate * feat.off_light_box_ypc_gain
feat["line_edge"] = feat.off_line_yds - feat.opp_def_line_yds_allowed
feat["blitz_x_targets"] = feat.opp_def_blitz_rate * feat.targets_l3
feat["pace_mismatch"] = feat.opp_off_plays_per_drive - feat.off_plays_per_drive
feat["p_trail_proxy"] = 1 / (1 + np.exp(-feat.spread / 6))  # ~prob of trailing from spread
feat["script_risk"] = feat.p_trail_proxy * feat.script_sens
feat.to_parquet(OUT / f"features_rb_{SEASON}.parquet", index=False)
print(feat.shape, "weeks", feat.week.min(), "-", feat.week.max())
print(feat[feat.played == 1].describe().T[["count", "mean"]].head(60).round(3).to_string())
