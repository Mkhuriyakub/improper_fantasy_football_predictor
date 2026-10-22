import pandas as pd, numpy as np, json, sys
D = sys.argv[1]
pbp = pd.read_parquet(f'{D}/pbp.parquet')
ftn = pd.read_parquet(f'{D}/ftn.parquet')
ngs_r = pd.read_parquet(f'{D}/ngs_rushing.parquet'); ngs_r['team_abbr']=ngs_r.team_abbr.replace({'LAR':'LA'})
ngs_p = pd.read_parquet(f'{D}/ngs_passing.parquet'); ngs_p['team_abbr']=ngs_p.team_abbr.replace({'LAR':'LA'})

ftn = ftn.rename(columns={'nflverse_game_id': 'game_id', 'nflverse_play_id': 'play_id'})
ftn['play_id'] = ftn['play_id'].astype(float)
pbp['play_id'] = pbp['play_id'].astype(float)
df = pbp.merge(ftn[['game_id', 'play_id', 'n_pass_rushers', 'n_defense_box', 'is_qb_fault_sack',
                    'is_screen_pass', 'is_qb_sneak', 'is_throw_away']], on=['game_id', 'play_id'], how='left')
df = df[(df.posteam.notna()) & (df.two_point_attempt != 1) & (df.qb_kneel != 1) & (df.qb_spike != 1)]
games = df.groupby('posteam').game_id.nunique()

# ---------- pass protection ----------
db = df[df.qb_dropback == 1].copy()
db['hit_or_sack'] = ((db.qb_hit == 1) | (db.sack == 1)).astype(int)
db['line_sack'] = ((db.sack == 1) & (db.is_qb_fault_sack != True)).astype(int)
db['std_rush'] = db.n_pass_rushers <= 4
db['non_screen'] = db.is_screen_pass != True

# opponent adjustment: each defense's sack rate vs everyone *else* (leave-one-out by offense)
def_tot = db.groupby('defteam').agg(s=('sack', 'sum'), n=('sack', 'size'))
pair = db.groupby(['defteam', 'posteam']).agg(s=('sack', 'sum'), n=('sack', 'size')).reset_index()
pair = pair.merge(def_tot, left_on='defteam', right_index=True, suffixes=('', '_tot'))
pair['exp_rate'] = (pair.s_tot - pair.s) / (pair.n_tot - pair.n)
db = db.merge(pair[['defteam', 'posteam', 'exp_rate']], on=['defteam', 'posteam'])

pp = db.groupby('posteam').apply(lambda g: pd.Series({
    'dropbacks': len(g),
    'sacks': int(g.sack.sum()),
    'sack_rate': g.sack.mean(),
    'line_sack_rate': g.line_sack.mean(),
    'qb_fault_sacks': int((g.is_qb_fault_sack == True).sum()),
    'hit_rate': g.hit_or_sack.mean(),
    'sack_rate_4man': g.loc[g.std_rush & g.non_screen, 'sack'].mean(),
    'hit_rate_4man': g.loc[g.std_rush & g.non_screen, 'hit_or_sack'].mean(),
    'blitz_faced': (g.n_pass_rushers >= 5).mean(),
    'sacks_over_exp_per100': 100 * (g.sack.sum() - g.exp_rate.sum()) / len(g),
    'epa_per_db': g.epa.mean(),
}), include_groups=False)

ttt = ngs_p[ngs_p.week > 0].copy()
if ttt.empty:
    ttt = ngs_p.copy()
ttt['w'] = ttt.attempts * ttt.avg_time_to_throw
t = ttt.groupby('team_abbr').agg(w=('w', 'sum'), a=('attempts', 'sum'))
pp['time_to_throw'] = t.w / t.a

# ---------- run blocking ----------
passers = set(df.loc[df.pass_attempt == 1, 'passer_player_id'].dropna())
ru = df[(df.play_type == 'run') & (df.qb_scramble != 1) & (df.is_qb_sneak != True)
        & (~df.rusher_player_id.isin(passers))].copy()

def aly(y):  # Football Outsiders-style line-yards credit
    if y < 0: return 1.2 * y
    if y <= 4: return y
    if y <= 10: return 4 + 0.5 * (y - 4)
    return 7
ru['line_yds'] = ru.yards_gained.apply(aly)
ru['stuff'] = (ru.yards_gained <= 0).astype(int)
ru['heavy_box'] = ru.n_defense_box >= 8

ldef = ru.groupby('defteam').agg(s=('line_yds', 'sum'), n=('line_yds', 'size'))
rp = ru.groupby(['defteam', 'posteam']).agg(s=('line_yds', 'sum'), n=('line_yds', 'size')).reset_index()
rp = rp.merge(ldef, left_on='defteam', right_index=True, suffixes=('', '_tot'))
rp['exp_ly'] = (rp.s_tot - rp.s) / (rp.n_tot - rp.n)
ru = ru.merge(rp[['defteam', 'posteam', 'exp_ly']], on=['defteam', 'posteam'])

rb = ru.groupby('posteam').apply(lambda g: pd.Series({
    'rb_runs': len(g),
    'ypc': g.yards_gained.mean(),
    'line_yards': g.line_yds.mean(),
    'adj_line_yards': g.line_yds.mean() - g.exp_ly.mean() + ru.line_yds.mean(),
    'stuff_rate': g.stuff.mean(),
    'success_rate': g.success.mean(),
    'epa_per_run': g.epa.mean(),
    'heavy_box_pct': g.heavy_box.mean(),
    'explosive_rate': (g.yards_gained >= 10).mean(),
}), include_groups=False)

nr = ngs_r[ngs_r.week > 0].copy()
if nr.empty:
    nr = ngs_r.copy()
nr = nr.groupby('team_abbr').agg(rye=('rush_yards_over_expected', 'sum'), att=('rush_attempts', 'sum'),
                                 exp=('expected_rush_yards', 'sum'))
rb['exp_rush_ypc'] = nr.exp / nr.att      # NGS expected yards (blocking + box context at handoff)
rb['ryoe_per_att'] = nr.rye / nr.att      # what the runner added beyond that
rb['ngs_rush_cov'] = nr.att / rb.rb_runs

out = pp.join(rb)
out['games'] = games

# ---------- composites (z-scores; signs so higher = better line) ----------
z = lambda s: (s - s.mean()) / s.std()
# weights favour the metrics that held up in the split-half stability check
out['pass_pro_score'] = 0.7 * z(-out.sacks_over_exp_per100) + 0.3 * z(-out.line_sack_rate)
out['run_block_score'] = 0.75 * z(out.adj_line_yards) + 0.25 * z(-out.stuff_rate)
out['overall_score'] = 0.6 * out.pass_pro_score + 0.4 * out.run_block_score
for c in ['pass_pro_score', 'run_block_score', 'overall_score']:
    out[c.replace('_score', '_rank')] = out[c].rank(ascending=False).astype(int)
out = out.sort_values('overall_rank')
out.round(4).to_csv(f'{D}/oline_2026.csv')
lg = {'sack_rate': db.sack.mean(), 'hit_rate_4man': db.loc[db.std_rush & db.non_screen, 'hit_or_sack'].mean(),
      'adj_line_yards': ru.line_yds.mean(), 'stuff_rate': ru.stuff.mean(), 'weeks': int(df.week.max()),
      'dropbacks': len(db), 'runs': len(ru), 'ftn_match': float(db.n_pass_rushers.notna().mean())}
json.dump(lg, open(f'{D}/league.json', 'w'), indent=1)
print(json.dumps(lg, indent=1))
pd.set_option('display.width', 250)
print(out[['games', 'overall_rank', 'pass_pro_rank', 'run_block_rank', 'sack_rate', 'line_sack_rate', 'hit_rate_4man',
           'sacks_over_exp_per100', 'time_to_throw', 'adj_line_yards', 'stuff_rate', 'exp_rush_ypc', 'ryoe_per_att']].round(3).to_string())
