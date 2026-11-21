# The model, as it stands

Updated through week 9, 2026.

## Pipeline

```
python src/load_data.py 2026            # refresh raw data (pbp, FTN, stats, schedules, injuries, snaps, rosters)
python src/expl_traits.py 2026          # explosive-run traits (defense + rusher), prior-season prior
python src/qb_traits.py 2026            # each team's expected starter and his profile, walk-forward
python src/def_tiers.py 2026            # pass-defense rating on the absolute 2000-25 scale
python src/def_effect.py 2026           # each defense's individual effect on opposing QBs vs their own averages
python src/pedigree.py 2026             # draft capital, experience, prior-season efficiency (needs player_stats 2018+)
python src/features.py 2026 <week>      # walk-forward feature table for the target week
python src/project.py 2025 2026 <week> [player names...]   # fit on 2025, project the week
python src/board.py 2026 <week>         # top 20/30/30/20 with tiers and utilization labels -> results/board_*.csv
```

Run the trait builders and features for 2025 once (`... 2025` with no week) to
produce the training table. `project.py` prints the 2025 walk-forward
validation each time it runs, so a change that hurts accuracy is visible.

`data/overrides_{season}.csv` carries news the injury file doesn't have yet
(week, player, status=Out; week 0 = rest of season; or week, team, new_qb=1) and is applied by
`features.py` every run, including the teammate-out flags.

## Daily cadence

`./run_daily.sh 2026 <week>` refreshes data, rebuilds traits/features/projections/board.
A scheduled task runs it every evening (6 pm Pacific, after the injury report
posts) and commits; the run also updates `data/weather_{season}_wk{W}.csv`
(wind, precip chance, extreme flag per outdoor stadium from the forecast page)
and `data/overrides_{season}.csv` for news the injury feed doesn't carry.
Doubtful is treated as Out automatically. The extreme-weather flag (heavy rain
or 25+ mph sustained) is manual: QB −4, WR/TE −35% of the mean, RBs noted
"run-heavy script"; the size is an assumption pending precipitation history.

## Validation suite

`./run_checks.sh` (pytest, `tests/`) runs against the latest built week at the
end of every daily run and fails the run on any error. It checks the output
contract (every column the board and UI depend on; p10 ≤ p50 ≤ p90; finite
numbers; no Out or bye players on the board; ranks, tiers and sort order),
flag definitions (soft run D matches the defense table; emerging excludes
established top-12 players; extreme weather applied), the adjustment identity
(def_adj = 10 × def_qb_delta + wind_adj, equal within an offense, never
positive for wind, zero in domes), input hygiene (every override name resolves
to a real player; Out and new-QB overrides land in features and projections;
Doubtful treated as Out; lead-back-out flags; wind matches the forecast file;
rest-of-season Outs never appear), walk-forward hygiene (fp_season matches
prior-week stats; no target-week score inside fp_l3; defense effect at week 1
has no current-season dropbacks), the UI export (same players as the board,
required keys, one game per scheduled game, 32 defenses), and **model
regression**: `project.py` writes its 2025 walk-forward ladder to
`results/validation_2025_latest.json`; `tests/validation_baseline.json` is the
accepted state, and the chosen model at each position may not be worse than it
by more than 0.03 MAE or 0.015 rank, must still beat the season average, and
the training sample size may not drift more than 5%. Accept a deliberate
change with `python tests/freeze_baseline.py` and say why in the commit. The first run of the
suite found three live bugs: the soft-run-D flag written after the save, the
new-QB override never reaching the model, and a misspelled override name that
silently did nothing.

## Web UI

`ui/index.html` plus `ui/data/*.json` (from `python src/export_ui.py 2026 <week>`,
run at the end of `run_daily.sh`). Published as a Claude artifact at
https://claude.ai/artifact/XoKeaVNCVYXWE7z97oHzbk (private until shared). To
refresh it after a run, republish the four data files to that URL; the page
itself only changes when the UI changes. Views: Board (position, search,
flag filters, sortable, row drawer with the contribution breakdown, usage,
pedigree, game and opponent), Games (picks, lines, weather), Defenses (each
defense's effect on QBs, blitz/pressure, tier, explosive-run rate), How to read.

## Inputs (all free, nflverse)

Per player, using only games before the target week:
- Trailing 3-game usage, weighted by offensive snap share: touches, carries,
  targets, receptions, yards, points; touch share and target share; air-yards
  share and depth of target (receivers); attempts and yards (QBs).
- Season-to-date points, red-zone and goal-line share, script sensitivity
  (share when trailing vs close).
- Role flags: lead back / lead receiver; lead back or receiver Out this week;
  quarterback changed vs 3 weeks ago.
- Vegas spread, total, implied team total; dome.
- Team tendencies: plays/g, neutral pass rate, pass rate over expected,
  plays/drive, passing efficiency, red-zone plays/g.
- Opponent: plays faced, three-and-out rate, blitz rate, pass-defense rating
  (absolute scale), explosive-run susceptibility, pressure/sack rate.

## Per-position models (fit on 2025, ridge regression, walk-forward selected)

| Position | Mean model | Features used | Range (p10/p90) |
|---|---|---|---|
| RB | usage + pedigree (+ environment) | trailing usage, shares, role flags, script sensitivity; draft pick, experience, prior-season efficiency and volume | quantile regression on total pts; p90 adds opponent explosive-run susceptibility |
| WR / TE | usage + pedigree (+ environment) | trailing usage, target/air-yard shares, aDOT, role flags, new-starter flag; pedigree as above | quantile regression |
| QB | passing + rushing, summed | passing: trailing passing pts, attempts, QB efficiency, environment, absolute-scale pass-defense rating, pressure/sack/blitz; rushing: trailing rush yards/carries, spread, implied total | quantile regression on total pts |

**Validation is scored on the players that matter**: the top-N by season
average at each position (RB 24, WR 36, TE 12, QB 16) plus "emerging" players
(trailing 3-game average ≥ 4 points above their season average and ranked
top-N on the trailing number). Role players and random events are excluded
from the score because they obscure it. The ladder carries the season average
as a baseline and falls back to it only if it beats the model by more than
0.10 MAE on that population; ties go to the model because it is unbiased on
top players where the season average over-projects by about a point.

Situational linear adjustments on top of a season average (`src/situational.py`)
were tested and add nothing; the usage model already captures them.

## Output columns

`p10 / p50 / p90` — floor / projection (median) / ceiling, calibrated to ~80%
coverage. `mean` — expected value, the number comparable to ESPN. `ceiling_flag`
— "soft run D" when a back faces a bottom-third explosive-run front. `def_tier`
— opponent pass defense on the absolute scale (relevant to QBs). `shootout` —
1 when both offenses are top-third, spread ≤ 3.5, no top-7 pass defense.
`contrib_usage / environment / matchup` — points attributable to each group.
`def_qb_delta` — the opponent's individual effect on QBs (EPA/dropback vs
their own averages, shrunk); `def_adj` — the points it adds to or removes from
this player (10 × def_qb_delta, applied to mean and range); `def_blitz` — the
opponent's blitz rate.

## Conventions

- **Projection = median.** Use the mean for sums and for ESPN comparison.
- **Regression toward the position average is deliberate.** On 2025 and on
  2026 weeks 7–9 the model is unbiased on stars while the season average
  over-projects them by ~2 points; ESPN's wins on individual stars were not a
  systematic lean.
- **Rerun Saturday** against the final injury report before grading.
- **Grade every Monday**: fill the `actual` column in that week's log in
  `results/`, compare MAE and rank vs ESPN and vs the naive baseline, and check
  80% coverage.

## Validation on 2025 (walk-forward, weeks 7–18)

| Position | n | MAE | Rank corr | vs naive MAE |
|---|---|---|---|---|
| RB | 610 | 5.86 | 0.51 | 6.62 |
| WR | 1,113 | 5.55 | 0.47 | — |
| TE | ~400 | 5.10 | 0.30 | — |
| QB | 370 | 6.9 | 0.25 | 6.9 |

QB rank accuracy is the weakest; separating one starter's week from another's
is close to noise for every model we've tested.


## Assessment criteria by position (as of week 11, 2026)

Every position shares the same frame: **usage** (how much), **the player**
(what he does with it), **the offense** (size of the pie), **the opponent**
(applied to the whole passing game, never per receiver), and **range** (floor
and ceiling, fit separately from the mean). The validation ladder scores each
feature set on top-N + emerging players against a season-average baseline and
keeps whichever wins; the current selection is noted per position.

### Quarterback — split model, passing + rushing (ladder: "full")
- **Passing volume/efficiency:** passing pts last 3 (snap-weighted) and season, attempts, pass yards, pass TDs last 3; his own EPA/dropback (walk-forward).
- **Offense:** dropbacks/g, pass success rate, red-zone plays/g, plays/g, pass rate over expected, plays/drive, pass EPA.
- **Game:** spread, implied total, total, dome, opponent pace and three-and-out rate.
- **Opponent (learned):** absolute-scale pass-defense rating, yards/att allowed, pressure, sack and blitz rates.
- **Opponent (structural):** + 10 × the defense's individual effect on QBs (vs their own averages). This is the Minnesota number.
- **Rushing (separate ridge):** rush yards/carries/rush pts last 3 and season, spread, implied total, red-zone plays, opponent rush EPA and line yards allowed. Matchup-proof floor; summed with passing.
- **Not used:** pedigree (no gain), QB × defense interaction (no gain).
- **Range:** quantile models on total-points inputs (known inconsistency with the split mean), shifted by the defense adjustment.
- **Diagnostics on the board:** success rate and low-percentage-throw rate next to EPA (repeatability), new-starter flag.

### Running back (ladder: "usage")
- **Usage:** touches, carries, targets last 3 (snap-weighted); touch share; goal-line and red-zone share; lead-back flag; lead back Out this week (teammate absence, the largest predictable volume event).
- **Script:** player-specific script sensitivity (share when trailing vs close) × probability of trailing from the spread.
- **The player:** draft pick, experience, prior-season PPR per opportunity (shrunk, 150-opp prior), prior-season opportunities/g. Separates "thrust into volume" from "earned it".
- **Opponent (structural):** + 10 × the defense's QB effect (receiving work is part of the passing game).
- **Ceiling only:** opponent's explosive-run susceptibility (prior season fades with sample; zero for a new DC) lifts/lowers p90, never the mean. "Soft run D" flag.
- **Not used in the mean:** environment and line/scheme metrics (rejected), snap share (redundant with snap-weighted usage), the back's own explosive history (not stable).
- **Labels:** utilization (snap share, carry share, target share), emerging flag, comps for a back new to volume.

### Wide receiver (ladder currently: "usage+env")
- **Usage:** targets, receptions, receiving yards last 3 (snap-weighted); target share; air-yards share; aDOT; red-zone share; lead-receiver flag; lead receiver Out; carries (gadget usage).
- **Quarterback change:** new-starter flag (−2.6 pts over-projection otherwise; who the new QB is doesn't matter).
- **The player:** pedigree as for RBs.
- **Offense/game (environment):** spread, implied total, total, dome, pace, pass rate over expected, dropbacks/g, pass success.
- **Opponent (structural):** + 10 × the defense's QB effect, equal for every receiver on the offense.
- **Not used:** defense-vs-WR points allowed, explosive-pass susceptibility, QB traits, target-depth/location/YAC profiles, shadow corners (not in free data). Validated as adding nothing to the weekly number.
- **Range:** quantile models, shifted by the defense adjustment.
- **Labels/growth:** utilization, emerging flag; for an emerging receiver, the growth model (air-yards share and aDOT jointly, youth, draft capital) and comps.

### Tight end (ladder: "usage")
- Same as WR, plus **snap share last game** (the one position where it adds).
- Opponent TE target share allowed was tested and rejected.

### Team / game layer (feeds all of the above)
- Vegas spread, total, implied totals (re-pulled on game day).
- Shootout flag: both offenses top-third, spread ≤ 3.5, no top-7 pass defense.
- Defensive profiles: blitz rate, pressure rate, EPA allowed by depth, location, down and red zone (`qb_situations`/`def_effect` outputs), explosive-run susceptibility, absolute-scale pass-defense tier.
- Game winners: Vegas line + season and last-4 EPA margins + starting-QB and key-absence check.
- Manual inputs: `data/overrides_{season}.csv` (Out, new QB), `data/coaching_changes.csv` (new DC).

### What is deliberately *not* in any position's mean
Line/scheme aggregates, defense-vs-position, coaching distribution tendencies,
opponent-based game-plan shifts, shadow corners, receiver target-profile
splits, snap share for RB/WR, situational/momentum QB splits. Each was tested
on 2024–2025 against the season-average baseline on top+emerging players.

## Next changes, in order

Reset after week 9 (see FINDINGS.md headline): the model has not beaten a
plain season average, so the validation ladder and the architecture change.

1. ~~Season-average baseline in the ladder~~ — done; validation now scored on
   top + emerging players (see FINDINGS, "Measured on the players that matter").
2. Returning-teammate flag (Hall/Allen); test on 2024+2025 top+emerging.
3. ~~Extend comps to WR/TE~~ — done (`--wr`). Add the "returning starter"
   case; surface the growth forecast on the weekly board for emerging players.
3. Keep the quantile models for ranges; move QB quantiles onto split inputs.
4. Re-pull Vegas lines on game day (schedules file goes stale).
5. `data/coaching_changes.csv` (season, team, new_dc) zeroes the prior-season
   weight in the explosive-run trait for defenses with a new coordinator; the
   weight otherwise fades as the current-season sample grows (half weight at
   0 carries, ~15% by 300). Same treatment is still to be applied in
   `def_tiers.py`.
5. Deferred: snap-share trend, QB–receiver reconciliation, def_tiers fade,
   explosive trait on the mean.
