# The model, as it stands

Updated through week 9, 2026.

## Pipeline

```
python src/load_data.py 2026            # refresh raw data (pbp, FTN, stats, schedules, injuries, snaps, rosters)
python src/expl_traits.py 2026          # explosive-run traits (defense + rusher), prior-season prior
python src/qb_traits.py 2026            # each team's expected starter and his profile, walk-forward
python src/def_tiers.py 2026            # pass-defense rating on the absolute 2000-25 scale
python src/pedigree.py 2026             # draft capital, experience, prior-season efficiency (needs player_stats 2018+)
python src/features.py 2026 <week>      # walk-forward feature table for the target week
python src/project.py 2025 2026 <week> [player names...]   # fit on 2025, project the week
python src/board.py 2026 <week>         # top 20/30/30/20 with tiers and utilization labels -> results/board_*.csv
```

Run the trait builders and features for 2025 once (`... 2025` with no week) to
produce the training table. `project.py` prints the 2025 walk-forward
validation each time it runs, so a change that hurts accuracy is visible.

`data/overrides_{season}.csv` carries news the injury file doesn't have yet
(week, player, status=Out; or week, team, new_qb=1) and is applied by
`features.py` every run, including the teammate-out flags.

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
