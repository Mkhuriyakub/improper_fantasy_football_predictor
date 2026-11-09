# The model, as it stands

Updated through week 9, 2026.

## Pipeline

```
python src/load_data.py 2026            # refresh raw data (pbp, FTN, stats, schedules, injuries, snaps, rosters)
python src/expl_traits.py 2026          # explosive-run traits (defense + rusher), prior-season prior
python src/qb_traits.py 2026            # each team's expected starter and his profile, walk-forward
python src/def_tiers.py 2026            # pass-defense rating on the absolute 2000-25 scale
python src/features.py 2026 <week>      # walk-forward feature table for the target week
python src/project.py 2025 2026 <week> [player names...]   # fit on 2025, project the week
```

Run the trait builders and features for 2025 once (`... 2025` with no week) to
produce the training table. `project.py` prints the 2025 walk-forward
validation each time it runs, so a change that hurts accuracy is visible.

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
| RB | **season average** (usage ridge did not beat it on 2025) | — | quantile regression on usage features; p90 adds opponent explosive-run susceptibility |
| WR / TE | **season average** (same) | — | quantile regression on usage features |
| QB | passing + rushing, summed | passing: trailing passing pts, attempts, QB efficiency, environment, absolute-scale pass-defense rating, pressure/sack/blitz; rushing: trailing rush yards/carries, spread, implied total | quantile regression on total pts |

`project.py` now carries the season average in its validation ladder and
falls back to it as the mean whenever the best feature set does not beat it
by 0.05 MAE on the training season. As of week 10 that is every position but
QB, where the split passing/rushing model beats the season average (6.88 vs
7.16). Ranges still come from the quantile models, which are calibrated.

Situational adjustments on top of the season average (streak, star regression,
spread, teammate-out, new starter; `src/situational.py`) were tested on 2025
and 2026 weeks 7–9: none improved MAE or rank beyond noise (±0.1 / ±0.03).

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

1. ~~Add `fp_season` as a baseline in the ladder~~ — done; RB/WR/TE means are
   now the season average.
2. ~~Situational adjustments~~ — tested, no gain (see FINDINGS). Keep the
   flags (teammate out, returning teammate, new starter) as *notes* on the
   board, not point adjustments.
3. Keep the quantile models for ranges; move QB quantiles onto split inputs.
4. Re-pull Vegas lines on game day (schedules file goes stale).
5. Deferred: snap-share trend, QB–receiver reconciliation, def_tiers fade,
   explosive trait on the mean.
