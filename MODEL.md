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
| RB | usage only | trailing usage, shares, role flags, script sensitivity | quantile regression on total pts; p90 adds opponent explosive-run susceptibility |
| WR / TE | usage only | trailing usage, target/air-yard shares, aDOT, role flags, new-starter flag | quantile regression |
| QB | passing + rushing, summed | passing: trailing passing pts, attempts, QB efficiency, environment, absolute-scale pass-defense rating, pressure/sack/blitz; rushing: trailing rush yards/carries, spread, implied total | quantile regression on total pts |

Environment and matchup features were tried for RB/WR/TE and rejected by
validation; the quarterback is the only position where matchup earned a place.

## Output columns

`p10 / p50 / p90` — floor / projection (median) / ceiling, calibrated to ~80%
coverage. `mean` — expected value, the number comparable to ESPN. `ceiling_flag`
— "soft run D" when a back faces a bottom-third explosive-run front. `def_tier`
— opponent pass defense on the absolute scale (relevant to QBs). `shootout` —
1 when both offenses are top-third, spread ≤ 3.5, no top-7 pass defense.
`contrib_usage / environment / matchup` — points attributable to each group.

## Conventions

- **Projection = median.** Use the mean for sums and for ESPN comparison.
- **Regression toward the position average is deliberate** and currently
  heavier than ESPN's. Under review (see FINDINGS.md, live results).
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

1. Snap-share trend and depth-chart rank as usage inputs (role changes a week
   earlier).
2. Move the QB quantile models onto the split passing/rushing inputs.
3. Reconcile each team's pass-catcher projections to its QB's passing
   projection; validate on 2025 before adopting.
4. Fade the prior-season weight in `def_tiers.py` as the current season's
   sample grows.
5. Test letting the explosive trait lift the *mean* for top-decile backs.
