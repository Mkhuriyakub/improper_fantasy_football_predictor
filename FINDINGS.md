# Findings

What the data has shown so far, in order of confidence. Every claim below was
tested on free nflverse data, mostly the 2025 season walk-forward (each week
projected from prior weeks only) with 2023–24 and 2000–25 used where noted.
Updated through week 9 of the 2026 season.

## The headline, after week 9: the model has not beaten a season average

Every "validated" gain below was measured against a 3-game trailing average.
That was the wrong baseline. Against each player's plain season-to-date
average on the same sample:

| | Model MAE | Season-avg MAE | Model rank | Season-avg rank |
|---|---|---|---|---|
| 2025 walk-forward (2,204 player-weeks) | 5.85 | **5.79** | 0.38 | 0.38 |
| 2026 live, weeks 7–9 (503) | 5.65 | 5.64 | 0.41 | **0.42** |

The ridge over twelve usage features is a season average with ~0.1 points of
noise either way. The 3-game average is clearly worse than both (6.6 / 6.1).

Where the model does beat the season average (2025): stars with 15+ pts/g
(7.16 vs 7.47; the season average over-projects them by 2.0, the model is
unbiased, so the regression is right), hot and cold streaks (±5 vs season:
5.8 vs 6.0 and 6.1 vs 6.5), small samples of ≤5 games (5.41 vs 5.47), and
teammate-out weeks (6.76 vs 7.05, though the model over-shoots the boost by
2.2 pts). Where it is worse: low-volume players (over-projected by 0.6),
players with 10+ games (the average is already right), and big underdogs
(over-projected by 1.4; adding the spread removes the bias but not the MAE).

The "we regress stars too hard" worry from the ESPN comparisons does not hold
in aggregate: 2026 top-quartile RBs averaged 16.0 actual vs 17.9 projected and
19.9 season average; WRs 18.1 vs 16.9 vs 19.5; QBs 18.1 vs 17.5 vs 19.3. ESPN
won on individual players, not on a systematic lean.

What this implies: the right architecture is a season average as the spine
with a handful of situational adjustments (star regression, streak regression,
teammate-out at a smaller size than the model learned, spread for underdogs,
returning-teammate), not a ridge regression over usage features. The ranges
(80% coverage: 80%, 86%) and the game-level work are where the model has
shown value beyond the trivial baseline.

## Validated: these carry the projections

**Volume and role are the predictable core.** A player's recent touch/target
share, lead-back or lead-receiver status, and depth-chart vacancies are the
inputs that improve projections beyond a trailing average. Nothing about the
opponent improves volume prediction. (2025: 565 RB-weeks, 834 WR-weeks.)

**Teammate absences are the largest predictable volume event.** A back whose
backfield-mate (6+ carries/g) is Out gains ~4.4 touches vs a ~1-touch decline
otherwise. Same direction, smaller, for receivers.

**Extreme recent volume regresses.** Backs in the top 10% of trailing touches
lose ~4.5 touches the next week; receivers at the top lose ~2 targets. This is
sample noise around a true role, not coaching.

**Script sensitivity is player-specific and the strongest single modeling
variable.** Some backs lose half their share when their team trails (Bijan
Robinson: 45% → 25%); others don't. Computed per player from play-by-play.

**Offensive quality sets the size of the pie.** Lead backs on bottom-third
passing offenses score ~2 PPR/g less than those on top-third offenses on equal
carries, through fewer red-zone trips (CLE: 7 RZ plays/g, league 10) and worse
script. Holds even in close games.

**Pass-defense quality is a quarterback matchup, not a receiver matchup.**
Quarterbacks beat their own norm 66% of the time against the worst quarter of
pass defenses and 39% against the best. The effect is ~3 points of QB fantasy
output between extremes, mostly already in the Vegas line. Which receiver
collects it is not predictable (2025 DAL: top-2 opposing WRs +0.8 pts vs their
average; 7 of ~17 WR1s underperformed).

**The elite-defense effect is nonlinear and concentrated at the top.** On
passing points relative to each QB's own average (2024–25, 806 QB-games):
elite defenses −1.0, good −0.1, mid +0.3, bad +0.4. The best quarterbacks lose
the most to elite defenses (−3.1). Bad defenses don't lift anyone much.

**"Elite" must be defined on an absolute, era-adjusted scale.** Over 2000–25
(832 team-seasons) the top 5% of pass defenses are at ≤ −0.160 EPA/dropback
below league average; top 2% ≤ −0.205. By this standard 2024 and 2025 had zero
elite pass defenses; 2026 has one (Minnesota) entering week 9.

**Quarterback rushing is a matchup-proof floor.** Runners and pocket passers
throw for the same points (12.5 vs 12.8) and respond to matchups the same way;
runners add ~2.2 pts/g of rushing that is more predictable than passing
(r=0.34 vs 0.23). The QB model projects the two separately.

**Defenses' susceptibility to explosive runs is a stable trait; backs'
explosiveness mostly isn't.** Within-season stability: defense 0.46–0.52,
running back ~0 (0.23 year to year). The defense trait improves RB ceiling
(p90) projections and top-decile smash identification (34% → 38%) without
moving the mean. For elite backs vs the softest third of fronts: 14.6% vs
11.4% explosive rate, +1 YPC, ~+5 pts of p90.

**Receiver explosiveness is a stable trait of the offense/quarterback, not the
opponent.** Offense explosive-pass rate: 0.61 odd/even stability; defense
explosive-pass rate allowed: −0.07 half-to-half. Already captured by the
receiver's own air-yards profile.

**A quarterback change hurts pass catchers short-term regardless of who
replaces whom.** +2.6 pts over-projection in the 38 new-starter weeks of 2025;
a flag reduces it to +0.5. The new QB's stats don't predict the size.

**Shootouts come from two good offenses, not two bad defenses.** Games with
both offenses top-third: 28% reach 63+ (base 11%). Add close spread and no
elite defense: 33% reach 63+, 58% reach 56+, 83% go over the Vegas total,
+7.4 vs the line (12 games, 2023–25). Both-bad-defense games: no effect.

**Partial games must be weighted by snap share.** Weighting each trailing game
by offense snap share improved rank accuracy at every position on 2025.

**The projections are calibrated.** Across 171 players in week 7 2026, 80%
landed inside the 80% range (85/80/73/81% by position).

## Tested and rejected: these don't improve weekly projections

- Team-level line metrics (adjusted line yards, stuff rate, sack rate,
  opponent-adjusted) for RB projections: no gain, slightly negative.
- Defensive box tendency × offense light-box sensitivity: no gain.
- "Defense vs position" fantasy points allowed: TE and RB versions don't
  repeat within a season (0.17, −0.02); the WR version is weak (0.44) and
  worth ~+0.8 pts in real-time use with no increase in smash rate.
- Explosive-pass susceptibility of defenses for receivers: no gain.
- Opponent's offensive strength as a volume input (the "forced aggression"
  idea): r=+0.05, no gain. Teams throw more against good offenses only via
  script, which the spread already carries.
- Quarterback traits (level, efficiency, explosiveness) as receiver inputs: no
  gain. The receiver's own trailing stats already carry them.
- QB quality × defense interaction in the QB model: not significant; made the
  model slightly worse.
- Coaching tendencies as a predictor of target/carry distribution: the
  distribution stays with the roster, not the coach. Coaches who moved teams
  showed negative carryover on RB1/WR1 shares. Pass rate and TE usage are the
  two things that travel with a coach.
- Coaches adjusting run/pass or target distribution to the opponent: ~1 play
  per game. Scheme is identity; game-planning happens inside the play calls.
- Explosive runs by the back's own history: a back's prior explosive rate
  correlates +0.01 with his next-week rate.
- Early trailing (7+ in the first half) from pregame tendencies: spread alone
  AUC 0.63; adding first-half tendencies made it worse.

## Live results, 2026

| Week | Set | Our MAE | ESPN MAE | Notes |
|---|---|---|---|---|
| 7 | 7 tracked players | 5.3 (mean) | 6.1 | Closest on 5/7; regression won (6/7 under all projections) |
| 7 | 171 all positions | beat naive & season avg at every position | — | 80% range coverage 80% |
| 8 TNF | 9 GB/CAR players | 5.5 | 5.6 | 15/16 in range; ESPN "Shaky" WRs scored 55.5 combined |
| 8 | 15 QBs | 7.6 | **7.0** | ESPN closer on 8/15; our regression on top QBs too heavy |
| 8 MNF | CHI/SEA | all in range | — | Narrative wrong (possession), numbers fine |
| 9 | 83 board players (BUF/MIN pending) | 6.2 (mean), 6.0 (median) | — | Season average 5.9 beat the model; naive 7.1. Coverage 86%. TE rank 0.75, QB 0.28 |
| 9 | 7 roster players | 4.6 | **4.0** | ESPN higher on Bowers/Coleman/Collins and right each time |
| 9 | 13 game winners | 13/13 | Vegas 12/13 | EPA-margin + starter check; only GB@NE differed from the closing line |

Open question after 4 weeks: the model regresses high-usage stars harder than
ESPN and has now lost to ESPN three weeks running on tracked players, and in
week 9 lost to the plain season average across the board. 2025 said the
regression was right; 2026 says otherwise. The "soft run D" ceiling flag went
1-for-7 on p90 hits in week 9.

## Known gaps

- Three-game usage window lags role changes by 2–3 weeks (Coleman, Cooper).
  A snap-share-trend feature and depth-chart input would help; not built yet.
- Projections are bottom-up per player; receivers aren't reconciled to their
  quarterback's passing projection. A reconciliation step is designed, not
  built.
- The QB median (quantile model) and mean (split model) use different inputs
  and can disagree (Allen week 9: 16.6 vs 19.9). The quantile side should
  move to the split inputs.
- Shadow corners aren't in any free data; manual input from weekly previews.
- Prior-season weight in the defense trait is too heavy for defenses that
  changed (Tampa Bay 2026 graded "above avg" despite historic raw numbers).
- Injury statuses are read from the Wednesday report; a Saturday re-run is
  needed every week. The model flags a teammate who is Out but not one who is
  *returning* (Breece Hall, week 9: Allen projected as lead back, got 8 carries).
- Vegas lines come from the schedules file and go stale; the week 9 PHI line
  moved from −3.5 to +3.5 after QB news. Re-pull lines on game day.
- Custom-scoring conversion (`src/yahoo_convert.py`) scales ceilings by the PPR
  ratio, which understates TD-driven ceilings in TD-heavy formats.
