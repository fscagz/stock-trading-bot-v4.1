# Verdict: does "buy the morning momentum and ride it" work intraday?

**Answer: no — not in any form tested. The apparent edge is 28 lottery tickets out of 2,776 trades.**

This was the last untested version of the original product vision. Every prior study in this repo
used daily bars (open→close, close→close), which cannot see an intraday entry/exit. This one uses
1-minute bars and screens only on information available at the decision minute.

Script: `experiments/run_intraday_momentum_study.py`
Row cache: `backtest_results/intraday_study_rows.parquet`

## Method

- **106,387 candidate entries across 26,916 symbol-days** from the 28,471-file 1-min cache
  (`backtest_results/cache`), spanning 2022 (bear), 2025, 2026.
- **No look-ahead.** Screens use only what is knowable at the entry minute: opening gap, return
  since open, volume pace vs. prior 20-day average, position vs. VWAP, whether price is at the
  high of day so far. Candidate set is *all* cached symbol-days — the days that looked identical
  at 09:35 and then fizzled are included and drag the average, exactly as they would live.
  (Selecting on the intraday high, as the cohort study did, inflates entry statistics enormously
  and is not tradeable — see `2026-08-20_surge_cohort_study.md` §8.)
- **Grid:** 6 screens × 4 entry times (09:31 / 09:35 / 09:45 / 10:00) × 5 exits
  (15 / 30 / 60 / 120 min / EOD) = **120 cells**.
- **Costs:** 25 bps per side (0.50% round trip) as the base case, 50 bps per side as sensitivity.
  Calibrated to this repo's own measured fills (~0.55% round trip on thin low-float names).
- **Liquidity gate:** same-day dollar-volume pace ≥ $5M at the decision minute.

## Headline results across all 120 cells

| Robustness check | Result |
|---|---|
| Cells with a **positive median** trade | **3 / 120** |
| Cells still positive after **dropping the top 5 winners** | 28 / 120 |
| Cells with p < 0.05 | 20 (vs. 6.0 expected by chance at 120 tests) |
| Median across cells of "share of profit from top 5 trades" | −36.4% (i.e. typically the non-top-5 trades lose money in aggregate) |

**117 of 120 configurations have a negative median trade.** The typical trade loses money in
nearly every variant. Positive *means* coexist with negative *medians* — the signature of a
right-tail lottery, not an edge.

The naive "max cell" is a trap: `gap≥20% & at HOD & above VWAP | 10:00 | EOD` shows mean **+8.82%**
— but n=104, it is the maximum of 120 tests, and its per-year record is 2022 +4.7%, 2025 +19.9%,
**2026 −11.8% with a 15% win rate.** That is curve-fitting noise, not a strategy.

## The one cell that survives scrutiny — and why it still fails

`up ≥10% since open & rvol ≥3 | entry 10:00 | exit EOD` — largest sample, survives outlier
removal, p=0.001 overall:

| | Value |
|---|---|
| n | 2,776 |
| Mean net | **+1.47%** |
| Median net | **−0.12%** |
| Win rate | 49.2% |
| **Top 1% of trades' share of total net profit** | **106%** |

That last row is the finding. **Removing the best 28 trades (1%) makes the entire strategy net
negative.** Per-year significance confirms it is not a stable effect:

| Year | n | Mean | Median | Win % | t | p |
|---|---|---|---|---|---|---|
| 2022 | 694 | +0.80% | −0.16% | 49.1% | +1.22 | 0.221 |
| 2025 | 1,569 | +2.05% | −0.19% | 49.3% | +2.88 | 0.004 |
| 2026 | 513 | +0.62% | −0.09% | 49.3% | +0.86 | 0.390 |

Only 2025 is individually significant. A 49.2% win rate with a −0.12% median trade, where all
profit comes from 1% of trades, is not something you can run an account on — the drawdown path
between lottery hits would be brutal, and a single missed tail (a halt, a bad fill, being flat
that day) flips the whole result.

Note also this "best" variant holds to the close — it is not "ride the morning momentum" at all.
The genuinely fast exits (15/30 min) are negative nearly everywhere.

## Why this matters: the same pathology, a third time

This is now the **third independent occurrence of the identical failure mode** in this project:

1. `project-2026-07-01-full-diagnosis` — backtest edge traced to "10 impossible lottery fills."
2. `project-2026-07-01-edge-research` — fill-realism run: "top-10 trades = 282% of net profit";
   edge went −74% under honest fills.
3. **This study** — top 1% of trades = 106% of net profit.

Every time the long side of this cohort has looked profitable, the profit has been concentrated in
a handful of extreme winners that (a) do not repeat reliably across years and (b) are precisely
the fills hardest to actually obtain in thin, halted, fast-moving names. This is a property of the
cohort, not a bug in any one backtest.

## Conclusion

**The long side of explosive movers has now been tested at every timescale available and has not
produced a durable edge at any of them:**

- Multi-day holds (PEAD/episodic pivot) — refuted out-of-sample, `project-2026-07-02-multiregime-verdict`
- Daily open→close on gappers — negative every year 2017–2026, `2026-08-20_surge_cohort_study.md` §8
- **Intraday minute-level entries and exits — this document**

The one edge that has been robust in every regime tested remains **fading** these names
(junk-gapper short, median −16.2% over 20d, t=−13.15, negative every year 2018–2024), which is
blocked on hard-to-borrow availability and carries catastrophic tail risk — the Alpaca ETB
hit-rate measurement from `bot/short_qual_logger.py` is still the gating datapoint.

## Caveats

- Minute cache covers 2022, 2025, 2026 only — no 2020-style crash regime, no 2023/2024.
- Cache contents were assembled by prior backtests, so the symbol-day set is not a clean random
  sample of all market days; it is skewed toward names those backtests screened in.
- Costs are modeled as a flat per-side percentage. Real fills on halted, gapping micro-caps have
  fat-tailed slippage that a flat model understates — i.e. **real results would be worse**, and
  the tail trades that carry the entire result are the most likely to be unobtainable.
- No borrow costs, halt-handling, or partial-fill modeling.
