# Cross-sectional factor study: liquid large/mid + micro-cap tracks

Both tracks the user asked to pursue in parallel. Script: `experiments/run_factor_study.py`.

**Verdict: no price-based factor configuration beat buy-and-hold SPY, in either universe.**
The infrastructure now works, which is the real deliverable — but the price-only factor subset
available without a fundamentals key does not produce an edge.

---

## 1. Pipeline revival (done)

The dormant systematic stack was not broken — it had a `sys.path` convention conflict. Two
subsystems coexist in this repo:

- live micro-cap bot (`bot/main.py`, `bot/intraday/*`) → `bot.`-prefixed imports, needs repo root
- systematic pipeline (`bot/backtest/engine.py`, `bot/data/*`, `bot/risk/*`) → bare imports
  (`from data.universe import ...`), needs `bot/` on the path

**Fixes applied:**
1. `conftest.py` at repo root adds both paths, so pytest collects either subsystem.
2. `CACHE_DIR` added to `bot/config.py`. `data/store.py`, `data/simfin_loader.py`, and
   `data/fundamental_store.py` were ported from an earlier systematic project and import
   `from config import CACHE_DIR`, which never existed in this repo — they raised ImportError on
   load. (Checked the archived worktree in `.claude/worktrees/` — its config lacks it too, so this
   was never present, not a regression.)

With those, all of `backtest.engine`, `backtest.walk_forward`, `backtest.integrity`,
`backtest.costs`, `backtest.metrics`, `data.point_in_time`, `data.universe`, `risk.risk_model`,
`risk.position_sizing`, `risk.constraints` import cleanly.

`backtest/metrics.py` already contains real factor-research tooling (`compute_ic`,
`compute_ic_series`, `ic_tstat`, `ic_decay`) — reused here rather than reimplemented.

## 2. Method

- **Data:** `screener_cache/2017-06-01_2025-01-31.pkl` only — the split-adjusted cache
  (5,553 symbols after warrant exclusion, 1,929 trading days). The 2024-2026 pkl is deliberately
  excluded: it is not split-adjusted and would inject fake reverse-split returns into every
  momentum/reversal factor.
- **83 monthly rebalances**, 2018-03 → 2025-01. Factors computed from data through the rebalance
  date; return measured to the next rebalance.
- **Factors (price-based only** — no `SIMFIN_API_KEY`/`EODHD_API_KEY` configured, so value and
  quality are unavailable): 12-1 momentum, 1-month reversal, 60-day low-volatility, small-size
  (inverse log dollar volume).
- **Portfolios:** long-only top decile, equal weight, 20 bps per unit turnover.
- **Universes:** `liquid_large_mid` (20d avg dollar volume ≥ $20M, price ≥ $5; median 1,429 names)
  and `microcap` ($100k ≤ DV < $5M, price ≥ $1; median 2,024 names). Dollar volume is the size
  proxy since market cap needs fundamentals.

## 3. Results

**Benchmark SPY, same span: CAGR 13.05%, Sharpe 0.74, MaxDD −34.2%.**

### Liquid large/mid

| Factor | IC mean | IC t | CAGR % | Sharpe | MaxDD % | Decile spread t |
|---|---|---|---|---|---|---|
| momentum_12_1 | 0.021 | 1.10 | 11.03 | 0.50 | −40.2 | 2.11 |
| reversal_1m | 0.009 | 0.62 | −1.81 | 0.11 | −53.0 | 1.14 |
| low_volatility | 0.043 | 1.76 | 3.28 | 0.32 | −21.6 | 0.38 |
| small_size | −0.020 | **−2.35** | −1.48 | 0.05 | −39.9 | −1.78 |

### Micro-cap

| Factor | IC mean | IC t | CAGR % | Sharpe | MaxDD % | Decile spread t |
|---|---|---|---|---|---|---|
| momentum_12_1 | 0.065 | **4.23** | 12.60 | 0.54 | −38.1 | 1.09 |
| reversal_1m | −0.002 | −0.17 | −0.42 | 0.16 | −67.6 | 0.26 |
| low_volatility | 0.105 | **4.77** | −0.72 | −0.06 | −20.4 | −0.18 |
| small_size | −0.012 | −1.55 | 5.23 | 0.34 | −37.5 | 1.43 |

**Not one cell beats SPY on both return and Sharpe.** The best (micro-cap momentum, 12.60% CAGR /
0.54 Sharpe) still loses to the index on return, risk-adjusted return, and drawdown.

## 4. Excess return vs SPY — the decisive table

| Track / factor | Excess vs SPY | t | Years beating SPY |
|---|---|---|---|
| liquid / momentum_12_1 | +0.070%/mo | +0.11 | 3 of 7 |
| micro-cap / momentum_12_1 | +0.190%/mo | +0.28 | 2 of 7 |
| liquid / low_volatility | −0.808%/mo | **−2.98** | 2 of 7 |
| micro-cap / low_volatility | −1.168%/mo | **−2.82** | 1 of 7 |

Momentum's excess is statistically indistinguishable from zero. Low-volatility is *significantly
negative*.

### Per-year (%), top decile vs SPY

| Year | liquid mom | liquid lowvol | micro mom | micro lowvol | SPY |
|---|---|---|---|---|---|
| 2018 | −9.47 | 0.44 | −7.76 | −1.99 | −0.56 |
| 2019 | 15.81 | 9.66 | 8.00 | 7.20 | 19.25 |
| 2020 | **85.68** | −3.10 | **95.76** | 3.04 | 15.04 |
| 2021 | −25.50 | 11.96 | −0.15 | −2.16 | 21.55 |
| 2022 | −4.14 | −5.62 | −1.08 | −7.79 | −9.65 |
| 2023 | 7.38 | 0.55 | −1.43 | −2.44 | 18.80 |
| 2024 | 33.66 | 8.78 | 15.31 | 0.96 | 24.63 |

**The same pathology, a fourth time.** Momentum's entire CAGR comes from 2020 (+85.7% / +95.8%).
Strip that one year and the record is 2018 −9.5, 2019 +15.8, 2021 −25.5, 2022 −4.1, 2023 +7.4,
2024 +33.7 — including a −25.5% momentum crash in 2021. Previously the concentration was
cross-sectional (1% of trades); here it is temporal (1 of 7 years). Same failure to diversify away
from a single lucky episode.

## 5. The informative anomaly: significant IC, no long-only profit

Micro-cap momentum (IC t=4.23) and micro-cap low-volatility (IC t=4.77) have **highly significant
cross-sectional information coefficients** — real rank-ordering signal — yet their long-only top
decile does not beat SPY, and low-volatility loses money outright (−0.72% CAGR, Sharpe −0.06).

The most likely reading: the exploitable part of the signal lives in the **bottom** of the
cross-section (avoiding/shorting the worst names), not the top. That is consistent with every
other result in this project — the junk-gapper fade being the one robust edge — and it is
precisely the leg the user has ruled out.

A caution on micro-cap low-volatility specifically: thinly traded names have stale prints, which
mechanically depress measured volatility. A large part of that IC may be a data artifact rather
than a tradeable property.

## 6. Caveats

- **Survivorship bias (dominant).** Universe = symbols active at fetch time; delisted/bankrupt/
  acquired names absent. Inflates long-only results, severely for micro-cap. Micro-cap figures are
  an **upper bound**. Fixing needs a survivorship-free constituent source
  (`bot/data/universe_eodhd.py`, requires an EODHD key ~$20–50/mo).
- **No delisting returns modeled** (a delisted name should realize ≈ −100%, not vanish).
- **No fundamental factors.** Value, profitability/quality and investment have the strongest
  replication record in the literature; none could be tested here. This study covers the *weaker*
  half of the factor zoo.
- Costs are flat 20 bps per unit turnover — optimistic for micro-caps, where spreads alone often
  exceed that. Micro-cap turnover averaged 0.74–1.6 per rebalance.
- Trades assume execution at the rebalance close; next-open execution would be more conservative.
- 83 rebalances / 7 years is a short sample for factor inference.

## 6b. UPDATE — fundamentals wired in (SimFin free tier)

SimFin key added; `simfin` installed; bulk download + point-in-time store now working.
Run with `experiments/run_factor_study.py --fundamentals`.

### Three loader bugs found and fixed (all silent — none raised an error)

1. **`FREE_CASH_FLOW` does not exist in `simfin.names` (>=1.0).** The whole
   `from simfin.names import (...)` block raised ImportError, setting
   `_SIMFIN_AVAILABLE = False`, which surfaced later as a misleading
   *"simfin is not installed"*. Now derived: `FCF = Net Cash from Operating
   Activities + Change in Fixed Assets` (capex is negative, so a sum).
2. **The annual balance sheet labels its fiscal period `Q4`, not `FY`.** The
   hardcoded `== "FY"` filter reduced the balance sheet to **zero rows**, so
   `total_assets` / `total_equity` merged in as 100% NaN and every quality
   factor (ROE, leverage) was silently unusable. Now accepts both labels, with
   a warning if a filter would empty a frame.
3. **`EPS Diluted`, `EBITDA` and `Total Debt` are not columns of the free
   annual statements** (they live in the paid Derived Figures dataset). Now
   derived from raw line items. Coverage after fixes: total_assets/equity/debt
   95.9%, eps_diluted 98.3%, fcf 90.5%, ebitda 39.7%.

Also added `shares_diluted` to the store (needed for market-cap-relative
factors) and `CACHE_DIR`/`SIMFIN_API_KEY` to `bot/config.py` with `.env` loading.

### Coverage reality

Free tier gives **~3,000–3,600 annual filings/year for 2021–2025 only**. Against
a price cache ending 2025-01, the usable window is **48 monthly rebalances
(2021-02 → 2025-01)** with median ~1,600 covered names per rebalance. The study
auto-restricts to this span so price and fundamental factors are compared on the
same sample.

### Results, fundamental factors (48–51 rebalances; SPY same span: CAGR 12.84%, Sharpe 0.82, MaxDD −25.4%)

Liquid large/mid — nothing beat SPY:

| Factor | IC t | CAGR % | Sharpe |
|---|---|---|---|
| earnings_yield | 3.20 | 11.11 | 0.56 |
| fcf_yield | 3.19 | 9.17 | 0.48 |
| roe | 3.31 | 7.96 | 0.49 |
| low_leverage | −2.67 | −8.44 | −0.25 |

(`low_leverage` being significantly *negative* means levered names outperformed
over 2021–2025 — a regime artifact of that window, not a durable finding.)

Micro-cap — one cell beat SPY:

| Factor | IC t | CAGR % | Sharpe | MaxDD % |
|---|---|---|---|---|
| **fcf_yield** | **4.75** | **20.93** | **0.89** | **−22.4** |
| book_to_price | 1.28 | 13.74 | 0.55 | −30.6 |
| momentum_12_1 | 4.36 | 12.68 | 0.55 | −38.1 |
| roe | 3.99 | −7.37 | −0.24 | −42.6 |

### Micro-cap FCF yield against the standing acceptance test

| Check | Result | Verdict |
|---|---|---|
| Beats SPY | 20.93% vs 12.41% CAGR | pass |
| Survives costs | +14.84% CAGR even at 100 bps | **pass** |
| Bear-market behaviour | 2022: **+9.35%** vs SPY −9.65% | **pass** |
| Beat SPY per year | 3 of 5 | marginal |
| Statistical significance | excess +0.751%/mo, **t = +1.00** | **fail** |
| Profit concentration | **top 3 of 48 months = 51% of return** | **fail** |
| Ex-top-3 CAGR | **+8.85%** — below SPY's 12.41% | **fail** |

**Verdict: does NOT clear the bar.** Removing 3 months out of 48 drops it below
the benchmark, and the excess is not statistically distinguishable from zero.
This is the same concentration signature that killed the previous four
strategies — milder here (54% hit rate, cost-robust, genuinely defensive in
2022) but present.

It is nonetheless the **most promising result found in this project so far**, and
unlike the earlier candidates it is not obviously a fill/liquidity artifact. The
binding constraint is now sample size: 48 rebalances cannot separate a real
effect from noise, and survivorship bias is at its worst in a long-only
micro-cap portfolio.

### Known issue

Micro-cap `low_leverage` returns all zeros / NaN Sharpe — the debt/equity field
is too sparse in that universe for the decile mask to select anything. Needs a
minimum-coverage guard in `run_track()`.

## 7. What this implies

1. **Buy-and-hold SPY beat every strategy tested in this project** — gap-hold longs, PEAD, episodic
   pivot, intraday momentum, and now all eight factor/universe cells. That is the honest
   benchmark to measure against, and nothing has cleared it yet.
2. The pipeline is now **live and reusable** — this is the durable gain. Adding a SimFin key
   (free tier = annual fundamentals) unlocks value/quality/investment, the factors that actually
   survive replication, using the same driver.
3. The recurring lesson across four independent studies: **any result whose profit concentrates
   into one year or one percent of trades is noise.** Worth making that the standing acceptance
   test for future work here.
