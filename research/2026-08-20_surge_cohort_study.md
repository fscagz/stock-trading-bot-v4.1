# What do one-day explosive movers have in common? A cohort study

Generalizes the four-name pass in `2026-08-20_gapper_signals.md` (JZ, BRLS, BTCT, INDP) to
**11,420 historical surge events across 2017–2026**, with a matched control group, to test which
traits are actually distinctive rather than merely present.

**Bottom line up front:** float size is overwhelmingly the distinguishing trait (median 5.68M vs
52.43M shares, p ≈ 6e-100). Sector matters mildly (healthcare 2.2x over-represented). Catalyst
quality and short interest do **not** distinguish these names — over half of surge days have no
real company catalyst at all, and short interest is statistically indistinguishable from control.
And the "enter early and ride it" trade is negative in every year tested, monotonically worse the
bigger the gap.

---

## 1. Method

**Event definition:** intraday high ≥ +50% above prior close, on daily bars.
**Sources:** `screener_cache/2017-06-01_2025-01-31.pkl` (split-adjusted, built by
`experiments/build_daily_history_2018_2024.py`) and `screener_cache/2024-11-28_2026-06-01.pkl`.
**Control group:** symbols in the same universe/window that *never* produced a ≥50% surge
(n=742 with fundamentals). Without this baseline, statements like "84% are under $5" are
meaningless — most micro-caps are under $5 regardless.
**Fundamentals:** yfinance (sector, float, shares outstanding, short % of float, market cap).
**Catalysts:** 51,983 cached Benzinga news JSONs in `backtest_results/news_cache`, headline-
classified by regex into archetypes.

### Data-quality fixes applied (both materially changed the results)

1. **SPAC warrants/units/rights excluded.** Tickers like `OXBRW`, `XBPEW`, `PBMWW` are leveraged
   derivatives that swing hugely on small underlying moves; they dominated the raw "repeat
   offender" rankings and are not common stock. ~550–900 symbols removed per window.
2. **Reverse-split artifacts excluded (critical).** The 2024-11→2026-06 cache is **not**
   split-adjusted. A reverse split appears as a massive "spike" with *collapsing* volume. Example:
   INDP on 2025-06-27 shows $0.459 → $12.41 (+2,603%) while volume fell from 3.5M to 41K — a
   ~1:27 reverse split, not a rally. Diagnostic: for spike >250% events, median relative volume
   was **0.29** in the unadjusted cache vs **278x** in the split-adjusted one. Genuine surges have
   enormous volume expansion. **Filter: require relative volume ≥ 3.0.** This removed 14.4% of
   raw events (1,925 of 13,345).

> **Correction to the prior turn's reported figures.** The 2024–2026 price-bucket numbers reported
> before this filter existed (e.g. "84% under $5, 46% under $1") were inflated by these split
> artifacts, since reverse splits happen disproportionately to sub-$1 stocks. Corrected figures
> are in §3. The core fade finding was unaffected in direction, but its relationship to spike size
> was actually *reversed* by the contamination (see §2).

---

## 2. The core structural fact: the fade

Measured as close vs. the same day's intraday high, on split-filtered events.

| Window | n | Median fade from high | % closing ≥10% off high |
|---|---|---|---|
| 2017–2025 | 7,812 | **−22.4%** | 81.3% |
| 2024–2026 | 3,608 | **−22.5%** | 81.3% |
| **All** | **11,420** | **−22.4%** | **81.3%** |

Overall: mean −23.9%, **t = −173.0**, p ≈ 0.

### Per-year stability — it never breaks

| Year | n | Median fade | % negative | t |
|---|---|---|---|---|
| 2017 | 199 | −23.0% | 97.5% | −24.8 |
| 2018 | 429 | −24.2% | 97.9% | −35.8 |
| 2019 | 472 | −23.3% | 99.6% | −37.2 |
| 2020 | 1,428 | −23.1% | 99.0% | −62.7 |
| 2021 | 842 | −21.0% | 98.3% | −46.4 |
| 2022 | 855 | −20.0% | 98.7% | −47.9 |
| 2023 | 1,291 | −22.2% | 99.3% | −61.4 |
| 2024 | 2,107 | −22.9% | 98.6% | −73.8 |
| 2025 | 2,705 | −23.0% | 99.3% | −83.2 |
| 2026 | 1,092 | −21.8% | 99.1% | −50.2 |

Bull markets, bear markets, the 2020 crash, the 2021 meme era — the fade is ~−22% every single
year, with 97–99% of events closing below their high.

### Bigger spike → harder fade (monotonic, on clean data)

| Intraday spike | n | Median fade from high | % ≥10% off high | Median rvol |
|---|---|---|---|---|
| 50–75% | 5,189 | −17.9% | 74.7% | 18.9x |
| 75–100% | 2,334 | −23.7% | 84.6% | 36.7x |
| 100–150% | 1,922 | −27.6% | 87.2% | 67.0x |
| 150–250% | 1,150 | −31.6% | 89.3% | 164.5x |
| 250%+ | 771 | **−37.5%** | 89.2% | 240.6x |

Perfectly monotonic. **On the contaminated data this relationship was non-monotonic and appeared
to reverse at the top bucket** — the split artifacts (which have ~zero fade, since they open and
close near the new post-split price) were diluting the most extreme genuine events. This is the
clearest example of why the filter mattered.

The four names that started this investigation are the +75% to +95% class → the −24% to −32%
fade bands.

---

## 3. Trait 1 — FLOAT (the dominant signal)

Surge symbols vs. control symbols, current fundamentals:

| Metric (median) | Surge (n=788) | Control (n=742) | Ratio |
|---|---|---|---|
| **Float** | **5.68M shares** | **52.43M shares** | **0.11x** |
| Shares outstanding | 10.11M | 67.02M | 0.15x |
| Market cap | $21.3M | $1,668.1M | 0.013x |
| Price | $2.33 | $21.86 | 0.11x |

**Mann-Whitney U (surge float < control float): p = 6.05e-100.** This is not a marginal effect.

### Float distribution

| Float bucket | Surge % | Control % | Lift |
|---|---|---|---|
| <5M | 45.6% | 6.2% | **7.35x** |
| 5–15M | 23.5% | 11.5% | 2.04x |
| 15–50M | 17.1% | 24.4% | 0.70x |
| 50–200M | 8.1% | 28.3% | 0.29x |
| 200M+ | 2.0% | 16.2% | 0.12x |

69% of surgers have a float under 15M shares vs 18% of control.

### Locked-up supply

Float as a share of total shares outstanding (vendor-error rows with ratio >1 dropped — 231 of
1,530): surge median **0.660** vs control **0.863**. Surge names have materially more of their
share count locked up/unregistered, so the tradeable supply is even thinner than the float
number alone suggests. This matches INDP's extreme case (4.74M float against 133.24M outstanding).

### Float also predicts fade severity

| Float bucket | n | Median fade |
|---|---|---|
| <5M | 1,394 | −25.5% |
| 5–15M | 579 | −22.5% |
| 15–50M | 365 | −23.5% |
| 50–200M | 171 | −19.3% |
| 200M+ | 45 | −18.5% |

Thinner float → bigger spike *and* bigger give-back. Consistent with a mechanical
liquidity-driven interpretation: thin supply lets price overshoot in both directions.

---

## 4. Trait 2 — PRICE LEVEL (real, but mostly a proxy for float)

Surge events vs control, 2024–2026 window, split-filtered:

| Prior-day price | Surge % | Control % | Lift |
|---|---|---|---|
| <$1 | 41.9% | 0.7% | **59.9x** |
| $1–2 | 20.9% | 2.7% | 7.7x |
| $2–5 | 22.0% | 7.0% | 3.1x |
| $5–10 | 9.1% | 11.3% | 0.81x |
| $10–20 | 3.8% | 20.2% | 0.19x |
| $20+ | 2.2% | 58.1% | 0.04x |

85% of surge events start under $5. The <$1 lift of ~60x is the single largest ratio in this
study, but price and float are heavily collinear — this is largely the same phenomenon as §3.

---

## 5. Trait 3 — SECTOR (mild frequency effect, no behavioral effect)

| Sector | Surge % | Control % | Lift |
|---|---|---|---|
| Healthcare | 33.8% | 15.5% | **2.18x** |
| Communication Services | 5.6% | 3.5% | 1.60x |
| Consumer Defensive | 5.1% | 3.5% | 1.46x |
| Technology | 15.4% | 11.0% | 1.40x |
| Industrials | 13.1% | 9.9% | 1.32x |
| Consumer Cyclical | 10.5% | 8.9% | 1.18x |
| Basic Materials | 3.1% | 5.6% | 0.55x |
| Real Estate | 2.9% | 4.9% | 0.59x |
| Energy | 1.4% | 4.1% | 0.34x |
| Financial Services | 7.6% | 30.3% | 0.25x |

Healthcare (clinical-stage biotech: binary catalysts, no revenue, chronic dilution) is 2.2x
over-represented — a third of all surge events. Financial Services is heavily *under*-represented,
though this partly reflects the control being rich in banks/REITs/funds.

**But sector does not predict behavior.** Median fade by sector spans only −20.9% (Energy) to
−25.9% (Utilities); healthcare is −24.2%. Sector tells you *which* names spike, not what happens
after. It is not a useful conditioning variable for a trade.

---

## 6. Trait 4 — CATALYST QUALITY (weak signal; mostly absent)

3,772 split-filtered surge events with news coverage. **Coverage caveat:** the cache holds no
2017–2021 files, so this section describes 2022–2026 only, and coverage is partial (36.5% of
events), skewed toward names touched by prior backtests.

**53.6% of surge days have no real company catalyst at all** — only reactive coverage
("12 Health Care Stocks Moving In Friday's Session"), exchange volatility-halt notices, or
nothing whatsoever.

| Catalyst archetype | n | Median fade | Median spike |
|---|---|---|---|
| movers roundup (reactive only) | 966 | −20.3% | 80.7% |
| none (no news at all) | 935 | −20.2% | 76.9% |
| other | 788 | −20.7% | 101.1% |
| earnings | 308 | −18.6% | 87.0% |
| analyst action | 164 | **−12.0%** | 70.7% |
| clinical trial / FDA | 138 | −22.2% | 85.0% |
| halt / circuit breaker | 120 | −27.8% | 123.5% |
| partnership / contract | 108 | −24.2% | 82.1% |
| offering / dilution | 91 | −24.6% | 100.0% |
| crypto / AI pivot | 64 | −21.5% | 91.2% |
| M&A / buyout | 64 | −21.7% | 93.8% |
| reverse split / compliance | 18 | −22.0% | 92.2% |

Two useful observations:

- **Catalyst type barely moves the fade.** Almost every archetype clusters at −20% to −25%. The
  fade is structural (float mechanics), not news-driven. Real news does not protect the move.
- **Analyst-driven moves fade least (−12.0%)** and are the smallest spikes (70.7%) — the one
  category with a genuinely different profile, plausibly because they occur in more liquid,
  better-covered names. **Offering/dilution (−24.6%) and partnership/contract (−24.2%) fade
  worst** among real catalysts. Volatility halts mark the most violent, worst-fading events
  (−27.8% on a median +123% spike).

This validates the four-name qualitative read (JZ and BTCT's AI-narrative press releases, INDP's
no-catalyst speculation) as *typical* rather than exceptional — but shows catalyst quality is not
a useful filter, because thin/absent catalysts are the norm and behave the same as real ones.

---

## 7. Trait 5 — SHORT INTEREST (**refuted** — no signal)

| Group | n | Mean short % of float | Median | 75th pct | 90th pct |
|---|---|---|---|---|---|
| Surge | 762 | 7.32% | 3.08% | 7.90% | 16.18% |
| Control | 646 | 7.33% | 4.45% | 9.28% | 16.17% |

**Statistically indistinguishable — surge names actually have slightly *lower* median short
interest.** 

> **Correction to the four-name pass.** That writeup highlighted BTCT's 25.6% short float and
> BRLS's high borrow rate as a "squeeze fuel" common thread. Across 788 surge names that does not
> hold up — it was cherry-picking two salient values from a four-name sample. Short interest is
> not a distinguishing characteristic of one-day explosive movers. (High short interest may still
> matter for *specific* squeezes; it is simply not a general trait of this cohort.)

---

## 8. The trade you actually asked about — and why the obvious test is a trap

The original goal was "detect these early in the day and ride the momentum." Testing that
correctly requires care.

### The look-ahead trap

Among the 11,420 surge events, open→close is **+22.6% mean / +16.7% median, 69.1% positive,
t = +46.1** — apparently a strong long edge. **It is not tradeable.** The cohort is *defined* by
the intraday high reaching +50%, which is information unavailable at 09:30. Selecting on the
outcome guarantees a flattering entry statistic. Any backtest that screens on "stocks that
surged today" and then measures from the open inherits this bias.

### The honest version: condition only on the opening gap (observable at 09:30)

Filtered to genuinely tradeable events (20-day avg dollar volume ≥ $1M, rvol ≥ 3):

**2017–2025 (split-adjusted):**

| Opening gap | n | Mean o→c | Median o→c | % positive | t |
|---|---|---|---|---|---|
| 20–30% | 1,468 | +0.1% | −2.0% | 43.5% | +0.2 |
| 30–50% | 948 | −1.6% | −5.9% | 34.0% | −2.0 |
| 50–100% | 549 | −2.3% | −11.1% | 29.7% | −1.3 |
| 100%+ | 186 | **−12.7%** | **−19.4%** | 22.6% | −5.0 |

**2024–2026:**

| Opening gap | n | Mean o→c | Median o→c | % positive | t |
|---|---|---|---|---|---|
| 20–30% | 721 | +2.7% | −0.1% | 49.1% | +2.0 |
| 30–50% | 513 | −2.1% | −4.8% | 38.6% | −1.8 |
| 50–100% | 304 | −4.4% | −9.2% | 31.2% | −2.7 |
| 100%+ | 162 | −5.2% | −13.7% | 27.8% | −1.4 |

Monotonic in both windows: **the bigger the gap, the worse buying the open performs.** Per-year
for gap ≥30%, the share of positive days is 23–37% — **it never exceeded 50% in any year from
2017 to 2026.**

### Why it feels like it should work

For gap ≥30% names, median open→high is **+11% to +18%** — the pop is real. But median open→low
is **−14% to −29%**, and it is generally hit too. You are buying an instrument whose median path
goes meaningfully against you before any favorable excursion, and whose median close is below
your entry. Capturing the +11% requires exiting near a high you cannot identify in advance;
holding to the close loses.

---

## 9. Summary: which traits are real?

| Trait | Distinctive vs control? | Predicts fade? | Verdict |
|---|---|---|---|
| **Float size** | **Yes — 0.11x, p≈6e-100** | Yes (−25.5% <5M vs −18.5% 200M+) | **Dominant signal** |
| Price level (<$5) | Yes — up to 60x lift | Weakly | Real, mostly collinear with float |
| Locked-up supply | Yes (0.66 vs 0.86) | Not tested directly | Real, amplifies float effect |
| Sector (healthcare) | Yes — 2.18x | No (all sectors −21% to −26%) | Frequency only, not behavior |
| Catalyst quality | Absent >50% of the time | Barely (−12% to −25% spread) | Weak; not a useful filter |
| **Short interest** | **No — 7.32% vs 7.33%** | Not tested | **Refuted** |

**The profile:** a sub-$5, sub-15M-float, sub-$50M-market-cap company — disproportionately
clinical-stage biotech — that on any given day may explode 50–100%+ on volume 20–240x its
average, more likely than not with no real news attached, and which then gives back a median
22% of the move by the close, with 97–99% consistency in every year since 2017.

---

## 10. What this does and does not establish

- These are **descriptive cohort characteristics plus one look-ahead-free directional test**, not
  a validated strategy. Nothing here has been run through a fill-realistic backtest.
- **Survivorship bias:** both caches contain only symbols still active at fetch time. Pump-and-dumps
  that delisted are missing. This biases the long side *upward* — real long results would be worse,
  and the fade finding correspondingly stronger.
- **Fundamentals are current, not point-in-time.** Float today ≠ float on a 2019 surge day,
  especially after offerings/reverse splits. Directional, not exact. The 2024–2026 restriction on
  the float comparison keeps this gap as small as practical.
- **News coverage is partial (36.5%) and 2022–2026 only**, skewed toward symbols touched by prior
  backtests.
- The short-side implication (fading these) is **not** validated here. Prior work
  (`project-2026-07-02-multiregime-verdict`) found the fade robust across 2018–2024 but with
  catastrophic tails (worst trades −215% to −483% of notional) and unmodeled hard-to-borrow costs
  of 50–300% annualized. Borrow availability remains the gating unknown.

## Reproduction

Scripts in session scratchpad (`extract_events.py`, `enrich.py`, `enrich_control.py`,
`news_quality.py`, `final_analysis.py`, `final_d.py`, `stats.py`, `tradeable.py`). Requires
`pyarrow` (installed to `.venv`) and `yfinance`. Note `.venv/bin/python` is a broken symlink to a
missing 3.13; use `.venv/bin/python3.11`.
