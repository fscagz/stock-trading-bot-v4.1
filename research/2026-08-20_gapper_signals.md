# Signal investigation: 2026-08-20 top movers (JZ, BRLS, BTCT, INDP)

> **SUPERSEDED IN PART — see `2026-08-20_surge_cohort_study.md`**, which generalizes this to
> 11,420 events with a control group. Two claims below did not survive that test:
> 1. **Short interest / squeeze fuel is NOT a common trait.** Across 788 surge names, short % of
>    float is statistically identical to control (7.32% vs 7.33%). BTCT's 25.6% and BRLS's high
>    borrow rate were cherry-picked from a 4-name sample.
> 2. **Catalyst quality is a weak signal.** It is genuinely thin/absent for most surgers
>    (>50% have no real catalyst), but it barely predicts behavior — nearly every catalyst type
>    fades −20% to −25%.
>
> What *did* generalize: small float (the dominant signal, p≈6e-100), sub-$5 price, healthcare
> over-representation, and the same-day fade from the high.

Follow-up to the historical fade-rate study (8,799 events 2017–2025, 4,801 events 2024–2026,
intraday spike ≥50%, SPAC warrants/units/rights excluded via ticker-root heuristic). That study
found a robust ~20–22% median same-day fade from the intraday high, ~75–80% of events closing
≥10% off the high, and — in the 2024–2026 window specifically — 84% of events priced under $5.
This pass drills into *why* these four specific names moved: float, sector, and catalyst quality.

Data pulled 2026-08-20 via finviz (float/shares/sector) and web search (news/catalyst). Finviz
float figures are self-reported/vendor-estimated and can lag actual free float after recent
offerings — treat as directional, not exact.

**Correction:** in the prior conversation turn, JZ was misidentified as Zhengye Biotechnology
(ticker ZYBT). They are different companies — ZYBT data has been discarded. JZ is Jianzhi
Education Technology Group (see below).

## Summary table

| Ticker | Company | Sector / Industry | Price | Market Cap | Shares Out | Float | Short Float | Avg Volume |
|---|---|---|---|---|---|---|---|---|
| JZ | Jianzhi Education Technology Group | Technology / IT Services | $3.32 | $8.00M | 2.41M | not disclosed | — | — |
| BRLS | Borealis Foods | Consumer Defensive / Packaged Foods | $1.75 | $37.55M | 21.46M | 6.47M | 2.82% | 22.45K |
| BTCT | BTC Digital | Technology / Computer Hardware | $1.47 | $13.98M | 9.52M | 8.62M | 25.60% | 3.94M |
| INDP | Indaptus Therapeutics | Healthcare / Biotechnology | $1.38 | $183.88M | 133.24M | 4.74M | 1.97% | 1.04M |

Note on INDP: 133.24M shares outstanding but only 4.74M float (3.6%) — the widest
outstanding/float gap of the four, consistent with a large recent dilutive raise where most new
shares are still restricted/unregistered.

## Per-ticker detail

### JZ — Jianzhi Education Technology Group
- **Business:** Educational content and IT services for Chinese higher-ed institutions.
- **Fundamentals:** 2025 revenue $70.18M, down 71.8% YoY (core business is shrinking sharply).
- **Recent corporate actions:** ADS ratio changed from 1 ADS : 6 ordinary shares to 1 ADS : 60
  ordinary shares (~10x share-count consolidation, reverse-split-like — mechanically concentrates
  float and inflates % price swings) (~7 weeks before this date). ADR trading was halted with
  "news pending" (~6 weeks before this date).
- **Catalyst quality:** Partnership/collaboration announcements with SeaArt AI and DeepSeek AI
  integration, plus a $5M registered direct offering (~2 months prior) — AI-buzzword narrative
  layered on a company whose actual revenue is collapsing, not fundamentals-driven.

### BRLS — Borealis Foods
- **Business:** Packaged instant-noodle foods (Chef Woo, Palermo's brands).
- **Fundamentals:** Real revenue growth (+8% YoY per Q1 2026), improving operating loss,
  refinanced its primary secured debt to remove an August 2026 balloon maturity.
- **Float:** Tight — only 6.47M of 21.46M shares outstanding actually float (30%).
- **Catalyst quality:** Least clearly a pure lottery pump of the four — has an actual operating
  business with improving metrics. Search results reference "acquisition news" and the company
  is flagged as carrying the highest stock-borrow rate on the market, which independently fuels
  volatility (shorts get squeezed, can't easily re-short). Worth tracking as the one name that
  might have a genuine fundamental leg under the move.

### BTCT — BTC Digital
- **Business:** Formerly crypto-mining-adjacent hardware/computing; announced pivot to AI
  computing infrastructure.
- **Catalyst:** Appointed Wei Sun (ex-SF Supply Chain, JD Digits co-founder) as "Chief AI
  Business Growth Officer" (announced ~3 weeks before this date) to build out AI data-center
  services. No revenue or contracts disclosed yet — a personnel announcement and a stated
  intent, not a signed deal.
- **Float mechanics:** 25.6% short float against an 8.62M-share float and 3.94M average daily
  volume — float turns over almost fully every ~2 trading days on average even before a spike
  day. This is short-squeeze-primed structurally, independent of the AI narrative.
- **Intraday behavior (from earlier pass):** opened $0.48, spiked to $1.32, closed $0.8367 —
  a −36.6% fade from the high, worse than the ~−20% historical median.

### INDP — Indaptus Therapeutics
- **Business:** Clinical-stage biotech (Decoy immunotherapy platform).
- **Fundamentals:** Cash balance sharply reduced as of Q1 2026; company explicitly stated it is
  evaluating financing alternatives and strategic options. Filed a new securities-offering
  registration statement (~2 months before this date). History of private placements of
  convertible notes/warrants and a standby equity purchase agreement — repeated dilutive
  financing pattern.
- **Catalyst quality:** The June 2026 +53.8% jump had no clear announced catalyst tied to it —
  attributed to investor speculation around financing/strategic-option uncertainty, i.e. betting
  on an acquisition or buyout rather than any disclosed news. Same ambiguous-catalyst pattern
  repeating in August.
- **Float mechanics:** Extreme outstanding/float gap (133.24M outstanding vs. 4.74M float) —
  most shares are not actually tradeable, meaning even modest dollar buying pressure moves the
  quoted price disproportionately.

## Cross-ticker common threads

1. **All sub-$5, three of four sub-$2.** Market caps range $8M–$184M; three of four are under
   $40M. This matches the 2024–2026 historical regime finding that 84% of ≥50%-intraday-spike
   events are priced under $5.
2. **Float is small and, in two cases (BTCT, INDP), dramatically smaller than shares
   outstanding.** Small floats mean normal-sized buy orders produce outsized % price moves —
   this is a mechanical amplifier independent of any real news.
3. **High or elevated short interest on at least one (BTCT 25.6%), and BRLS independently
   flagged as the market's highest borrow-rate name.** Squeeze dynamics (shorts forced to cover
   into a thin float) are a plausible mechanical driver alongside or instead of the news.
4. **Catalyst quality is thin for 3 of 4.** JZ and BTCT both lean on AI-buzzword corporate
   narratives (partnerships/pivots/hires) with no revenue attached yet. INDP's moves have
   repeatedly lacked any disclosed same-day catalyst at all, tied instead to financing-distress
   speculation. BRLS is the outlier with real, if modest, fundamental improvement.
5. **Recent corporate actions that mechanically shrink or reshuffle float** (JZ's 10x ADS
   consolidation, INDP's recent dilutive offerings creating a large outstanding/float gap) show
   up in two of the four — worth checking as a screenable pre-condition in future study.

## What this doesn't establish

This is a same-day snapshot of four names, not a statistical test. It's consistent with the
prior quantitative finding (thin float + weak/no catalyst + micro-cap price level → same-day
fade from the high, historically ~20% median) but doesn't itself prove continuation or fade for
these four specifically going forward. No trading conclusion should be drawn from this file alone.
