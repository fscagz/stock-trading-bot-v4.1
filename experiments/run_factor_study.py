"""
Cross-sectional factor study on cached daily bars — liquid AND micro-cap tracks.

Motivation: every concentrated intraday/gapper strategy tested in this repo has
failed the same way (profit concentrated in ~1% of trades). This tests the
opposite construction: many small positions, monthly rebalance, ranked
cross-sectionally on price-based factors, held for weeks.

Reuses the repo's dormant systematic pipeline: backtest.metrics (IC, IC t-stat,
compute_metrics) and backtest.costs (CostModel).

DATA CHOICE — deliberate:
Only `screener_cache/2017-06-01_2025-01-31.pkl` is used. It was built with
Alpaca `adjustment="split"` (see experiments/build_daily_history_2018_2024.py)
and is therefore split-adjusted. The 2024-2026 pkl is NOT split-adjusted and
would corrupt every momentum/reversal factor with fake reverse-split "returns"
(documented in research/2026-08-20_surge_cohort_study.md).

SURVIVORSHIP BIAS — the dominant caveat:
The universe is symbols that were ACTIVE on Alpaca at fetch time. Companies
that delisted, went bankrupt, or were taken under are absent. This inflates
long-only results, and the distortion is far worse in the micro-cap track,
where delisting is common. Micro-cap numbers here should be read as an UPPER
BOUND, not an estimate. Fixing this requires a survivorship-free constituent
source (bot/data/universe_eodhd.py, needs an EODHD key).

Factors are price-based only, because no fundamental data source is configured
(no SIMFIN_API_KEY / EODHD_API_KEY). Value and quality factors require those.

Usage:
    .venv/bin/python3.11 experiments/run_factor_study.py
    .venv/bin/python3.11 experiments/run_factor_study.py --top-pct 0.1 --cost-bps 20
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "bot")]

from backtest.metrics import compute_ic_series, ic_tstat, compute_metrics  # noqa: E402

CACHE = ROOT / "screener_cache" / "2017-06-01_2025-01-31.pkl"
BENCH = "SPY"


def log(m):
    print(m, flush=True)


def is_derivative(sym: str, universe: set) -> bool:
    for s in ("WS", "WW", "W", "U", "R"):
        if sym.endswith(s) and len(sym) > len(s) and sym[: -len(s)] in universe:
            return True
    if 4 <= len(sym) <= 6 and sym.isalpha() and sym.endswith(("WS", "WW", "W")):
        return True
    return False


def build_panels():
    log(f"loading {CACHE.name} ...")
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    universe = set(cache)
    closes, dvols = {}, {}
    for sym, df in cache.items():
        if sym != BENCH and is_derivative(sym, universe):
            continue
        if len(df) < 300:
            continue
        df = df[~df.index.duplicated(keep="last")].sort_index()
        c = df["close"].astype(float)
        if (c <= 0).all():
            continue
        closes[sym] = c
        dvols[sym] = (c * df["volume"].astype(float))
    close = pd.DataFrame(closes).sort_index()
    dvol = pd.DataFrame(dvols).reindex_like(close)
    log(f"panel: {close.shape[0]} days x {close.shape[1]} symbols "
        f"({close.index.min().date()} → {close.index.max().date()})")
    return close, dvol


def month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).last().values)


def build_factors(close: pd.DataFrame, dvol: pd.DataFrame, rb: pd.DatetimeIndex):
    """All factors use data through the rebalance date only (no look-ahead).
    Higher score = more attractive."""
    c = close
    ret = c.pct_change()

    mom = (c.shift(21) / c.shift(252) - 1.0)          # 12-1 momentum, skips last month
    rev = -(c / c.shift(21) - 1.0)                     # short-term reversal (contrarian)
    vol60 = ret.rolling(60).std()
    lowvol = -vol60                                    # low-volatility
    liq = dvol.rolling(20).mean()
    small = -np.log(liq.replace(0, np.nan))            # smaller (illiquid) = higher score

    return {
        "momentum_12_1": mom.reindex(rb),
        "reversal_1m": rev.reindex(rb),
        "low_volatility": lowvol.reindex(rb),
        "small_size": small.reindex(rb),
    }


def build_fundamental_factors(close: pd.DataFrame, rb: pd.DatetimeIndex):
    """Value/quality factors from the SimFin point-in-time store.

    Point-in-time discipline: for each rebalance date we take the latest filing
    whose FILING date (not fiscal period end) is on or before that date, via a
    backward merge_asof. Using period_end_date would leak — a Dec-31 fiscal year
    is not public until the 10-K is filed months later.

    Returns {} when the store is empty or SIMFIN_API_KEY is unset, so the caller
    degrades to price-only factors rather than failing.
    """
    try:
        from data.fundamental_store import load_fundamentals
        f = load_fundamentals(source="simfin")
    except Exception as exc:
        log(f"fundamentals unavailable ({type(exc).__name__}: {exc}) — price factors only")
        return {}
    if f is None or f.empty:
        log("fundamental store is empty — price factors only")
        return {}

    f = f.copy()
    # merge_asof requires identical datetime resolution on both keys; the store
    # round-trips through Parquet as datetime64[s] while the price index is [ns].
    f["filing_date"] = pd.to_datetime(f["filing_date"]).astype("datetime64[ns]")
    f = f.dropna(subset=["filing_date", "ticker"]).sort_values("filing_date")
    rb = pd.DatetimeIndex(rb).astype("datetime64[ns]")

    tickers = [t for t in close.columns if t != BENCH]
    # as-of join each rebalance date against each ticker's filing history
    frames = []
    for tkr, g in f[f["ticker"].isin(tickers)].groupby("ticker", sort=False):
        g = g.sort_values("filing_date")
        j = pd.merge_asof(
            pd.DataFrame({"filing_date": rb}), g,
            on="filing_date", direction="backward",
        )
        # annual filings arrive ~yearly; anything older than ~500 days is stale.
        # (fundamental_store's own 180-day default would discard nearly every
        # annual filing, since consecutive 10-Ks are ~365 days apart.)
        age = (rb - j["filing_date"]).dt.days
        stale = age.isna() | (age > 500)
        val_cols = [c for c in j.columns if c not in ("filing_date", "ticker")]
        j.loc[stale.values, val_cols] = np.nan
        j["ticker"] = tkr
        j.index = rb
        frames.append(j)
    if not frames:
        return {}
    panel = pd.concat(frames)

    def wide(col):
        if col not in panel.columns:
            return pd.DataFrame(index=rb, columns=tickers, dtype=float)
        return panel.pivot_table(index=panel.index, columns="ticker", values=col,
                                 aggfunc="last").reindex(index=rb, columns=tickers)

    px = close.reindex(rb)[tickers]
    eps, ni = wide("eps_diluted"), wide("net_income")
    equity, revenue = wide("total_equity"), wide("revenue")
    gp, debt = wide("gross_profit"), wide("total_debt")
    fcf, sh = wide("free_cash_flow"), wide("shares_diluted")
    mcap = px * sh

    covered = eps.notna().sum(axis=1)
    log(f"fundamental coverage: median {covered.median():.0f} names/rebalance "
        f"(max {covered.max():.0f})")

    return {
        # value — higher = cheaper
        "earnings_yield": eps / px.replace(0, np.nan),
        "book_to_price": equity / mcap.replace(0, np.nan),
        "fcf_yield": fcf / mcap.replace(0, np.nan),
        # quality — higher = better
        "roe": ni / equity.where(equity > 0),
        "gross_margin": gp / revenue.where(revenue > 0),
        "low_leverage": -(debt / equity.where(equity > 0)),
    }


def universe_masks(close: pd.DataFrame, dvol: pd.DataFrame, rb: pd.DatetimeIndex):
    px = close.reindex(rb)
    liq = dvol.rolling(20).mean().reindex(rb)
    return {
        # dollar volume is the only size proxy available without fundamentals
        "liquid_large_mid": (liq >= 20e6) & (px >= 5.0),
        "microcap": (liq >= 1e5) & (liq < 5e6) & (px >= 1.0),
    }


def run_track(name, factors, mask, close, rb, top_pct, cost_bps, bench_ret_d):
    log(f"\n{'='*94}\nTRACK: {name}\n{'='*94}")
    px = close.reindex(rb)
    fwd = px.shift(-1) / px - 1.0            # next-month return, the prediction target
    counts = mask.sum(axis=1)
    log(f"universe size per rebalance: median {counts.median():.0f} "
        f"(min {counts.min():.0f}, max {counts.max():.0f})")

    rows = []
    for fname, fval in factors.items():
        sig = fval.where(mask)
        fw = fwd.where(mask)
        valid = sig.notna().sum(axis=1) >= 30
        sig, fw = sig[valid], fw[valid]
        if len(sig) < 24:
            log(f"  {fname}: insufficient rebalances")
            continue

        ic = compute_ic_series(sig, fw)
        t = ic_tstat(ic)

        # long-only top-decile, equal weight, monthly rebalance
        rk = sig.rank(axis=1, pct=True, ascending=False)
        sel = rk <= top_pct
        w = sel.div(sel.sum(axis=1), axis=0).fillna(0.0)
        gross = (w * fw.fillna(0.0)).sum(axis=1)
        turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
        net = gross - turn * (cost_bps / 10000.0)

        # decile spread: pure signal diagnostic (not tradeable long-only)
        selb = rk >= (1 - top_pct)
        wb = selb.div(selb.sum(axis=1), axis=0).fillna(0.0)
        spread = gross - (wb * fw.fillna(0.0)).sum(axis=1)

        eq = (1 + net).cumprod()
        yrs = max((net.index[-1] - net.index[0]).days / 365.25, 1e-9)
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        sharpe = (net.mean() / net.std() * np.sqrt(12)) if net.std() > 0 else np.nan
        dd = (eq / eq.cummax() - 1).min()

        rows.append({
            "factor": fname, "n_rb": len(net),
            "IC_mean": ic.mean(), "IC_t": t,
            "CAGR_%": cagr * 100, "Sharpe": sharpe, "MaxDD_%": dd * 100,
            "hit_%": (net > 0).mean() * 100,
            "spread_mean_%": spread.mean() * 100,
            "spread_t": spread.mean() / spread.std() * np.sqrt(len(spread)) if spread.std() > 0 else np.nan,
            "turnover": turn.mean(),
        })
        globals().setdefault("_curves", {})[(name, fname)] = net

    res = pd.DataFrame(rows)
    if res.empty:
        log("  no factor produced enough data")
        return res
    log("\n" + res.round(3).to_string(index=False))

    # benchmark over the same span
    span = (bench_ret_d.index >= rb[0]) & (bench_ret_d.index <= rb[-1])
    b = bench_ret_d[span]
    beq = (1 + b).cumprod()
    byrs = max((b.index[-1] - b.index[0]).days / 365.25, 1e-9)
    log(f"\nBenchmark {BENCH} same span: CAGR {(beq.iloc[-1]**(1/byrs)-1)*100:.2f}%  "
        f"Sharpe {b.mean()/b.std()*np.sqrt(252):.2f}  "
        f"MaxDD {(beq/beq.cummax()-1).min()*100:.1f}%")
    return res


def per_year(name, factor):
    c = globals().get("_curves", {}).get((name, factor))
    if c is None:
        return
    log(f"\nper-year net return — {name} / {factor}:")
    y = c.groupby(c.index.year).apply(lambda s: (1 + s).prod() - 1) * 100
    log(y.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-pct", type=float, default=0.10, help="top fraction held (0.10 = decile)")
    ap.add_argument("--cost-bps", type=float, default=20.0, help="one-way cost per unit turnover")
    ap.add_argument("--fundamentals", action="store_true",
                    help="add SimFin value/quality factors (needs SIMFIN_API_KEY + built store)")
    args = ap.parse_args()

    close, dvol = build_panels()
    rb = month_ends(close.index)
    rb = rb[(rb >= close.index[0] + pd.Timedelta(days=300)) & (rb <= close.index[-1])]
    log(f"rebalances: {len(rb)}  ({rb[0].date()} → {rb[-1].date()})")

    bench_d = close[BENCH].pct_change().dropna() if BENCH in close else pd.Series(dtype=float)

    factors = build_factors(close, dvol, rb)
    if args.fundamentals:
        fund = build_fundamental_factors(close, rb)
        if fund:
            # Fundamentals only exist for part of the price history; restrict the
            # whole study to that span so price and fundamental factors are
            # compared over the SAME rebalances rather than different samples.
            any_cov = pd.concat([v.notna().sum(axis=1) for v in fund.values()], axis=1).max(axis=1)
            live = any_cov[any_cov >= 30].index
            if len(live) >= 12:
                log(f"restricting study to fundamental-covered span: "
                    f"{live[0].date()} → {live[-1].date()} ({len(live)} rebalances)")
                rb = pd.DatetimeIndex(live)
                factors = build_factors(close, dvol, rb)
                fund = build_fundamental_factors(close, rb)
            factors.update(fund)
    masks = universe_masks(close, dvol, rb)

    all_res = {}
    for uname, mask in masks.items():
        all_res[uname] = run_track(uname, factors, mask, close, rb,
                                   args.top_pct, args.cost_bps, bench_d)

    log(f"\n{'='*94}\nPER-YEAR DETAIL (best factor by IC_t in each track)\n{'='*94}")
    for uname, res in all_res.items():
        if res.empty:
            continue
        best = res.iloc[res["IC_t"].abs().values.argmax()]["factor"]
        per_year(uname, best)

    log("\n" + "!"*94)
    log("SURVIVORSHIP BIAS: universe = symbols active at fetch time. Delisted names are")
    log("absent, which inflates long-only results — severely so for the micro-cap track.")
    log("Treat micro-cap numbers as an UPPER BOUND. No delisting returns are modeled.")
    log("!"*94)


if __name__ == "__main__":
    main()
