"""
SimFin fundamental data loader — point-in-time safe.

Every data point is tagged with its SimFin 'Publish Date', which is the date
the filing became publicly available (approximately the SEC filing date). This
is the correct date for point-in-time backtesting — it is NOT the fiscal
period end date.

Setup
-----
1. Obtain a free API key at https://simfin.com/api/v2/
2. Set SIMFIN_API_KEY in your .env (or environment)
3. Optionally set BOT_USE_SIMFIN=true (already default in config.py)

The free tier provides annual statements for US equities. A paid plan adds
quarterly data, which is better for higher-frequency signals.

Data Flow
---------
download_bulk() → caches raw SimFin data locally as Parquet (via SimFin's
  built-in bulk download mechanism)
build_fundamental_store() → parses and normalises the bulk data, then
  persists to fundamental_store.py (our canonical cache)
get_fundamentals_as_of() → point-in-time query for one ticker and date

Derived Metrics
---------------
The following metrics require a price at the rebalance date and are computed
here (not in fundamental_store) because price is loaded separately:
  pe_ratio, ev_ebitda, fcf_yield, accruals_ratio
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# SimFin's Python library handles bulk downloads and caching internally.
try:
    import simfin as sf
    from simfin.names import (
        TICKER, PUBLISH_DATE, REPORT_DATE, FISCAL_YEAR, FISCAL_PERIOD,
        REVENUE, NET_INCOME, EPS_DILUTED, GROSS_PROFIT, EBITDA,
        TOTAL_ASSETS, TOTAL_EQUITY, TOTAL_DEBT,
        SHARES_DILUTED, TOTAL_LIABILITIES,
        NET_CASH_OPS, CAPEX,
    )
    # simfin>=1.0 has no FREE_CASH_FLOW constant, and 'Free Cash Flow' is a
    # column of the *derived* dataset, not of the raw cash-flow statement this
    # module downloads. Importing it unconditionally made the whole block raise
    # ImportError, silently setting _SIMFIN_AVAILABLE=False and surfacing later
    # as a misleading "simfin is not installed". Derive it instead:
    #   FCF = Net Cash from Operating Activities + Change in Fixed Assets
    # (CAPEX is reported negative, so this is a sum, not a difference.)
    FREE_CASH_FLOW = getattr(__import__("simfin.names", fromlist=["FCF"]), "FCF",
                             "Free Cash Flow")
    _SIMFIN_AVAILABLE = True
except ImportError:
    _SIMFIN_AVAILABLE = False

from data.fundamental_store import save_fundamentals, get_fundamentals_as_of as _store_get

# SimFin column → our canonical store column mapping
_COLUMN_MAP = {
    TICKER if _SIMFIN_AVAILABLE else "Ticker": "ticker",
    PUBLISH_DATE if _SIMFIN_AVAILABLE else "Publish Date": "filing_date",
    REPORT_DATE if _SIMFIN_AVAILABLE else "Report Date": "period_end_date",
    REVENUE if _SIMFIN_AVAILABLE else "Revenue": "revenue",
    NET_INCOME if _SIMFIN_AVAILABLE else "Net Income": "net_income",
    EPS_DILUTED if _SIMFIN_AVAILABLE else "EPS Diluted": "eps_diluted",
    GROSS_PROFIT if _SIMFIN_AVAILABLE else "Gross Profit": "gross_profit",
    EBITDA if _SIMFIN_AVAILABLE else "EBITDA": "ebitda",
    TOTAL_ASSETS if _SIMFIN_AVAILABLE else "Total Assets": "total_assets",
    TOTAL_EQUITY if _SIMFIN_AVAILABLE else "Total Equity": "total_equity",
    TOTAL_DEBT if _SIMFIN_AVAILABLE else "Total Debt": "total_debt",
    FREE_CASH_FLOW if _SIMFIN_AVAILABLE else "Free Cash Flow": "free_cash_flow",
    # Needed to turn per-share figures into market-cap-relative ones
    # (book-to-price, FCF yield) at a given price.
    SHARES_DILUTED if _SIMFIN_AVAILABLE else "Shares (Diluted)": "shares_diluted",
}

_FISCAL_PERIOD_COL = FISCAL_PERIOD if _SIMFIN_AVAILABLE else "Fiscal Period"

# SimFin is NOT consistent about how it labels full-year rows across statement
# types: in the us-income-annual dataset Fiscal Period is 'FY', but in
# us-balance-annual and us-cashflow-annual it is 'Q4' (the balance sheet is a
# point-in-time snapshot taken at the fiscal year end). Filtering everything on
# 'FY' silently reduced the balance sheet to ZERO rows, so every balance-sheet
# field merged in as NaN and ROE / debt-to-equity were unusable — with no error
# raised anywhere. Accept both labels. (We already request variant='annual', so
# this filter is only guarding against stray rows.)
_ANNUAL_PERIOD = "FY"
_ANNUAL_PERIODS = ("FY", "Q4")


def _require_simfin() -> None:
    if not _SIMFIN_AVAILABLE:
        raise ImportError("simfin is not installed. Run: pip install simfin")


def _setup(api_key: Optional[str] = None, data_dir: Optional[Path] = None) -> None:
    """Configure SimFin API key and local data directory."""
    _require_simfin()
    if api_key is None:
        from config import SIMFIN_API_KEY
        api_key = SIMFIN_API_KEY
    if not api_key:
        raise ValueError(
            "SimFin API key is not set. "
            "Set SIMFIN_API_KEY in your .env or environment. "
            "Free key available at https://simfin.com/api/v2/"
        )
    if data_dir is None:
        from config import CACHE_DIR
        data_dir = CACHE_DIR / "simfin_raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    sf.set_api_key(api_key)
    sf.set_data_dir(str(data_dir))


# ---------------------------------------------------------------------------
# Bulk download and normalisation
# ---------------------------------------------------------------------------

def download_bulk(
    api_key: Optional[str] = None,
    data_dir: Optional[Path] = None,
    refresh_days: int = 7,
) -> None:
    """
    Download SimFin bulk data for US equities (income, balance, cashflow).

    SimFin caches the raw files locally — subsequent calls within refresh_days
    will use the cache rather than re-downloading.

    Parameters
    ----------
    api_key : str, optional
        SimFin API key. Defaults to config.SIMFIN_API_KEY.
    data_dir : Path, optional
        Local cache directory for SimFin raw files. Defaults to
        config.CACHE_DIR / 'simfin_raw'.
    refresh_days : int
        Minimum days between re-downloads. SimFin's own refresh logic applies.
    """
    _setup(api_key, data_dir)
    for variant in ("annual",):  # add 'quarterly' with paid plan
        sf.load_income(variant=variant, market="us", refresh_days=refresh_days)
        sf.load_balance(variant=variant, market="us", refresh_days=refresh_days)
        sf.load_cashflow(variant=variant, market="us", refresh_days=refresh_days)


def build_fundamental_store(
    api_key: Optional[str] = None,
    data_dir: Optional[Path] = None,
    store_dir: Optional[Path] = None,
    tickers: Optional[List[str]] = None,
    refresh_days: int = 7,
) -> pd.DataFrame:
    """
    Parse SimFin bulk data and write to the fundamental store (Parquet).

    This merges income, balance, and cash flow statements on
    (ticker, filing_date, period_end_date), then saves to fundamental_store
    with source='simfin'.

    Parameters
    ----------
    api_key : str, optional
    data_dir : Path, optional
        SimFin raw cache. Defaults to config.CACHE_DIR / 'simfin_raw'.
    store_dir : Path, optional
        Fundamental store directory. Defaults to config.CACHE_DIR / 'fundamentals'.
    tickers : list of str, optional
        Restrict to these tickers. If None, processes all US equities.
    refresh_days : int
        Passed to SimFin download.

    Returns
    -------
    pd.DataFrame
        The normalised DataFrame that was written to the store.
    """
    _setup(api_key, data_dir)

    income = sf.load_income(variant="annual", market="us", refresh_days=refresh_days)
    balance = sf.load_balance(variant="annual", market="us", refresh_days=refresh_days)
    cashflow = sf.load_cashflow(variant="annual", market="us", refresh_days=refresh_days)

    # SimFin returns multi-index (Ticker, Report Date) — reset for easier merging
    income = income.reset_index()
    balance = balance.reset_index()
    cashflow = cashflow.reset_index()

    # Filter to annual periods only. See _ANNUAL_PERIODS: income uses 'FY',
    # balance/cashflow use 'Q4' for the same full-year filing.
    for _df_name, _df in (("income", income), ("balance", balance), ("cashflow", cashflow)):
        if _FISCAL_PERIOD_COL in _df.columns:
            kept = _df[_df[_FISCAL_PERIOD_COL].isin(_ANNUAL_PERIODS)]
            if kept.empty and not _df.empty:
                warnings.warn(
                    f"SimFin {_df_name}: fiscal-period filter removed every row "
                    f"(labels present: {sorted(set(_df[_FISCAL_PERIOD_COL].dropna().unique()))[:5]}). "
                    "Using unfiltered data."
                )
                kept = _df
            if _df_name == "income":
                income = kept
            elif _df_name == "balance":
                balance = kept
            else:
                cashflow = kept

    ticker_col = TICKER if _SIMFIN_AVAILABLE else "Ticker"
    publish_col = PUBLISH_DATE if _SIMFIN_AVAILABLE else "Publish Date"
    report_col = REPORT_DATE if _SIMFIN_AVAILABLE else "Report Date"

    merge_keys = [ticker_col, publish_col, report_col]

    # Select only the columns we need before merging
    def _select(df: pd.DataFrame, extra_cols: List[str]) -> pd.DataFrame:
        keep = [c for c in merge_keys + extra_cols if c in df.columns]
        return df[keep]

    # SimFin's free annual statements do not ship EPS-diluted, EBITDA or
    # Total Debt as columns (they live in the paid 'Derived Figures' dataset).
    # Derive them from raw line items that ARE present, so quality/value factors
    # are not silently dropped.
    _shares = "Shares (Diluted)"
    _op_inc = "Operating Income (Loss)"
    _dep_amort = "Depreciation & Amortization"
    _ni = NET_INCOME if _SIMFIN_AVAILABLE else "Net Income"
    _eps = EPS_DILUTED if _SIMFIN_AVAILABLE else "EPS Diluted"
    _ebitda = EBITDA if _SIMFIN_AVAILABLE else "EBITDA"

    if _eps not in income.columns and {_ni, _shares} <= set(income.columns):
        income = income.copy()
        income[_eps] = income[_ni] / income[_shares].replace(0, pd.NA)
    if _ebitda not in income.columns and {_op_inc, _dep_amort} <= set(income.columns):
        income = income.copy() if _eps in income.columns else income
        # D&A is reported negative in SimFin's income statement; EBITDA adds it back.
        income[_ebitda] = income[_op_inc] - income[_dep_amort]

    _st_debt, _lt_debt = "Short Term Debt", "Long Term Debt"
    _td = TOTAL_DEBT if _SIMFIN_AVAILABLE else "Total Debt"
    if _td not in balance.columns and {_st_debt, _lt_debt} <= set(balance.columns):
        balance = balance.copy()
        balance[_td] = (balance[_st_debt].fillna(0) + balance[_lt_debt].fillna(0))

    income_cols = [
        REVENUE if _SIMFIN_AVAILABLE else "Revenue",
        _ni, _eps, _shares,
        GROSS_PROFIT if _SIMFIN_AVAILABLE else "Gross Profit",
        _ebitda,
    ]
    balance_cols = [
        TOTAL_ASSETS if _SIMFIN_AVAILABLE else "Total Assets",
        TOTAL_EQUITY if _SIMFIN_AVAILABLE else "Total Equity",
        _td,
    ]
    # The raw cash-flow statement has no 'Free Cash Flow' column (that lives in
    # SimFin's *derived* dataset). Take it if present, otherwise derive it from
    # the two raw components below.
    _ops = NET_CASH_OPS if _SIMFIN_AVAILABLE else "Net Cash from Operating Activities"
    _capex = CAPEX if _SIMFIN_AVAILABLE else "Change in Fixed Assets & Intangibles"
    _fcf = FREE_CASH_FLOW if _SIMFIN_AVAILABLE else "Free Cash Flow"
    cashflow_cols = [_fcf, _ops, _capex]

    inc = _select(income, income_cols)
    bal = _select(balance, balance_cols)
    cf = _select(cashflow, cashflow_cols)

    if _fcf not in cf.columns and {_ops, _capex} <= set(cf.columns):
        # CAPEX is reported as a negative number, so this is a sum.
        cf[_fcf] = cf[_ops] + cf[_capex]
    cf = cf[[c for c in cf.columns if c not in (_ops, _capex)]]

    merged = inc.merge(bal, on=merge_keys, how="left")
    merged = merged.merge(cf, on=merge_keys, how="left")

    # Rename to canonical schema
    merged = merged.rename(columns=_COLUMN_MAP)

    # Add period_type column
    merged["period_type"] = "annual"

    # Optional ticker filter
    if tickers is not None:
        merged = merged[merged["ticker"].isin(tickers)]

    # Drop rows without a filing date (required for point-in-time safety)
    merged = merged.dropna(subset=["filing_date", "ticker"])
    merged = merged[merged["filing_date"].astype(str) != ""]

    if merged.empty:
        warnings.warn("SimFin build_fundamental_store: resulting DataFrame is empty.")
        return merged

    save_fundamentals(merged, source="simfin", store_dir=store_dir, merge=True)
    return merged


# ---------------------------------------------------------------------------
# Point-in-time query
# ---------------------------------------------------------------------------

def get_fundamentals_as_of(
    ticker: str,
    as_of: date,
    close_price: Optional[float] = None,
    store_dir: Optional[Path] = None,
    max_staleness_days: int = 180,
) -> Optional[Dict[str, Optional[float]]]:
    """
    Return point-in-time fundamental metrics for one ticker and rebalance date.

    Uses only filings where filing_date <= as_of, so no future information
    leaks into the feature vector.

    Parameters
    ----------
    ticker : str
    as_of : date
        Rebalance date.
    close_price : float, optional
        Point-in-time closing price at as_of. Required to compute pe_ratio,
        ev_ebitda, and fcf_yield. If None, those metrics are returned as None.
    store_dir : Path, optional
        Fundamental store directory.
    max_staleness_days : int
        If best filing is older than this, return None for all metrics.

    Returns
    -------
    dict[str, float | None] or None
        Keys match FUNDAMENTAL_KEYS from data.fundamentals. Returns None if
        no eligible filing exists for this ticker and date.
    """
    record = _store_get(
        ticker=ticker,
        as_of=as_of,
        source="simfin",
        period_type="annual",
        store_dir=store_dir,
        max_staleness_days=max_staleness_days,
    )
    if record is None:
        return None

    def _f(col: str) -> Optional[float]:
        val = record.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    revenue = _f("revenue")
    net_income = _f("net_income")
    eps_diluted = _f("eps_diluted")
    gross_profit = _f("gross_profit")
    ebitda = _f("ebitda")
    total_equity = _f("total_equity")
    total_debt = _f("total_debt")
    free_cash_flow = _f("free_cash_flow")
    total_assets = _f("total_assets")

    # ---- Derived metrics that don't need price ----
    roe = (net_income / total_equity) if (net_income is not None and total_equity and total_equity != 0) else None
    gross_margin = (gross_profit / revenue) if (gross_profit is not None and revenue and revenue != 0) else None
    debt_to_equity = (total_debt / total_equity) if (total_debt is not None and total_equity and total_equity != 0) else None

    # Revenue growth: compare with prior year's record
    prior_record = _store_get(
        ticker=ticker,
        as_of=date(as_of.year - 1, as_of.month, as_of.day),
        source="simfin",
        period_type="annual",
        store_dir=store_dir,
        max_staleness_days=max_staleness_days + 365,
    )
    prior_revenue = None
    prior_eps = None
    if prior_record is not None:
        v = prior_record.get("revenue")
        prior_revenue = float(v) if v is not None and not pd.isna(v) else None
        v = prior_record.get("eps_diluted")
        prior_eps = float(v) if v is not None and not pd.isna(v) else None

    revenue_growth = (
        (revenue - prior_revenue) / abs(prior_revenue)
        if (revenue is not None and prior_revenue is not None and prior_revenue != 0)
        else None
    )
    eps_growth = (
        (eps_diluted - prior_eps) / abs(prior_eps)
        if (eps_diluted is not None and prior_eps is not None and prior_eps != 0)
        else None
    )

    # Accruals ratio = (Net Income - FCF) / Total Assets (Sloan accrual)
    accruals_ratio = (
        (net_income - free_cash_flow) / total_assets
        if (net_income is not None and free_cash_flow is not None
            and total_assets is not None and total_assets != 0)
        else None
    )

    # ---- Price-dependent metrics ----
    pe_ratio = None
    ev_ebitda = None
    fcf_yield = None

    if close_price is not None and close_price > 0:
        if eps_diluted is not None and eps_diluted != 0:
            pe_ratio = close_price / eps_diluted

        # Approximate market cap from price and shares (SimFin sometimes provides shares)
        # We use price * shares_diluted if available; otherwise skip EV metrics
        shares = _f("shares_diluted") if "shares_diluted" in record.index else None
        if shares is not None:
            market_cap = close_price * shares
            if ebitda is not None and ebitda != 0 and total_debt is not None:
                ev = market_cap + total_debt  # simplified: ignores cash
                ev_ebitda = ev / ebitda
            if free_cash_flow is not None and market_cap > 0:
                fcf_yield = free_cash_flow / market_cap

    return {
        "pe_ratio": pe_ratio,
        "ev_ebitda": ev_ebitda,
        "fcf_yield": fcf_yield,
        "roe": roe,
        "gross_margin": gross_margin,
        "debt_to_equity": debt_to_equity,
        "revenue_growth": revenue_growth,
        "eps_growth": eps_growth,
        "accruals_ratio": accruals_ratio,
        "earnings_consistency": None,  # requires per-quarter data; available with paid SimFin plan
    }


def get_fundamentals_batch_as_of(
    tickers: List[str],
    as_of: date,
    prices: Optional[Dict[str, float]] = None,
    store_dir: Optional[Path] = None,
    max_staleness_days: int = 180,
    verbose: bool = False,
) -> Dict[str, Optional[Dict[str, Optional[float]]]]:
    """
    Batch point-in-time fundamentals for a list of tickers.

    Parameters
    ----------
    tickers : list of str
    as_of : date
        Rebalance date.
    prices : dict, optional
        {ticker: close_price} at as_of. Used for pe_ratio, ev_ebitda, fcf_yield.
    store_dir : Path, optional
    max_staleness_days : int
    verbose : bool

    Returns
    -------
    dict[str, dict | None]
        ticker → metrics dict or None if no data available.
    """
    out: Dict[str, Optional[Dict]] = {}
    for i, ticker in enumerate(tickers):
        if verbose and (i + 1) % 50 == 0:
            print(f"[simfin] {i + 1}/{len(tickers)}")
        close_price = prices.get(ticker) if prices else None
        out[ticker] = get_fundamentals_as_of(
            ticker=ticker,
            as_of=as_of,
            close_price=close_price,
            store_dir=store_dir,
            max_staleness_days=max_staleness_days,
        )
    return out
