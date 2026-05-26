"""
src/cross_section.py
--------------------
Cross-sectional analysis of ETF reconstitution price effects.

Data conventions
----------------
per_df : pd.DataFrame
    Long format.  columns: date (DatetimeIndex), stock_id, dividend_yield,
    PER, PBR.  Source: FinMind TaiwanStockPER.
    Build with::

        fetch_stock_per(sid, start, end)      # see below
        pd.concat([...]).sort_index()

shares_df : pd.DataFrame
    Long format.  columns: date (DatetimeIndex), stock_id, shares_issued.
    Source: TaiwanStockDividend → ParticipateDistributionOfTotalShares.
    Build with::

        fetch_shares_outstanding(sid, start, end)   # see below
        pd.concat([...]).sort_index()

prices_wide, turnover_wide, market_returns : same as event_study.py.

Look-ahead policy
-----------------
Every feature uses data strictly *before* ref_date (< ref_date).
This is enforced by filtering with ``df.index < ref_date`` in all lookups.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from _plot_config import apply_chinese_style
apply_chinese_style()
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from event_study import compute_event_car

logger = logging.getLogger(__name__)

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
_INTER_CALL_SLEEP = 0.5  # seconds between FinMind calls

# ── FinMind fetch helpers (cross-section specific) ─────────────────────────────


@retry(
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError, requests.Timeout)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fm_get(params: dict) -> list[dict]:
    """Single FinMind GET, returns data list or [] on empty."""
    from dotenv import load_dotenv
    import os
    load_dotenv()
    token = os.getenv("FINMIND_TOKEN", "")
    if token:
        params = {**params, "token": token}
    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") not in (200, None):
        raise RuntimeError(f"FinMind error {payload.get('status')}: {payload.get('msg')}")
    return payload.get("data", [])


def fetch_stock_per(
    stock_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily PER / PBR / dividend_yield from TaiwanStockPER.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex("date"), columns: stock_id, dividend_yield, PER, PBR.
        dividend_yield is in percent (e.g. 3.5 means 3.5%).
    """
    rows = _fm_get({"dataset": "TaiwanStockPER", "data_id": stock_id,
                    "start_date": start_date, "end_date": end_date})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for col in ["dividend_yield", "PER", "PBR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_shares_outstanding(
    stock_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Derive shares outstanding from TaiwanStockDividend.

    FinMind does not have a dedicated shares-outstanding dataset for free-tier
    users.  We proxy it via ParticipateDistributionOfTotalShares (= total shares
    eligible for each dividend distribution).  This is updated quarterly and
    gives a good approximation of float shares.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex("date" = AnnouncementDate), columns: stock_id, shares_issued.
        Use the most-recent row before ref_date in downstream functions.
    """
    rows = _fm_get({"dataset": "TaiwanStockDividend", "data_id": stock_id,
                    "start_date": start_date, "end_date": end_date})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Use announcement date as the date of record
    df["date"] = pd.to_datetime(df["AnnouncementDate"], errors="coerce")
    df["shares_issued"] = pd.to_numeric(df["ParticipateDistributionOfTotalShares"], errors="coerce")
    df["stock_id"] = stock_id
    df = df[["date", "stock_id", "shares_issued"]].dropna(subset=["date", "shares_issued"])
    df = df.set_index("date").sort_index()
    return df


def fetch_per_multiple(
    stock_ids: list[str],
    start_date: str,
    end_date: str,
    sleep_seconds: float = _INTER_CALL_SLEEP,
) -> pd.DataFrame:
    """Batch-fetch PER data for multiple stocks, respecting rate limits.

    Returns
    -------
    pd.DataFrame
        Long format, DatetimeIndex("date"), columns: stock_id, dividend_yield, PER, PBR.
    """
    frames = []
    for i, sid in enumerate(stock_ids, 1):
        logger.info("Fetching PER %s (%d/%d) …", sid, i, len(stock_ids))
        df = fetch_stock_per(sid, start_date, end_date)
        if not df.empty:
            df["stock_id"] = sid
            frames.append(df)
        if i < len(stock_ids):
            time.sleep(sleep_seconds)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def fetch_shares_multiple(
    stock_ids: list[str],
    start_date: str,
    end_date: str,
    sleep_seconds: float = _INTER_CALL_SLEEP,
) -> pd.DataFrame:
    """Batch-fetch shares-outstanding proxy for multiple stocks."""
    frames = []
    for i, sid in enumerate(stock_ids, 1):
        logger.info("Fetching shares %s (%d/%d) …", sid, i, len(stock_ids))
        df = fetch_shares_outstanding(sid, start_date, end_date)
        if not df.empty:
            frames.append(df)
        if i < len(stock_ids):
            time.sleep(sleep_seconds)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


# ── Private helpers ────────────────────────────────────────────────────────────


def _prior_value(
    long_df: pd.DataFrame,
    stock_id: str,
    ref_date: pd.Timestamp,
    value_col: str,
) -> float:
    """Return the most-recent non-null value of ``value_col`` strictly before ``ref_date``.

    long_df must have a DatetimeIndex and a stock_id column.
    Returns np.nan if no such value exists.
    """
    mask = (long_df.index < ref_date) & (long_df["stock_id"] == stock_id)
    subset = long_df.loc[mask, value_col].dropna()
    return float(subset.iloc[-1]) if not subset.empty else np.nan


def _winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip series at given quantiles. NaNs are preserved."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


# ── Feature computation ────────────────────────────────────────────────────────


def compute_stock_features(
    stock_id: str,
    ref_date: pd.Timestamp | str,
    prices_wide: pd.DataFrame,
    turnover_wide: Optional[pd.DataFrame] = None,
    shares_df: Optional[pd.DataFrame] = None,
    per_df: Optional[pd.DataFrame] = None,
) -> dict:
    """Compute pre-announcement features for a single stock.

    All data accesses are strictly before ref_date (no look-ahead bias).

    Parameters
    ----------
    stock_id : str
    ref_date : Timestamp or str
        Announcement date = event day 0.  Features use data before this date.
    prices_wide : pd.DataFrame
        Wide close-price DataFrame (DatetimeIndex).
    turnover_wide : pd.DataFrame, optional
        Wide daily 成交金額 DataFrame.  If None, avg_volume_60d = NaN.
    shares_df : pd.DataFrame, optional
        Long format, DatetimeIndex, columns [stock_id, shares_issued].
        Use fetch_shares_outstanding() to build.  If None, market_cap = NaN.
    per_df : pd.DataFrame, optional
        Long format, DatetimeIndex, columns [stock_id, dividend_yield, PER, PBR].
        Use fetch_stock_per() to build.  If None, dividend_yield = NaN.

    Returns
    -------
    dict
        Keys: stock_id, ref_date, price_prev, market_cap, avg_volume_60d,
              dividend_yield, log_market_cap, log_avg_volume.
    """
    ref_date = pd.Timestamp(ref_date)
    trading_days = prices_wide.index
    base = {"stock_id": stock_id, "ref_date": ref_date}

    # ── Previous trading day close price (reference for all calculations) ──
    pre_days = trading_days[trading_days < ref_date]
    if pre_days.empty or stock_id not in prices_wide.columns:
        return {
            **base,
            "price_prev": np.nan, "market_cap": np.nan,
            "avg_volume_60d": np.nan, "dividend_yield": np.nan,
            "log_market_cap": np.nan, "log_avg_volume": np.nan,
        }

    prev_day = pre_days[-1]
    price = float(prices_wide.loc[prev_day, stock_id])
    if pd.isna(price) or price <= 0:
        price = np.nan
    base["price_prev"] = price

    # ── Market cap ──
    shares = (
        _prior_value(shares_df, stock_id, ref_date, "shares_issued")
        if shares_df is not None else np.nan
    )
    base["market_cap"] = (
        price * shares
        if (not pd.isna(price) and not pd.isna(shares)) else np.nan
    )

    # ── Average daily turnover (60 trading days strictly before ref_date) ──
    if turnover_wide is not None and stock_id in turnover_wide.columns:
        window_days = pre_days[-60:]
        avg_vol = float(turnover_wide.loc[window_days, stock_id].mean())
        base["avg_volume_60d"] = avg_vol if not pd.isna(avg_vol) else np.nan
    else:
        base["avg_volume_60d"] = np.nan

    # ── Dividend yield (most recent value before ref_date, from TaiwanStockPER) ──
    base["dividend_yield"] = (
        _prior_value(per_df, stock_id, ref_date, "dividend_yield")
        if per_df is not None else np.nan
    )

    # ── Log transforms ──
    mc = base["market_cap"]
    av = base["avg_volume_60d"]
    base["log_market_cap"] = float(np.log(mc)) if (not pd.isna(mc) and mc > 0) else np.nan
    base["log_avg_volume"] = float(np.log(av)) if (not pd.isna(av) and av > 0) else np.nan

    return base


def compute_impact_ratio(
    target_weight: float,
    etf_aum: float,
    market_cap: float,
) -> float:
    """Estimated ETF buying as fraction of target stock's market cap.

    impact_ratio = target_weight × etf_aum / market_cap

    A higher value means the ETF will absorb a larger fraction of the stock's
    float, implying greater price pressure around the rebalancing.

    Parameters
    ----------
    target_weight : float
        Stock's target weight in the ETF (e.g. 0.03 for 3 %).
    etf_aum : float
        ETF assets under management in TWD.
    market_cap : float
        Stock market cap in TWD.

    Returns
    -------
    float, or nan if any input is missing or market_cap ≤ 0.
    """
    if any(pd.isna(v) for v in [target_weight, etf_aum, market_cap]):
        return np.nan
    if market_cap <= 0:
        return np.nan
    return float(target_weight * etf_aum / market_cap)


# ── Dataset builder ────────────────────────────────────────────────────────────


def build_cross_section_dataset(
    events: pd.DataFrame,
    prices_wide: pd.DataFrame,
    market_returns: pd.Series,
    turnover_wide: Optional[pd.DataFrame] = None,
    shares_df: Optional[pd.DataFrame] = None,
    per_df: Optional[pd.DataFrame] = None,
    winsorize_impact: bool = True,
    stock_col: str = "added_stocks",
) -> pd.DataFrame:
    """Build one row per (event, stock) with CAR and pre-event features.

    CAR is computed from announcement day (relative_day=0) to effective day
    using the market-adjusted method from event_study.compute_event_car.

    Parameters
    ----------
    events : pd.DataFrame
        From load_events().  Expected columns: event_id, announcement_date,
        effective_date, added_stocks (or removed_stocks), and optionally
        target_weight (float, uniform across all added stocks) and etf_aum (float).
    prices_wide : pd.DataFrame
    market_returns : pd.Series
    turnover_wide : pd.DataFrame, optional
    shares_df : pd.DataFrame, optional
    per_df : pd.DataFrame, optional
    winsorize_impact : bool
        Winsorize impact_ratio at 1 % / 99 % quantiles.  Keeps the raw value
        in impact_ratio_raw for diagnostics.
    stock_col : str
        Column in events with the list of stock tickers to analyse.

    Returns
    -------
    pd.DataFrame
        Columns: event_id, etf_code, announcement_date, stock_id,
                 CAR_ann_to_eff, market_cap, avg_volume_60d, dividend_yield,
                 log_market_cap, log_avg_volume, target_weight, etf_aum,
                 impact_ratio, log_impact_ratio.
    """
    trading_days = prices_wide.index
    records = []

    for _, event in events.iterrows():
        tickers: list[str] = event[stock_col]
        if not tickers:
            continue

        ann_date = pd.Timestamp(event["announcement_date"])
        eff_date = pd.Timestamp(event["effective_date"])

        # Resolve to actual trading days
        ann_future = trading_days[trading_days >= ann_date]
        eff_future = trading_days[trading_days >= eff_date]
        if ann_future.empty or eff_future.empty:
            logger.warning("Event %s: date out of range — skipping", event.get("event_id"))
            continue

        ann_idx = trading_days.get_loc(ann_future[0])
        eff_idx = trading_days.get_loc(eff_future[0])
        eff_rel = eff_idx - ann_idx

        if eff_rel < 0:
            logger.warning("Event %s: effective_date < announcement_date — skipping", event.get("event_id"))
            continue

        etf_aum = pd.to_numeric(event.get("etf_aum", np.nan), errors="coerce")
        target_weight_raw = event.get("target_weight", np.nan)

        for stock_id in tickers:
            if stock_id not in prices_wide.columns:
                logger.debug("skip %s: not in prices_wide", stock_id)
                continue

            # ── CAR: announcement → effective ──
            try:
                car_df = compute_event_car(
                    stock_id=stock_id,
                    event_announcement_date=ann_date,
                    prices_wide=prices_wide,
                    market_returns=market_returns,
                    event_window=(0, eff_rel),
                )
                car_val = float(car_df["CAR"].iloc[-1]) if not car_df.empty else np.nan
            except Exception as exc:
                logger.warning("CAR error %s / event %s: %s", stock_id, event.get("event_id"), exc)
                car_val = np.nan

            # ── Features ──
            feat = compute_stock_features(
                stock_id=stock_id,
                ref_date=ann_date,
                prices_wide=prices_wide,
                turnover_wide=turnover_wide,
                shares_df=shares_df,
                per_df=per_df,
            )

            # ── Impact ratio ──
            tw = (
                pd.to_numeric(target_weight_raw.get(stock_id, np.nan), errors="coerce")
                if isinstance(target_weight_raw, dict)
                else pd.to_numeric(target_weight_raw, errors="coerce")
            )
            impact = compute_impact_ratio(tw, etf_aum, feat["market_cap"])

            records.append({
                "event_id":           event.get("event_id"),
                "etf_code":           event.get("etf_code"),
                "announcement_date":  ann_date,
                "stock_id":           stock_id,
                "CAR_ann_to_eff":     car_val,
                "market_cap":         feat["market_cap"],
                "avg_volume_60d":     feat["avg_volume_60d"],
                "dividend_yield":     feat["dividend_yield"],
                "log_market_cap":     feat["log_market_cap"],
                "log_avg_volume":     feat["log_avg_volume"],
                "target_weight":      float(tw) if not pd.isna(tw) else np.nan,
                "etf_aum":            float(etf_aum) if not pd.isna(etf_aum) else np.nan,
                "impact_ratio":       impact,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).reset_index(drop=True)

    # ── Log impact ratio ──
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_impact_ratio"] = np.where(df["impact_ratio"] > 0,
                                           np.log(df["impact_ratio"]), np.nan)

    # ── Winsorize (only if enough non-null observations) ──
    valid_impact = df["impact_ratio"].notna().sum()
    if winsorize_impact and valid_impact >= 10:
        df["impact_ratio_raw"] = df["impact_ratio"].copy()
        df["impact_ratio"] = _winsorize(df["impact_ratio"])
        df["log_impact_ratio"] = np.where(df["impact_ratio"] > 0,
                                           np.log(df["impact_ratio"]), np.nan)
        n_clipped = (df["impact_ratio"] != df["impact_ratio_raw"]).sum()
        if n_clipped:
            logger.info("Winsorized %d impact_ratio values (1%%/99%% quantiles)", n_clipped)
    elif winsorize_impact and valid_impact > 0:
        logger.warning("Too few impact_ratio obs (%d) for winsorize — skipped", valid_impact)

    return df


# ── Regression ─────────────────────────────────────────────────────────────────


def regression_analysis(
    cross_section_df: pd.DataFrame,
    save_path: str | Path = "output/tables/regression_results.txt",
    dep_var: str = "CAR_ann_to_eff",
    predictors: Optional[list[str]] = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS regression of CAR on stock characteristics.

    Default model: CAR ~ log_market_cap + log_impact_ratio + dividend_yield

    Uses HC3 heteroskedasticity-robust standard errors (standard in
    cross-sectional finance regressions).

    Parameters
    ----------
    cross_section_df : pd.DataFrame
        Output of build_cross_section_dataset().
    save_path : str or Path
        Path to save the formatted result table.
    dep_var : str
        Dependent variable column name.
    predictors : list[str], optional
        Override the default predictor set.

    Returns
    -------
    statsmodels RegressionResultsWrapper (HC3-adjusted).
    """
    if predictors is None:
        predictors = ["log_market_cap", "log_impact_ratio", "dividend_yield"]

    cols = [dep_var] + predictors
    df = cross_section_df[cols].dropna()
    n = len(df)

    if n < 3:
        raise ValueError(f"Only {n} complete observations — cannot run regression.")

    small_sample = n < 20

    y = df[dep_var]
    X = sm.add_constant(df[predictors], has_constant="add")

    model = sm.OLS(y, X).fit(cov_type="HC3")

    # ── VIF ──
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_vals = [variance_inflation_factor(X.values, i + 1) for i in range(len(predictors))]
        vif_df = pd.DataFrame({"variable": predictors, "VIF": vif_vals})
    except Exception:
        vif_df = None

    # ── Format output ──
    sig_map = lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

    coef_table = pd.DataFrame({
        "coef":    model.params[1:],
        "std_err": model.bse[1:],
        "t":       model.tvalues[1:],
        "P>|t|":   model.pvalues[1:],
        "sig":     [sig_map(p) for p in model.pvalues[1:]],
    })

    sep = "─" * 68
    lines = [
        sep,
        "  OLS: CAR (announcement → effective date)",
        sep,
        f"  N = {n}   Adj. R² = {model.rsquared_adj:.4f}   F-stat p = {model.f_pvalue:.4f}",
        "",
    ]
    if small_sample:
        lines.append(f"  ⚠  N = {n} < 20  —  small sample, interpret with caution\n")

    lines += [
        "  Robust standard errors (HC3)\n",
        coef_table.to_string(float_format="{:.6f}".format),
        "",
        "  Significance: * p<.10  ** p<.05  *** p<.01",
    ]

    if vif_df is not None:
        lines += [
            "",
            "  Variance Inflation Factors",
            vif_df.to_string(index=False, float_format="{:.2f}".format),
            "  (VIF > 10 → multicollinearity concern)",
        ]

    lines.append(sep)
    out_text = "\n".join(lines)
    print(out_text)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(out_text, encoding="utf-8")
        logger.info("Regression table saved → %s", save_path)

    return model


# ── Group analysis ─────────────────────────────────────────────────────────────


def group_analysis(
    cross_section_df: pd.DataFrame,
    group_var: str = "impact_ratio",
    n_groups: int = 3,
    dep_var: str = "CAR_ann_to_eff",
    save_path: str | Path = "output/figures/group_car_by_impact.png",
) -> pd.DataFrame:
    """Sort stocks into equal-frequency groups by group_var; plot mean CAR.

    Parameters
    ----------
    cross_section_df : pd.DataFrame
        Output of build_cross_section_dataset().
    group_var : str
        Column to sort / group by.
    n_groups : int
        Number of quantile groups (default 3 = terciles).
    dep_var : str
        Dependent variable to summarise.
    save_path : str or Path
        Destination PNG.

    Returns
    -------
    pd.DataFrame
        Summary table: group, mean_CAR, std_CAR, N, se, ci95.
    """
    df = cross_section_df[[group_var, dep_var]].dropna().copy()
    if len(df) < n_groups * 2:
        raise ValueError(
            f"Too few observations ({len(df)}) for {n_groups} groups. "
            "Need at least 2 × n_groups."
        )

    _default_labels = {2: ["Low", "High"], 3: ["Low", "Mid", "High"],
                       4: ["Q1", "Q2", "Q3", "Q4"]}
    labels = _default_labels.get(n_groups, [f"G{i+1}" for i in range(n_groups)])

    df["group"] = pd.qcut(df[group_var], q=n_groups, labels=labels)

    summary = (
        df.groupby("group", observed=True)[dep_var]
        .agg(mean_CAR="mean", std_CAR="std", N="count")
        .reset_index()
    )
    summary["se"]   = summary["std_CAR"] / np.sqrt(summary["N"])
    summary["ci95"] = 1.96 * summary["se"]

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = ["#5b9bd5", "#70ad47", "#ed7d31", "#ffc000"][:n_groups]
    x = np.arange(n_groups)

    bars = ax.bar(
        x, summary["mean_CAR"],
        yerr=summary["ci95"],
        color=palette,
        capsize=5, width=0.5,
        error_kw={"elinewidth": 1.3, "ecolor": "#333333"},
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["group"].astype(str), fontsize=11)
    ax.set_xlabel(f"{group_var} 分組（等頻率三分位）", fontsize=11)
    ax.set_ylabel("平均累積異常報酬（CAR）", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_title(
        f"按 {group_var} 分組的平均 CAR\n（公告日 → 生效日）",
        fontsize=12, fontweight="bold",
    )

    # N annotation and bar-top labels
    y_min = ax.get_ylim()[0]
    for i, row in summary.iterrows():
        bar_top = row["mean_CAR"] + (row["ci95"] if row["mean_CAR"] >= 0 else -row["ci95"])
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
        va = "bottom" if row["mean_CAR"] >= 0 else "top"
        ax.text(i, bar_top + offset * (1 if row["mean_CAR"] >= 0 else -1),
                f"{row['mean_CAR']:.2%}", ha="center", va=va, fontsize=9, fontweight="bold")
        ax.text(i, y_min + (ax.get_ylim()[1] - y_min) * 0.02,
                f"N={int(row['N'])}", ha="center", va="bottom", fontsize=8, color="gray")

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("Group chart saved → %s", save_path)
    plt.close(fig)

    print("\nGroup summary:")
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    return summary
