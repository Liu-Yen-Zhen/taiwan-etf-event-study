"""
src/event_study.py
------------------
Event study toolkit for ETF index reconstitution research.

All functions work in **trading-day space**. Calendar days with no price data
(weekends, holidays, halt days) are silently skipped at every step.

Data conventions expected by callers
-------------------------------------
prices_wide : pd.DataFrame
    DatetimeIndex (trading days), columns = stock_ids (e.g. "2330", "TAIEX"),
    values = daily closing price. Build with::

        prices_wide = (
            long_df                          # from stock_prices.parquet
            .reset_index()
            .pivot(index="date", columns="stock_id", values="close")
            .sort_index()
        )

market_returns : pd.Series
    DatetimeIndex, values = simple daily returns of the benchmark (TAIEX).
    Build with::

        market_returns = prices_wide["TAIEX"].pct_change()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from _plot_config import apply_chinese_style
apply_chinese_style()

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_t0(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> int:
    """Return integer position of the first trading day on-or-after `date`.

    Raises ValueError if no such day exists in trading_days.
    """
    candidates = trading_days[trading_days >= date]
    if candidates.empty:
        raise ValueError(f"No trading day on or after {date.date()}")
    return trading_days.get_loc(candidates[0])


def _simple_return(prices: pd.Series, day_idx: int, prev_idx: int) -> Optional[float]:
    """Return (P_t / P_{t-1}) - 1, or None if either price is missing / zero."""
    try:
        p1 = prices.iloc[day_idx]
        p0 = prices.iloc[prev_idx]
    except IndexError:
        return None
    if pd.isna(p1) or pd.isna(p0) or p0 == 0:
        return None
    return float(p1 / p0 - 1)


# ── Core functions ─────────────────────────────────────────────────────────────


def compute_abnormal_return(
    stock_returns: pd.Series,
    market_returns: pd.Series,
) -> pd.Series:
    """Market-adjusted abnormal return: AR_t = r_stock_t - r_market_t.

    Aligns both series on their shared index before subtracting.

    Parameters
    ----------
    stock_returns : pd.Series
        DatetimeIndex, simple daily returns for one stock.
    market_returns : pd.Series
        DatetimeIndex, simple daily returns for the benchmark.

    Returns
    -------
    pd.Series
        AR series on the intersection of both indices.
    """
    s, m = stock_returns.align(market_returns, join="inner")
    return s - m


def compute_event_car(
    stock_id: str,
    event_announcement_date: pd.Timestamp | str,
    prices_wide: pd.DataFrame,
    market_returns: pd.Series,
    event_window: tuple[int, int] = (-30, 60),
) -> pd.DataFrame:
    """Compute per-trading-day AR and CAR for one stock around one event.

    Parameters
    ----------
    stock_id : str
        Must be a column in ``prices_wide``.
    event_announcement_date : Timestamp or str
        The announcement date. If it falls on a non-trading day, the next
        trading day is used as day 0.
    prices_wide : pd.DataFrame
        Wide close-price DataFrame (see module docstring).
    market_returns : pd.Series
        Benchmark daily returns (see module docstring).
    event_window : tuple[int, int]
        (start_rel_day, end_rel_day) relative to day 0, inclusive.
        Negative values = pre-event. Default: (-30, 60).

    Returns
    -------
    pd.DataFrame
        Columns: relative_day (int), date (Timestamp), AR (float), CAR (float).
        Rows only exist for trading days with complete data; missing days are
        dropped rather than filled.
    """
    if stock_id not in prices_wide.columns:
        raise KeyError(f"stock_id '{stock_id}' not found in prices_wide")

    trading_days = prices_wide.index
    t0_idx = _resolve_t0(pd.Timestamp(event_announcement_date), trading_days)

    stock_prices = prices_wide[stock_id]
    records = []
    cumulative_ar = 0.0

    for rel_day in range(event_window[0], event_window[1] + 1):
        day_idx = t0_idx + rel_day
        prev_idx = day_idx - 1  # previous trading day (as integer position)

        if day_idx < 0 or day_idx >= len(trading_days):
            continue
        if prev_idx < 0:
            continue

        day = trading_days[day_idx]

        stock_ret = _simple_return(stock_prices, day_idx, prev_idx)
        mkt_ret = market_returns.get(day, np.nan)
        if stock_ret is None or pd.isna(mkt_ret):
            continue  # drop this day for this stock — do NOT drop the whole event

        ar = stock_ret - float(mkt_ret)
        cumulative_ar += ar
        records.append({"relative_day": rel_day, "date": day, "AR": ar, "CAR": cumulative_ar})

    return pd.DataFrame(records)


def aggregate_cars_across_events(
    events: pd.DataFrame,
    prices_wide: pd.DataFrame,
    market_returns: pd.Series,
    event_window: tuple[int, int] = (-30, 60),
    stock_col: str = "added_stocks",
) -> pd.DataFrame:
    """Aggregate AR and CAR across all events and stocks in ``stock_col``.

    For each (event, stock) pair, ``compute_event_car`` is called. Results are
    stacked and summarised by relative_day. Trading days with missing price or
    market data are dropped at the per-(day, stock, event) level; other days
    for the same pair are retained.

    Parameters
    ----------
    events : pd.DataFrame
        Output of ``load_events()`` — must contain ``announcement_date`` and
        the column named by ``stock_col`` (a list of ticker strings per row).
    prices_wide : pd.DataFrame
        Wide close-price DataFrame (see module docstring).
    market_returns : pd.Series
        Benchmark daily returns.
    event_window : tuple[int, int]
        Relative-day range, inclusive. Default: (-30, 60).
    stock_col : str
        Column in ``events`` containing the list of stocks to analyse.
        Typically "added_stocks" or "removed_stocks".

    Returns
    -------
    pd.DataFrame
        Columns: relative_day, mean_AR, mean_CAR, std_CAR, N
        where N = number of (event, stock) observations available on that day.
    """
    all_records: list[pd.DataFrame] = []

    for _, event in events.iterrows():
        tickers: list[str] = event[stock_col]
        if not tickers:
            continue

        for stock_id in tickers:
            if stock_id not in prices_wide.columns:
                logger.debug("skip %s (not in price data)", stock_id)
                continue
            try:
                car_df = compute_event_car(
                    stock_id=stock_id,
                    event_announcement_date=event["announcement_date"],
                    prices_wide=prices_wide,
                    market_returns=market_returns,
                    event_window=event_window,
                )
            except (ValueError, KeyError) as exc:
                logger.warning("skip event %s / stock %s: %s", event.get("event_id"), stock_id, exc)
                continue

            if car_df.empty:
                continue

            car_df["event_id"] = event.get("event_id", "?")
            car_df["stock_id"] = stock_id
            all_records.append(car_df)

    if not all_records:
        return pd.DataFrame(columns=["relative_day", "mean_AR", "mean_CAR", "std_CAR", "N"])

    combined = pd.concat(all_records, ignore_index=True)

    agg = (
        combined.groupby("relative_day", sort=True)
        .agg(
            mean_AR=("AR", "mean"),
            mean_CAR=("CAR", "mean"),
            std_CAR=("CAR", "std"),
            N=("CAR", "count"),
        )
        .reset_index()
    )
    return agg


def plot_average_car(
    aggregated_df: pd.DataFrame,
    save_path: str | Path,
    effective_day: Optional[int] = None,
    title: str = "Average CAR around ETF Reconstitution Announcement",
    label: str = "Added stocks",
) -> None:
    """Plot mean CAR with 95% confidence band and event-day markers.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Output of ``aggregate_cars_across_events``.
    save_path : str or Path
        Destination PNG path. Parent directory is created if needed.
    effective_day : int, optional
        Relative day of effective rebalancing (垂直虛線). If None, not drawn.
    title : str
        Chart title.
    label : str
        Legend label for the CAR line.
    """
    df = aggregated_df.sort_values("relative_day").copy()

    # 95% CI: mean ± 1.96 * std / sqrt(N)
    df["ci95"] = 1.96 * df["std_CAR"] / np.sqrt(df["N"])
    df["upper"] = df["mean_CAR"] + df["ci95"]
    df["lower"] = df["mean_CAR"] - df["ci95"]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.fill_between(df["relative_day"], df["lower"], df["upper"],
                    alpha=0.20, color="#1f77b4", label="95% 信賴區間")
    ax.plot(df["relative_day"], df["mean_CAR"],
            color="#1f77b4", linewidth=1.8, label=label)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="-")

    # Day-0 vertical line (announcement date)
    ax.axvline(0, color="#d62728", linewidth=1.2, linestyle="--", label="公告日（第 0 天）")

    # Effective date line
    if effective_day is not None:
        ax.axvline(effective_day, color="#ff7f0e", linewidth=1.2,
                   linestyle="--", label=f"生效日（第 {effective_day} 天）")

    ax.set_xlabel("相對公告日交易日數", fontsize=11)
    ax.set_ylabel("平均累積異常報酬（CAR）", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Annotate N range
    n_min, n_max = df["N"].min(), df["N"].max()
    ax.text(0.02, 0.03, f"樣本數：{n_min}–{n_max} 觀測值/天",
            transform=ax.transAxes, fontsize=8, color="gray")

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("CAR chart saved → %s", save_path)


def t_test_car(
    events: pd.DataFrame,
    prices_wide: pd.DataFrame,
    market_returns: pd.Series,
    window: tuple[int, int] = (0, 5),
    stock_col: str = "added_stocks",
) -> pd.DataFrame:
    """One-sample t-test on per-(event, stock) CAR over a specified window.

    For each (event, stock) pair, the CAR is the sum of AR from
    ``window[0]`` to ``window[1]`` inclusive (in trading-day space).
    The null hypothesis is H₀: mean CAR = 0.

    Parameters
    ----------
    events : pd.DataFrame
        Output of ``load_events()``.
    prices_wide : pd.DataFrame
        Wide close-price DataFrame.
    market_returns : pd.Series
        Benchmark daily returns.
    window : tuple[int, int]
        (start_rel_day, end_rel_day), inclusive. Default: (0, 5).
    stock_col : str
        "added_stocks" or "removed_stocks".

    Returns
    -------
    pd.DataFrame
        One row with columns: N, mean_CAR, std_CAR, t_stat, p_value, window.
        Also prints a formatted summary.
    """
    car_values: list[float] = []

    for _, event in events.iterrows():
        tickers: list[str] = event[stock_col]
        for stock_id in tickers:
            if stock_id not in prices_wide.columns:
                continue
            try:
                car_df = compute_event_car(
                    stock_id=stock_id,
                    event_announcement_date=event["announcement_date"],
                    prices_wide=prices_wide,
                    market_returns=market_returns,
                    event_window=window,
                )
            except (ValueError, KeyError):
                continue

            if car_df.empty:
                continue

            # CAR for this pair = last value of the cumulative series
            # (which runs from window[0] to window[1])
            terminal_car = car_df["CAR"].iloc[-1]
            car_values.append(terminal_car)

    n = len(car_values)
    if n < 2:
        print("Insufficient observations for t-test.")
        return pd.DataFrame()

    arr = np.array(car_values)
    mean_car = arr.mean()
    std_car = arr.std(ddof=1)
    t_stat, p_value = stats.ttest_1samp(arr, popmean=0)

    result = pd.DataFrame([{
        "window": f"[{window[0]}, {window[1]}]",
        "N": n,
        "mean_CAR": mean_car,
        "std_CAR": std_car,
        "t_stat": t_stat,
        "p_value": p_value,
    }])

    sig = (
        "***" if p_value < 0.01 else
        "**"  if p_value < 0.05 else
        "*"   if p_value < 0.10 else ""
    )

    print(f"\n{'─'*50}")
    print(f"  One-sample t-test  H₀: mean CAR = 0")
    print(f"  Window : day {window[0]} to {window[1]}  |  col: {stock_col}")
    print(f"{'─'*50}")
    print(f"  N         : {n}")
    print(f"  Mean CAR  : {mean_car:+.4f}  ({mean_car:.2%})")
    print(f"  Std CAR   : {std_car:.4f}")
    print(f"  t-stat    : {t_stat:.4f}")
    print(f"  p-value   : {p_value:.4f}  {sig}")
    print(f"{'─'*50}\n")

    return result
