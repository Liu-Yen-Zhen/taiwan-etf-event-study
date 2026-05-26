"""
src/backtest.py
---------------
Hypothetical event-driven strategy backtest for ETF reconstitution predictions.

IMPORTANT DISCLAIMER
--------------------
All results produced by this module are **hypothetical** and are presented
for research purposes only.  They assume:
  - Trades execute at closing prices with zero market impact
  - No transaction costs, bid-ask spread, borrowing costs, or slippage
  - Perfect position sizing (equal weight within each event)
  - No portfolio-level constraints (leverage, concentration, etc.)

These assumptions make returns look better than achievable in practice.
This module exists to understand the *signal quality* of the prediction model,
not to estimate real-world profitability.

Strategy logic
--------------
Prediction date  = announcement_date + signal_date_offset  (offset < 0, e.g. -14)
Entry date       = prediction date (close price, equal weight)

At announcement:
  - Predicted AND actually added → hold, exit at effective_date + 1 trading day (close)
  - Predicted BUT NOT actually added → exit at announcement_date (close); these
    are called "rejected" trades

Exit date:
  - confirmed  : effective_date + 1 trading day (close)
  - rejected   : announcement_date (close)
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from _plot_config import apply_chinese_style
apply_chinese_style()

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_td(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """First trading day on-or-after date.  Returns None if out of range."""
    future = trading_days[trading_days >= date]
    return future[0] if not future.empty else None


def _offset_td(date: pd.Timestamp, offset: int,
               trading_days: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """date shifted by offset trading days.  Negative offset = before."""
    base = _resolve_td(date, trading_days)
    if base is None:
        return None
    idx = trading_days.get_loc(base) + offset
    if idx < 0 or idx >= len(trading_days):
        return None
    return trading_days[idx]


def _close_price(date: pd.Timestamp, stock_id: str,
                 prices_wide: pd.DataFrame) -> float:
    """Close price for stock_id on date.  Returns nan if unavailable."""
    if date not in prices_wide.index or stock_id not in prices_wide.columns:
        return np.nan
    return float(prices_wide.loc[date, stock_id])


# ── Core functions ─────────────────────────────────────────────────────────────


def generate_signals(
    events: pd.DataFrame,
    prediction_func: Callable[[pd.Timestamp, Optional[set]], list[str]],
    prices_wide: pd.DataFrame,
    signal_date_offset: int = -14,
    initial_constituents: Optional[set] = None,
) -> pd.DataFrame:
    """Generate hypothetical entry/exit signals for each event.

    For each event (in chronological order):
    1. Compute prediction date = announcement_date + signal_date_offset (trading days)
    2. Call prediction_func(pred_date, current_constituents) → predicted_additions
    3. Generate trades:
       - Predicted AND actually added  → CONFIRMED trade:
             entry=pred_date, exit=effective_date+1
       - Predicted BUT NOT actually added → REJECTED trade:
             entry=pred_date, exit=announcement_date

    Parameters
    ----------
    events : pd.DataFrame
        From load_events().  Sorted by announcement_date.
    prediction_func : Callable[[Timestamp, Optional[set]], list[str]]
        Should replicate the prediction used in the research — no look-ahead allowed.
        Signature: (ref_date, current_constituents) → list of predicted stock_ids.
    prices_wide : pd.DataFrame
        Wide close-price DataFrame (used only for trading-day resolution).
    signal_date_offset : int
        Negative integer.  Number of trading days before announcement to predict.
    initial_constituents : set or None
        ETF constituents before the first event.

    Returns
    -------
    pd.DataFrame
        Columns: event_id, stock_id, entry_date, exit_date, trade_type
                 (trade_type: "confirmed" | "rejected")
        Hypothetical trades only — does NOT include stocks we failed to predict.
    """
    assert signal_date_offset < 0, "signal_date_offset must be negative"

    trading_days = prices_wide.index
    events_sorted = events.sort_values("announcement_date").reset_index(drop=True)
    current_constituents = set(initial_constituents or [])
    records = []

    for _, event in events_sorted.iterrows():
        ann_date = pd.Timestamp(event["announcement_date"])
        eff_date = pd.Timestamp(event["effective_date"])

        ann_td = _resolve_td(ann_date, trading_days)
        eff_td = _resolve_td(eff_date, trading_days)
        if ann_td is None or eff_td is None:
            logger.warning("Event %s: date outside data range — skip",
                           event.get("event_id"))
            continue

        # Prediction date (strictly before announcement)
        pred_date = _offset_td(ann_td, signal_date_offset, trading_days)
        if pred_date is None:
            logger.warning("Event %s: prediction date before data start — skip",
                           event.get("event_id"))
            continue
        assert pred_date < ann_td, "Look-ahead bias detected"

        # Exit date for confirmed trades = effective + 1 trading day
        exit_confirmed = _offset_td(eff_td, +1, trading_days)
        if exit_confirmed is None:
            exit_confirmed = eff_td  # fallback: exit on effective day

        # Actual additions and predicted additions
        actual_added = set(event["added_stocks"])
        predicted = prediction_func(pred_date, current_constituents.copy())

        for stock_id in predicted:
            if stock_id in actual_added:
                trade_type = "confirmed"
                exit_date  = exit_confirmed
            else:
                trade_type = "rejected"
                exit_date  = ann_td  # sell on announcement day

            records.append({
                "event_id":   event.get("event_id"),
                "stock_id":   stock_id,
                "entry_date": pred_date,
                "exit_date":  exit_date,
                "trade_type": trade_type,
                "ann_date":   ann_td,
            })

        # Update constituents after effective date (for next event's buffer)
        current_constituents -= set(event["removed_stocks"])
        current_constituents |= set(event["added_stocks"])

    df = pd.DataFrame(records)
    logger.info(
        "generate_signals: %d trades total  (%d confirmed, %d rejected)",
        len(df),
        (df["trade_type"] == "confirmed").sum() if not df.empty else 0,
        (df["trade_type"] == "rejected").sum() if not df.empty else 0,
    )
    return df


def compute_trade_returns(
    trades: pd.DataFrame,
    prices_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Compute hypothetical return for each trade using close prices.

    hypothetical_return = (exit_close - entry_close) / entry_close

    Trades where either price is missing are flagged with NaN and logged.

    Parameters
    ----------
    trades : pd.DataFrame
        Output of generate_signals().
    prices_wide : pd.DataFrame
        Wide close-price DataFrame.

    Returns
    -------
    pd.DataFrame
        Original trades plus columns: entry_close, exit_close,
        hypothetical_return.
    """
    df = trades.copy()

    df["entry_close"] = df.apply(
        lambda r: _close_price(r["entry_date"], r["stock_id"], prices_wide), axis=1
    )
    df["exit_close"] = df.apply(
        lambda r: _close_price(r["exit_date"], r["stock_id"], prices_wide), axis=1
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        df["hypothetical_return"] = np.where(
            df["entry_close"].gt(0) & df["entry_close"].notna() & df["exit_close"].notna(),
            (df["exit_close"] - df["entry_close"]) / df["entry_close"],
            np.nan,
        )

    n_nan = df["hypothetical_return"].isna().sum()
    if n_nan:
        logger.warning("%d trade(s) have missing prices — excluded from analysis", n_nan)

    return df


def aggregate_event_returns(
    trades_with_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate hypothetical trade returns to event level (equal weight).

    Within each event, all trades receive equal weight regardless of market cap.
    (Equal weighting is a simplification; real ETF weight scheme is more complex.)

    Parameters
    ----------
    trades_with_returns : pd.DataFrame
        Output of compute_trade_returns().

    Returns
    -------
    pd.DataFrame
        Columns: event_id, ann_date, n_trades, n_confirmed, n_rejected,
                 hypothetical_event_return.
        One row per event.  Events with zero valid trades are excluded.
    """
    valid = trades_with_returns.dropna(subset=["hypothetical_return"])

    if valid.empty:
        return pd.DataFrame(columns=[
            "event_id", "ann_date", "n_trades", "n_confirmed",
            "n_rejected", "hypothetical_event_return",
        ])

    agg = (
        valid.groupby("event_id")
        .agg(
            ann_date=("ann_date", "first"),
            n_trades=("hypothetical_return", "count"),
            n_confirmed=("trade_type", lambda x: (x == "confirmed").sum()),
            n_rejected=("trade_type", lambda x: (x == "rejected").sum()),
            hypothetical_event_return=("hypothetical_return", "mean"),
        )
        .reset_index()
        .sort_values("ann_date")
    )
    return agg


def compute_strategy_metrics(
    event_returns: pd.DataFrame,
) -> dict:
    """Compute hypothetical strategy-level metrics.

    ALL values are hypothetical, assuming zero transaction costs and
    equal-weight positions.  These are reported purely for signal analysis.

    Parameters
    ----------
    event_returns : pd.DataFrame
        Output of aggregate_event_returns().

    Returns
    -------
    dict with keys:
      hypothetical_cumulative_return  — chained (1+r) product minus 1
      hypothetical_mean_event_return  — simple average across events
      hit_rate_events                 — fraction of events with return > 0
      max_single_event_loss           — worst single-event return
      n_events                        — number of events analysed
      note                            — disclaimer string
    """
    er = event_returns["hypothetical_event_return"].dropna()
    if er.empty:
        return {"note": "No valid event returns to compute metrics.",
                "n_events": 0}

    cum_return = float((1 + er).prod() - 1)
    mean_return = float(er.mean())
    hit_rate = float((er > 0).mean())
    max_loss = float(er.min())

    disclaimer = (
        "HYPOTHETICAL results. Assumes zero transaction costs, perfect fills "
        "at closing prices, equal-weight positions. NOT a forecast of "
        "real-world profitability."
    )

    metrics = {
        "hypothetical_cumulative_return": cum_return,
        "hypothetical_mean_event_return": mean_return,
        "hit_rate_events": hit_rate,
        "max_single_event_loss": max_loss,
        "n_events": int(len(er)),
        "note": disclaimer,
    }

    # ── Pretty print ──
    print(f"\n{'='*62}")
    print("  HYPOTHETICAL STRATEGY METRICS  (excl. transaction costs)")
    print(f"{'='*62}")
    print(f"  N events analysed           : {metrics['n_events']}")
    print(f"  Hypothetical cumul. return  : {cum_return:+.2%}")
    print(f"  Hypothetical mean / event   : {mean_return:+.2%}")
    print(f"  Hit rate (profitable events): {hit_rate:.0%}")
    print(f"  Max single-event loss       : {max_loss:.2%}")
    print(f"\n  ⚠  {disclaimer}")
    print(f"{'='*62}\n")

    return metrics


def plot_cumulative_return(
    event_returns: pd.DataFrame,
    save_path: str | Path,
    title: str = "假設累積報酬（不含交易成本）",
) -> None:
    """Plot hypothetical cumulative return per event.

    The chart explicitly labels itself as hypothetical in both the title
    and a bottom-edge watermark.

    Parameters
    ----------
    event_returns : pd.DataFrame
        Output of aggregate_event_returns().
    save_path : str or Path
        Destination PNG.
    title : str
        Chart title — MUST contain 「假設」or equivalent disclaimer.
    """
    df = event_returns.dropna(subset=["hypothetical_event_return"]).sort_values("ann_date").copy()
    df = df.reset_index(drop=True)
    df["cum_return"] = (1 + df["hypothetical_event_return"]).cumprod() - 1

    fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    # ── Top: cumulative return ──
    ax = axes[0]
    ax.plot(df.index, df["cum_return"] * 100, "o-",
            color="#1f77b4", linewidth=2, markersize=6, label="假設累積報酬")
    ax.fill_between(df.index, 0, df["cum_return"] * 100,
                    where=df["cum_return"] >= 0, alpha=0.12, color="#2ca02c")
    ax.fill_between(df.index, 0, df["cum_return"] * 100,
                    where=df["cum_return"] < 0,  alpha=0.12, color="#d62728")
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(df.index)
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in df["ann_date"]],
        rotation=35, ha="right", fontsize=9,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.set_ylabel("假設累積報酬 (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    # Annotate final cumulative return
    final = df["cum_return"].iloc[-1]
    ax.annotate(
        f"最終累計: {final:+.2%}",
        xy=(df.index[-1], final * 100),
        xytext=(-50, 15),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
        color="#1f77b4",
    )

    # ── Bottom: per-event return bars ──
    ax2 = axes[1]
    colors = ["#2ca02c" if r > 0 else "#d62728"
              for r in df["hypothetical_event_return"]]
    ax2.bar(df.index, df["hypothetical_event_return"] * 100,
            color=colors, alpha=0.75, width=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(df.index)
    ax2.set_xticklabels(
        [d.strftime("%Y-%m") for d in df["ann_date"]], fontsize=9
    )
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax2.set_ylabel("每事件報酬 (%)", fontsize=10)
    ax2.set_title("每事件假設報酬（不含成本）", fontsize=10)
    ax2.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    # Watermark disclaimer
    fig.text(0.5, 0.01,
             "假設結果 — 不含交易成本 — 不構成投資建議",
             ha="center", fontsize=8, color="gray", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Hypothetical return chart saved -> %s", save_path)


# ── Long / Short extension ─────────────────────────────────────────────────────


def generate_signals_long_short(
    events: pd.DataFrame,
    long_prediction_func: Callable[[pd.Timestamp, Optional[set]], list[str]],
    short_prediction_func: Callable[[pd.Timestamp, Optional[set]], list[str]],
    prices_wide: pd.DataFrame,
    signal_date_offset: int = -14,
    initial_constituents: Optional[set] = None,
) -> pd.DataFrame:
    """Generate hypothetical LONG (additions) and SHORT (removals) signals.

    Long leg  : predicted additions — same entry/exit logic as generate_signals().
    Short leg : predicted removals.
      - confirmed removal → cover at effective_date + 1 trading day (close)
      - rejected removal  → cover at announcement_date (close)

    Short P&L is realised when the stock *falls* after index removal pressure.
    All results remain hypothetical (zero borrow cost, perfect fill).

    Parameters
    ----------
    events : pd.DataFrame
        From load_events().
    long_prediction_func : Callable
        Returns predicted additions.  Signature: (ref_date, current_constituents).
    short_prediction_func : Callable
        Returns predicted removals.  Same signature.
    prices_wide : pd.DataFrame
        Wide close-price DataFrame.
    signal_date_offset : int
        Negative integer.  Trading-day offset before announcement.
    initial_constituents : set or None
        ETF constituents before the first event.

    Returns
    -------
    pd.DataFrame
        Same columns as generate_signals() plus ``direction`` ("long" | "short").
    """
    assert signal_date_offset < 0, "signal_date_offset must be negative"

    trading_days = prices_wide.index
    events_sorted = events.sort_values("announcement_date").reset_index(drop=True)
    current_constituents = set(initial_constituents or [])
    records = []

    for _, event in events_sorted.iterrows():
        ann_date = pd.Timestamp(event["announcement_date"])
        eff_date = pd.Timestamp(event["effective_date"])

        ann_td = _resolve_td(ann_date, trading_days)
        eff_td = _resolve_td(eff_date, trading_days)
        if ann_td is None or eff_td is None:
            logger.warning("Event %s: date outside data range — skip",
                           event.get("event_id"))
            continue

        pred_date = _offset_td(ann_td, signal_date_offset, trading_days)
        if pred_date is None:
            logger.warning("Event %s: prediction date before data start — skip",
                           event.get("event_id"))
            continue
        assert pred_date < ann_td, "Look-ahead bias detected"

        exit_confirmed = _offset_td(eff_td, +1, trading_days)
        if exit_confirmed is None:
            exit_confirmed = eff_td

        actual_added   = set(event["added_stocks"])
        actual_removed = set(event["removed_stocks"])

        # ── Long leg ──
        for stock_id in long_prediction_func(pred_date, current_constituents.copy()):
            if stock_id in actual_added:
                trade_type, exit_date = "confirmed", exit_confirmed
            else:
                trade_type, exit_date = "rejected", ann_td
            records.append({
                "event_id":   event.get("event_id"),
                "stock_id":   stock_id,
                "entry_date": pred_date,
                "exit_date":  exit_date,
                "trade_type": trade_type,
                "ann_date":   ann_td,
                "direction":  "long",
            })

        # ── Short leg ──
        for stock_id in short_prediction_func(pred_date, current_constituents.copy()):
            if stock_id in actual_removed:
                trade_type, exit_date = "confirmed", exit_confirmed
            else:
                trade_type, exit_date = "rejected", ann_td
            records.append({
                "event_id":   event.get("event_id"),
                "stock_id":   stock_id,
                "entry_date": pred_date,
                "exit_date":  exit_date,
                "trade_type": trade_type,
                "ann_date":   ann_td,
                "direction":  "short",
            })

        current_constituents -= set(event["removed_stocks"])
        current_constituents |= set(event["added_stocks"])

    df = pd.DataFrame(records)
    if not df.empty:
        n_long  = (df["direction"] == "long").sum()
        n_short = (df["direction"] == "short").sum()
    else:
        n_long = n_short = 0
    logger.info(
        "generate_signals_long_short: %d long trades, %d short trades",
        n_long, n_short,
    )
    return df


def compute_directional_returns(trades_with_returns: pd.DataFrame) -> pd.DataFrame:
    """Add ``directional_return`` column that accounts for long/short direction.

    For **long** trades  : directional_return =  hypothetical_return
    For **short** trades : directional_return = -hypothetical_return
                           (profit when price falls after removal)

    If the ``direction`` column is absent every trade is treated as long.

    Parameters
    ----------
    trades_with_returns : pd.DataFrame
        Output of compute_trade_returns(), optionally with a ``direction`` column.

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus ``directional_return`` column.
    """
    df = trades_with_returns.copy()
    if "direction" not in df.columns:
        df["directional_return"] = df["hypothetical_return"]
    else:
        df["directional_return"] = np.where(
            df["direction"] == "short",
            -df["hypothetical_return"],
            df["hypothetical_return"],
        )
    return df


def aggregate_long_short_event_returns(
    trades_with_dir_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate L/S directional returns to event level (equal weight per leg).

    Each leg (long / short) is equal-weighted independently within the event.
    The combined event return is the simple average of all directional returns.

    Parameters
    ----------
    trades_with_dir_returns : pd.DataFrame
        Output of compute_directional_returns() — must have ``directional_return``
        and ``direction`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: event_id, ann_date, n_long, n_short,
                 long_return, short_return, ls_event_return.
    """
    valid = trades_with_dir_returns.dropna(subset=["directional_return"])

    if valid.empty:
        return pd.DataFrame(columns=[
            "event_id", "ann_date", "n_long", "n_short",
            "long_return", "short_return", "ls_event_return",
        ])

    agg = (
        valid.groupby("event_id")
        .agg(
            ann_date=("ann_date", "first"),
            n_long=("direction", lambda x: (x == "long").sum()),
            n_short=("direction", lambda x: (x == "short").sum()),
            long_return=(
                "directional_return",
                lambda x: x[valid.loc[x.index, "direction"] == "long"].mean()
                if (valid.loc[x.index, "direction"] == "long").any()
                else np.nan,
            ),
            short_return=(
                "directional_return",
                lambda x: x[valid.loc[x.index, "direction"] == "short"].mean()
                if (valid.loc[x.index, "direction"] == "short").any()
                else np.nan,
            ),
            ls_event_return=("directional_return", "mean"),
        )
        .reset_index()
        .sort_values("ann_date")
    )
    return agg


# ── Risk metrics ───────────────────────────────────────────────────────────────


def compute_max_drawdown(
    event_returns: pd.DataFrame,
    return_col: str = "hypothetical_event_return",
) -> tuple[float, pd.Series]:
    """Compute maximum drawdown from the event-chained cumulative return.

    Parameters
    ----------
    event_returns : pd.DataFrame
        Output of aggregate_event_returns() (or equivalent).
    return_col : str
        Column of per-event returns to use.

    Returns
    -------
    max_drawdown : float
        Most negative trough-to-peak ratio (≤ 0).
    drawdown_series : pd.Series
        Per-event drawdown values (index = positional event number).
    """
    er = event_returns[return_col].dropna().reset_index(drop=True)
    cum = (1 + er).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min()), drawdown


def compute_holding_periods(
    trades: pd.DataFrame,
    prices_wide: pd.DataFrame,
) -> pd.Series:
    """Compute holding period in trading days for each trade.

    Parameters
    ----------
    trades : pd.DataFrame
        Must have ``entry_date`` and ``exit_date`` columns.
    prices_wide : pd.DataFrame
        Used to count trading days between dates.

    Returns
    -------
    pd.Series
        Integer holding periods (trading days), same index as trades.
    """
    trading_days = prices_wide.index

    def _td_diff(entry, exit_):
        try:
            i0 = trading_days.get_loc(entry)
            i1 = trading_days.get_loc(exit_)
            return i1 - i0
        except KeyError:
            return np.nan

    return trades.apply(
        lambda r: _td_diff(r["entry_date"], r["exit_date"]), axis=1
    )


# ── Benchmark comparison ───────────────────────────────────────────────────────


def compute_benchmark_event_returns(
    signals: pd.DataFrame,
    market_returns: pd.Series,
) -> pd.DataFrame:
    """Compute market-index return over each event's holding window.

    For each event the window is [min(entry_date), max(exit_date)] across
    all trades in that event.  The benchmark return is the compounded
    daily market return over that window.

    Parameters
    ----------
    signals : pd.DataFrame
        Output of generate_signals() or generate_signals_long_short().
        Must have: event_id, ann_date, entry_date, exit_date.
    market_returns : pd.Series
        Daily market returns (e.g. TAIEX) with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Columns: event_id, ann_date, entry_date, exit_date, benchmark_return.
    """
    records = []
    for event_id, grp in signals.groupby("event_id"):
        entry  = grp["entry_date"].min()
        exit_  = grp["exit_date"].max()
        ann_dt = grp["ann_date"].iloc[0]

        mask = (market_returns.index >= entry) & (market_returns.index <= exit_)
        period_ret = market_returns[mask]
        bm = float((1 + period_ret).prod() - 1) if not period_ret.empty else np.nan

        records.append({
            "event_id":        event_id,
            "ann_date":        ann_dt,
            "entry_date":      entry,
            "exit_date":       exit_,
            "benchmark_return": bm,
        })

    return pd.DataFrame(records).sort_values("ann_date").reset_index(drop=True)


# ── Additional charts ──────────────────────────────────────────────────────────


def plot_strategy_vs_benchmark(
    event_returns: pd.DataFrame,
    benchmark_event_returns: pd.DataFrame,
    save_path: str | Path,
    strategy_label: str = "假設策略（多頭）",
    benchmark_label: str = "大盤基準",
    title: str = "假設累積報酬 vs 大盤基準（不含成本）",
) -> None:
    """Plot hypothetical strategy cumulative return against a market benchmark.

    Parameters
    ----------
    event_returns : pd.DataFrame
        Output of aggregate_event_returns().
    benchmark_event_returns : pd.DataFrame
        Output of compute_benchmark_event_returns().  Must have ``benchmark_return``
        aligned by event order (merged on event_id internally).
    save_path : str or Path
        Destination PNG.
    strategy_label, benchmark_label, title : str
        Labels for the chart.
    """
    strat = (
        event_returns.dropna(subset=["hypothetical_event_return"])
        .sort_values("ann_date")
        .copy()
        .reset_index(drop=True)
    )
    strat["cum_strategy"] = (1 + strat["hypothetical_event_return"]).cumprod() - 1

    # Align benchmark by event_id
    bm = benchmark_event_returns[["event_id", "benchmark_return"]].copy()
    merged = strat.merge(bm, on="event_id", how="left")
    merged["cum_benchmark"] = (
        (1 + merged["benchmark_return"].fillna(0)).cumprod() - 1
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(merged.index, merged["cum_strategy"] * 100, "o-",
            color="#1f77b4", linewidth=2, markersize=5, label=strategy_label)
    ax.plot(merged.index, merged["cum_benchmark"] * 100, "s--",
            color="#ff7f0e", linewidth=1.8, markersize=4, alpha=0.8,
            label=benchmark_label)
    ax.axhline(0, color="black", linewidth=0.7)

    ax.set_xticks(merged.index)
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in merged["ann_date"]],
        rotation=35, ha="right", fontsize=9,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.set_ylabel("假設累積報酬 (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    # Excess return annotation
    final_strat = merged["cum_strategy"].iloc[-1]
    final_bm    = merged["cum_benchmark"].iloc[-1]
    excess      = final_strat - final_bm
    ax.annotate(
        f"假設超額報酬: {excess:+.2%}",
        xy=(merged.index[-1], final_strat * 100),
        xytext=(-80, 12),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray"),
        color="#1f77b4",
    )

    fig.text(0.5, 0.01,
             "假設結果 — 不含交易成本 — 不構成投資建議",
             ha="center", fontsize=8, color="gray", style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Strategy vs benchmark chart saved -> %s", save_path)


def plot_risk_breakdown(
    trades_with_returns: pd.DataFrame,
    event_returns: pd.DataFrame,
    prices_wide: pd.DataFrame,
    save_path: str | Path,
    title: str = "假設風險指標明細（不含成本）",
) -> None:
    """Four-panel risk breakdown chart.

    Panel 1 (top-left)  : Confirmed vs Rejected hypothetical return distributions
    Panel 2 (top-right) : Holding period distribution in trading days
    Panel 3 (bottom-left): Per-event hypothetical return scatter with type breakdown
    Panel 4 (bottom-right): Cumulative return and Max Drawdown shading

    Parameters
    ----------
    trades_with_returns : pd.DataFrame
        Output of compute_trade_returns().
    event_returns : pd.DataFrame
        Output of aggregate_event_returns().
    prices_wide : pd.DataFrame
        Used for trading-day counting.
    save_path : str or Path
        Destination PNG.
    """
    twr = trades_with_returns.dropna(subset=["hypothetical_return"]).copy()
    er  = event_returns.dropna(subset=["hypothetical_event_return"]).sort_values(
        "ann_date"
    ).reset_index(drop=True).copy()
    er["cum_return"] = (1 + er["hypothetical_event_return"]).cumprod() - 1

    # Holding periods
    twr["holding_td"] = compute_holding_periods(twr, prices_wide)

    # Drawdown series
    _, dd_series = compute_max_drawdown(er)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ── Panel 1: return distributions by trade type ──
    ax1 = axes[0, 0]
    for ttype, color, label in [
        ("confirmed", "#2ca02c", "確認交易（假設持有至生效日）"),
        ("rejected",  "#d62728", "拒絕交易（公告日出場）"),
    ]:
        data = twr.loc[twr["trade_type"] == ttype, "hypothetical_return"] * 100
        if not data.empty:
            ax1.hist(data, bins=15, alpha=0.6, color=color, label=label, edgecolor="white")
    ax1.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax1.set_xlabel("假設報酬 (%)", fontsize=10)
    ax1.set_ylabel("次數", fontsize=10)
    ax1.set_title("假設報酬分佈：Confirmed vs Rejected", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 2: holding period distribution ──
    ax2 = axes[0, 1]
    hp = twr["holding_td"].dropna()
    if not hp.empty:
        ax2.hist(hp, bins=20, color="#1f77b4", alpha=0.7, edgecolor="white")
        ax2.axvline(hp.mean(), color="orange", linewidth=1.5,
                    linestyle="--", label=f"平均 {hp.mean():.1f} 交易日")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("持倉期（交易日）", fontsize=10)
    ax2.set_ylabel("次數", fontsize=10)
    ax2.set_title("假設持倉期分佈", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 3: per-event return with confirmed/rejected split ──
    ax3 = axes[1, 0]
    if "n_confirmed" in er.columns and "n_rejected" in er.columns:
        colors = [
            "#2ca02c" if nc > nr else "#d62728" if nr > nc else "#ff7f0e"
            for nc, nr in zip(er["n_confirmed"], er["n_rejected"])
        ]
    else:
        colors = ["#1f77b4"] * len(er)
    ax3.bar(er.index, er["hypothetical_event_return"] * 100,
            color=colors, alpha=0.75, width=0.6)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xticks(er.index)
    ax3.set_xticklabels([d.strftime("%Y-%m") for d in er["ann_date"]], fontsize=8)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax3.set_ylabel("每事件假設報酬 (%)", fontsize=10)
    ax3.set_title("每事件假設報酬（綠=confirmed為主，紅=rejected為主）", fontsize=9)
    ax3.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Panel 4: cumulative return + drawdown shading ──
    ax4 = axes[1, 1]
    ax4.plot(er.index, er["cum_return"] * 100, "o-",
             color="#1f77b4", linewidth=2, markersize=5, label="假設累積報酬")
    ax4.fill_between(er.index, 0, er["cum_return"] * 100,
                     where=er["cum_return"] >= 0, alpha=0.10, color="#2ca02c")
    ax4.fill_between(er.index, 0, er["cum_return"] * 100,
                     where=er["cum_return"] < 0,  alpha=0.10, color="#d62728")

    # Overlay drawdown as shaded area
    ax4b = ax4.twinx()
    ax4b.fill_between(dd_series.index, dd_series * 100, 0,
                      alpha=0.18, color="#9467bd", label="假設回撤")
    ax4b.set_ylabel("假設回撤 (%)", fontsize=9, color="#9467bd")
    ax4b.tick_params(axis="y", labelcolor="#9467bd")
    ax4b.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))

    max_dd, _ = compute_max_drawdown(er)
    ax4.set_title(f"假設累積報酬與回撤  (Max DD={max_dd:.2%})", fontsize=9)
    ax4.set_xticks(er.index)
    ax4.set_xticklabels([d.strftime("%Y-%m") for d in er["ann_date"]], fontsize=8)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax4.set_ylabel("假設累積報酬 (%)", fontsize=10)
    ax4.axhline(0, color="black", linewidth=0.7)
    ax4.grid(axis="y", linestyle=":", alpha=0.5)

    handles1, labels1 = ax4.get_legend_handles_labels()
    handles2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(handles1 + handles2, labels1 + labels2, fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             "假設結果 — 不含交易成本 — 不構成投資建議",
             ha="center", fontsize=8, color="gray", style="italic")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Risk breakdown chart saved -> %s", save_path)


def plot_parameter_sensitivity(
    sensitivity_df: pd.DataFrame,
    save_path: str | Path,
    title: str = "假設參數敏感度：信號提前天數 vs 預測績效與假設報酬",
) -> None:
    """Plot parameter sensitivity: signal offset vs prediction metrics and return.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Must have columns:
          offset           — signal_date_offset values (e.g. -7, -10, -14, -17, -21)
          precision        — mean out-of-sample precision
          recall           — mean out-of-sample recall
          f1               — mean F1 score
          hypothetical_mean_return — mean per-event hypothetical return
    save_path : str or Path
        Destination PNG.
    title : str
        Chart title.
    """
    df = sensitivity_df.sort_values("offset").copy()
    x_labels = [str(v) for v in df["offset"]]
    x = range(len(df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── Top: prediction metrics ──
    ax1.plot(x, df["precision"], "o-", color="#1f77b4", linewidth=2,
             markersize=7, label="精確率")
    ax1.plot(x, df["recall"],    "s-", color="#2ca02c", linewidth=2,
             markersize=7, label="召回率")
    ax1.plot(x, df["f1"],        "^-", color="#ff7f0e", linewidth=2,
             markersize=7, label="F1 分數")
    ax1.set_ylabel("分數 (0–1)", fontsize=11)
    ax1.set_title("預測指標 vs 信號提前天數", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}"))

    # Highlight best F1
    best_idx = df["f1"].idxmax()
    best_pos = df.index.get_loc(best_idx)
    ax1.axvline(best_pos, color="red", linewidth=1.2, linestyle="--", alpha=0.5)
    ax1.annotate(
        f"最佳 F1={df.loc[best_idx,'f1']:.2f}\n(offset={df.loc[best_idx,'offset']})",
        xy=(best_pos, df.loc[best_idx, "f1"]),
        xytext=(10, -25),
        textcoords="offset points",
        fontsize=8,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    # ── Bottom: hypothetical mean return ──
    bar_colors = [
        "#2ca02c" if r > 0 else "#d62728"
        for r in df["hypothetical_mean_return"]
    ]
    ax2.bar(x, df["hypothetical_mean_return"] * 100,
            color=bar_colors, alpha=0.75, width=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axvline(best_pos, color="red", linewidth=1.2, linestyle="--", alpha=0.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(x_labels, fontsize=10)
    ax2.set_xlabel("signal_date_offset（交易日）", fontsize=11)
    ax2.set_ylabel("假設平均每事件報酬 (%)", fontsize=10)
    ax2.set_title("假設報酬 vs 信號提前天數（不含成本）", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             "假設結果 — 不含交易成本 — 不構成投資建議",
             ha="center", fontsize=8, color="gray", style="italic")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Parameter sensitivity chart saved -> %s", save_path)
