"""
src/sensitivity.py
------------------
Cost sensitivity and capacity analysis for the hypothetical ETF-reconstitution strategy.

IMPORTANT DISCLAIMER
--------------------
ALL results produced by this module are **hypothetical**.
 - Transaction costs are modelled as flat rates applied uniformly to all trades.
 - Slippage is approximated as a symmetric round-trip cost in basis points.
 - Market impact is NOT modelled (positions are assumed price-takers).
 - Borrowing costs for short trades are NOT included.
 - Capacity estimates assume stable liquidity equal to the 60-day average
   before entry — actual liquidity during index events may be materially lower.

These simplifications make the results optimistic relative to a real deployment.
The purpose of this module is to understand the *sensitivity* of the signal's
alpha to realistic cost assumptions, not to forecast profitability.

Cost structure (Taiwan equity market)
--------------------------------------
fee          : brokerage commission, applied on BOTH buy and sell legs
               Regulated minimum: 0.1425 % per side → 0.285 % round-trip
tax          : securities transaction tax (STT), applied on SELL side only
               Rate: 0.30 % for listed shares (ETF redemption uses 0.10 %)
slippage_bps : symmetric half-spread / market-impact estimate in basis points,
               applied on both entry and exit legs
               → total round-trip slippage = 2 × slippage_bps / 10,000

Net hypothetical return formula
--------------------------------
net_return = raw_return - 2 × fee - tax - 2 × slippage_bps / 10,000

Note: The formula applies the same cost to EVERY trade regardless of direction
(long or short).  For shorts, borrowing cost is excluded (see disclaimer above).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from _plot_config import apply_chinese_style
apply_chinese_style()

logger = logging.getLogger(__name__)

# ── Default Taiwan cost scenarios ───────────────────────────────────────────────

DEFAULT_SCENARIOS: dict[str, dict] = {
    "no_cost": {
        "fee": 0.0,
        "tax": 0.0,
        "slippage_bps": 0,
        "label": "無成本（假設）",
    },
    "basic_cost": {
        "fee": 0.001425,
        "tax": 0.003,
        "slippage_bps": 0,
        "label": "基本成本（手續費＋稅）",
    },
    "slippage_20bps": {
        "fee": 0.001425,
        "tax": 0.003,
        "slippage_bps": 20,
        "label": "＋20bps 滑點",
    },
    "slippage_50bps": {
        "fee": 0.001425,
        "tax": 0.003,
        "slippage_bps": 50,
        "label": "＋50bps 滑點",
    },
}


# ── Core functions ──────────────────────────────────────────────────────────────


def apply_transaction_costs(
    trades: pd.DataFrame,
    scenarios: Optional[dict[str, dict]] = None,
) -> dict[str, dict]:
    """Apply hypothetical transaction cost scenarios to raw trade returns.

    For each scenario, computes per-trade net hypothetical returns then aggregates
    to event-level equal-weight returns and strategy-level metrics.

    Net return formula (per trade)
    --------------------------------
    net_return = raw_return - 2 × fee - tax - 2 × slippage_bps / 10_000

    Parameters
    ----------
    trades : pd.DataFrame
        Output of ``compute_trade_returns()`` from ``backtest.py``.
        Required columns: ``event_id``, ``ann_date``, ``hypothetical_return``.
    scenarios : dict of dict, optional
        Mapping of scenario_name → cost parameters dict with keys:
          ``fee``           float  brokerage rate per side (e.g. 0.001425)
          ``tax``           float  sell-side STT (e.g. 0.003)
          ``slippage_bps``  float  round-trip slippage per side in basis points
          ``label``         str    (optional) display name for the scenario
        Defaults to ``DEFAULT_SCENARIOS`` if None.

    Returns
    -------
    dict[str, dict]
        Keyed by scenario name.  Each value is a dict with:
          scenario_params         — original cost parameters
          trades                  — pd.DataFrame with ``net_return`` added
          event_returns           — pd.DataFrame (event_id, ann_date,
                                    n_trades, net_event_return)
          hypothetical_cumulative_return
          hypothetical_mean_event_return
          hit_rate_events
          max_single_event_loss
          n_events
          total_round_trip_cost   — constant deducted from every trade return
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    valid_raw = trades.dropna(subset=["hypothetical_return"]).copy()

    results: dict[str, dict] = {}

    for name, params in scenarios.items():
        fee          = float(params.get("fee", 0.0))
        tax          = float(params.get("tax", 0.0))
        slip_bps     = float(params.get("slippage_bps", 0))
        label        = params.get("label", name)

        # Total deterministic round-trip cost deducted from every trade
        rt_cost = 2 * fee + tax + 2 * slip_bps / 10_000

        df = valid_raw.copy()
        df["net_return"] = df["hypothetical_return"] - rt_cost

        # Event-level aggregation (equal weight within event)
        ev_agg = (
            df.groupby("event_id")
            .agg(
                ann_date=("ann_date", "first"),
                n_trades=("net_return", "count"),
                net_event_return=("net_return", "mean"),
            )
            .reset_index()
            .sort_values("ann_date")
        )

        er = ev_agg["net_event_return"].dropna()

        cum_ret  = float((1 + er).prod() - 1) if not er.empty else np.nan
        mean_ret = float(er.mean())            if not er.empty else np.nan
        hit_rate = float((er > 0).mean())      if not er.empty else np.nan
        max_loss = float(er.min())             if not er.empty else np.nan

        results[name] = {
            "scenario_params":                  params,
            "label":                            label,
            "trades":                           df,
            "event_returns":                    ev_agg,
            "total_round_trip_cost":            rt_cost,
            "hypothetical_cumulative_return":   cum_ret,
            "hypothetical_mean_event_return":   mean_ret,
            "hit_rate_events":                  hit_rate,
            "max_single_event_loss":            max_loss,
            "n_events":                         int(len(er)),
        }

        logger.info(
            "Scenario %-22s  rt_cost=%.4f%%  cum=%+.2%%  mean=%+.4%%  hit=%.0%%",
            name, rt_cost * 100, cum_ret * 100, mean_ret * 100, hit_rate * 100,
        )

    return results


def estimate_capacity(
    trades: pd.DataFrame,
    price_data: pd.DataFrame,
    max_pct_of_volume: float = 0.10,
    lookback_days: int = 60,
) -> dict:
    """Estimate hypothetical strategy capacity based on historical turnover.

    For each CONFIRMED trade (predicted AND actually added), the maximum
    notional position is bounded by:
        capacity_per_stock = avg_daily_turnover_60d × max_pct_of_volume

    where avg_daily_turnover_60d is the mean TWD daily turnover over the
    ``lookback_days`` trading days **strictly before** the trade entry date.

    These estimates are highly sensitive to:
    - Actual liquidity on event days (often lower due to crowding)
    - Number of managers running similar strategies
    - Market impact not captured here

    Parameters
    ----------
    trades : pd.DataFrame
        Output of ``compute_trade_returns()`` from ``backtest.py``.
        Required columns: ``stock_id``, ``entry_date``, ``trade_type``.
    price_data : pd.DataFrame
        Wide TWD-turnover DataFrame (rows = trading days, cols = stock IDs).
        Pass ``turnover_wide`` from ``_make_stage4_data()`` or real data.
    max_pct_of_volume : float
        Maximum fraction of average daily volume tradeable in one day.
        Default 10 %.
    lookback_days : int
        Number of trading days before entry to average for liquidity estimation.

    Returns
    -------
    dict with keys:
      per_stock_capacity        — pd.DataFrame (stock_id, event_id, entry_date,
                                  avg_daily_turnover, capacity_twd)
      per_event_capacity        — pd.DataFrame (event_id, total_capacity_twd)
      mean_event_capacity_twd   — float
      median_event_capacity_twd — float
      min_event_capacity_twd    — float
      max_event_capacity_twd    — float
      note                      — disclaimer string
    """
    confirmed = trades[trades["trade_type"] == "confirmed"].copy()

    if confirmed.empty:
        return {
            "per_stock_capacity":      pd.DataFrame(),
            "per_event_capacity":      pd.DataFrame(),
            "mean_event_capacity_twd": np.nan,
            "note": "No confirmed trades found.",
        }

    trading_days = price_data.index
    records = []

    for _, row in confirmed.iterrows():
        sid   = row["stock_id"]
        entry = row["entry_date"]
        eid   = row["event_id"]

        if sid not in price_data.columns:
            records.append({
                "stock_id": sid, "event_id": eid,
                "entry_date": entry,
                "avg_daily_turnover": np.nan,
                "capacity_twd": np.nan,
            })
            continue

        # Strictly-before-entry trading days for lookback
        past = trading_days[trading_days < entry]
        window = past[-lookback_days:] if len(past) >= lookback_days else past

        if window.empty:
            avg_turn = np.nan
        else:
            avg_turn = float(price_data.loc[window, sid].mean())

        cap = avg_turn * max_pct_of_volume if not np.isnan(avg_turn) else np.nan

        records.append({
            "stock_id":           sid,
            "event_id":           eid,
            "entry_date":         entry,
            "avg_daily_turnover": avg_turn,
            "capacity_twd":       cap,
        })

    per_stock = pd.DataFrame(records)

    per_event = (
        per_stock.groupby("event_id")["capacity_twd"]
        .sum()
        .reset_index()
        .rename(columns={"capacity_twd": "total_capacity_twd"})
    )

    ev_caps = per_event["total_capacity_twd"].dropna()

    disclaimer = (
        "Capacity estimates assume static liquidity equal to 60-day pre-event "
        "average. Actual capacity during index events is likely materially lower "
        "due to crowding and elevated market impact. NOT an investment recommendation."
    )

    logger.info(
        "Capacity estimate: mean event cap = TWD %.2fM  (n=%d confirmed trades)",
        ev_caps.mean() / 1e6 if not ev_caps.empty else 0,
        len(confirmed),
    )

    return {
        "per_stock_capacity":        per_stock,
        "per_event_capacity":        per_event,
        "mean_event_capacity_twd":   float(ev_caps.mean())   if not ev_caps.empty else np.nan,
        "median_event_capacity_twd": float(ev_caps.median()) if not ev_caps.empty else np.nan,
        "min_event_capacity_twd":    float(ev_caps.min())    if not ev_caps.empty else np.nan,
        "max_event_capacity_twd":    float(ev_caps.max())    if not ev_caps.empty else np.nan,
        "n_confirmed_trades":        len(confirmed),
        "max_pct_of_volume":         max_pct_of_volume,
        "lookback_days":             lookback_days,
        "note":                      disclaimer,
    }


def build_sensitivity_table(
    scenarios_results: dict[str, dict],
    save_dir: str | Path = "output/tables",
) -> pd.DataFrame:
    """Build a formatted cost sensitivity summary table.

    Parameters
    ----------
    scenarios_results : dict[str, dict]
        Output of ``apply_transaction_costs()``.
    save_dir : str or Path
        Directory for output files.

    Returns
    -------
    pd.DataFrame
        Index = metric names, Columns = scenario names.
        Also saves:
          - ``output/tables/sensitivity_table.csv``
          - ``output/tables/sensitivity_table.md``
    """
    METRIC_LABELS = {
        "total_round_trip_cost":           "估計來回成本 (bps)",
        "hypothetical_cumulative_return":  "假設累積報酬",
        "hypothetical_mean_event_return":  "假設平均每事件報酬",
        "hit_rate_events":                 "勝率（事件）",
        "max_single_event_loss":           "最大單事件損失",
        "n_events":                        "事件數",
    }

    rows: dict[str, dict] = {label: {} for label in METRIC_LABELS.values()}

    for name, res in scenarios_results.items():
        col_label = res.get("label", name)
        bps = res["total_round_trip_cost"] * 10_000
        rows["估計來回成本 (bps)"][col_label]          = f"{bps:.1f} bps"
        rows["假設累積報酬"][col_label]                = f"{res['hypothetical_cumulative_return']:+.2%}"
        rows["假設平均每事件報酬"][col_label]          = f"{res['hypothetical_mean_event_return']:+.4%}"
        rows["勝率（事件）"][col_label]                = f"{res['hit_rate_events']:.0%}"
        rows["最大單事件損失"][col_label]              = f"{res['max_single_event_loss']:.2%}"
        rows["事件數"][col_label]                      = str(res["n_events"])

    table = pd.DataFrame(rows).T
    table.index.name = "指標"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(save_dir / "sensitivity_table.csv")

    # Markdown version
    md_lines = [
        "> **WARNING**: All values are hypothetical. Zero market impact assumed.",
        "",
        table.to_markdown(),
        "",
        "> 手續費: 0.1425% / 單邊；證交稅: 0.3% / 賣出；滑點: 單邊 bps",
    ]
    (save_dir / "sensitivity_table.md").write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("Sensitivity table saved -> %s", save_dir)

    # ── Pretty print to console ──
    print(f"\n{'='*70}")
    print("  HYPOTHETICAL COST SENSITIVITY  (excl. market impact, borrow cost)")
    print(f"{'='*70}")
    print(table.to_string())
    print(f"\n  ⚠  All results are hypothetical. See module docstring for caveats.")
    print(f"{'='*70}\n")

    return table


def plot_alpha_decay(
    scenarios_results: dict[str, dict],
    save_path: str | Path,
    metric: str = "hypothetical_mean_event_return",
    title: str = "假設 Alpha 隨成本遞減（不含市場衝擊）",
) -> None:
    """Bar chart of remaining hypothetical alpha under each cost scenario.

    Parameters
    ----------
    scenarios_results : dict[str, dict]
        Output of ``apply_transaction_costs()``.
    save_path : str or Path
        Destination PNG.
    metric : str
        Which metric to plot.  Default: mean per-event return.
    title : str
        Chart title — MUST reference hypothetical / 假設.
    """
    names  = list(scenarios_results.keys())
    labels = [scenarios_results[n].get("label", n) for n in names]
    values = [scenarios_results[n][metric] * 100 for n in names]
    rt_bps = [scenarios_results[n]["total_round_trip_cost"] * 10_000 for n in names]

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    fig, (ax_main, ax_cost) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Top: alpha bar chart ──
    bars = ax_main.bar(range(len(names)), values, color=colors, alpha=0.78,
                       width=0.55, edgecolor="white", linewidth=0.8)
    ax_main.axhline(0, color="black", linewidth=0.9)

    for bar, val in zip(bars, values):
        ypos = bar.get_height() + (0.003 if val >= 0 else -0.008)
        ax_main.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.3f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax_main.set_xticks(range(len(names)))
    ax_main.set_xticklabels(labels, fontsize=10)
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}%"))
    ax_main.set_ylabel("假設平均每事件報酬 (%)", fontsize=11)
    ax_main.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax_main.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    # Zero-cost alpha reference line
    zero_cost_val = values[0]
    ax_main.axhline(zero_cost_val, color="#1f77b4", linewidth=1.2,
                    linestyle="--", alpha=0.6, label=f"無成本假設基準 {zero_cost_val:+.3f}%")
    ax_main.legend(fontsize=9)

    # ── Bottom: round-trip cost bar ──
    ax_cost.bar(range(len(names)), rt_bps, color="#7f7f7f", alpha=0.65,
                width=0.55, edgecolor="white")
    for i, bps in enumerate(rt_bps):
        ax_cost.text(i, bps + 0.5, f"{bps:.1f} bps", ha="center",
                     va="bottom", fontsize=9)
    ax_cost.set_xticks(range(len(names)))
    ax_cost.set_xticklabels(labels, fontsize=9)
    ax_cost.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    ax_cost.set_ylabel("來回成本 (bps)", fontsize=10)
    ax_cost.set_title("估計來回成本（手續費＋稅＋滑點）", fontsize=10)
    ax_cost.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)

    fig.text(
        0.5, 0.01,
        "假設結果 — 不含市場衝擊 — 不構成投資建議",
        ha="center", fontsize=8, color="gray", style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Alpha decay chart saved -> %s", save_path)
