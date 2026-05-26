"""
scripts/run_car_analysis_real.py
--------------------------------
Stage-2 re-run with REAL FinMind data.

Inputs
------
  data/raw/events.csv                       — 7 real 00919 reconstitution events
  data/processed/stock_prices.parquet       — 89 stocks × 830 trading days

Method
------
  • Market-adjusted abnormal return: AR_t = r_stock_t - r_TAIEX_t
  • Event window in trading-day space (negative = before announcement)
  • Missing days dropped at the per-(day, stock, event) level

Outputs
-------
  output/figures/car_added_stocks_real.png
  output/figures/car_removed_stocks_real.png
  output/tables/car_summary_real.csv
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from data_fetcher import load_events
from event_study import (
    aggregate_cars_across_events,
    plot_average_car,
    t_test_car,
)

PARQUET = ROOT / "data" / "processed" / "stock_prices.parquet"
EVENTS  = ROOT / "data" / "raw" / "events.csv"
FIG_DIR = ROOT / "output" / "figures"
TBL_DIR = ROOT / "output" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ── Load price panel ───────────────────────────────────────────────────
    prices_long = pd.read_parquet(PARQUET)
    prices_wide = (
        prices_long
        .pivot(index="date", columns="stock_id", values="close")
        .sort_index()
    )

    if "TAIEX" not in prices_wide.columns:
        raise RuntimeError("TAIEX 不在 prices_wide — 無法計算市場調整 AR")
    market_returns = prices_wide["TAIEX"].pct_change()

    print(f"Price panel        : {prices_wide.shape[0]} 交易日 × {prices_wide.shape[1]} 股票")
    print(f"Date range         : {prices_wide.index.min().date()} → {prices_wide.index.max().date()}")

    # ── Load events ────────────────────────────────────────────────────────
    events = load_events(EVENTS)
    print(f"Events             : {len(events)} 筆")

    # ── Determine effective_day relative to announcement (trading-day) ────
    trading_days = prices_wide.index
    eff_offsets: list[int] = []
    for _, e in events.iterrows():
        ann_pos = trading_days.get_indexer([e["announcement_date"]], method="bfill")[0]
        eff_pos = trading_days.get_indexer([e["effective_date"]],   method="bfill")[0]
        eff_offsets.append(int(eff_pos - ann_pos))
    eff_series = pd.Series(eff_offsets)
    print(f"  公告 → 生效 相對交易日分布：{eff_series.value_counts().to_dict()}")
    EFF_DAY = int(eff_series.mode().iloc[0])
    print(f"  Modal effective_day = {EFF_DAY}")

    # Event window for the chart (covers -30 day pre-ann to +30 day post-eff)
    PLOT_WINDOW = (-30, EFF_DAY + 30)
    print(f"  Plot window        : {PLOT_WINDOW}")

    # ── Aggregate CAR  (added) ─────────────────────────────────────────────
    print("\n[1/2] Computing CAR for ADDED stocks …")
    agg_added = aggregate_cars_across_events(
        events, prices_wide, market_returns,
        event_window=PLOT_WINDOW, stock_col="added_stocks",
    )

    plot_average_car(
        agg_added,
        save_path=FIG_DIR / "car_added_stocks_real.png",
        effective_day=EFF_DAY,
        title=("[REAL DATA] 平均 CAR — 00919 調入股 "
               f"（{len(events)} 個事件、71 個股票事件配對）"),
        label="調入股 平均 CAR",
    )
    print(f"  ✓ 儲存 → output/figures/car_added_stocks_real.png")

    # ── Aggregate CAR  (removed) ──────────────────────────────────────────
    print("\n[2/2] Computing CAR for REMOVED stocks …")
    agg_removed = aggregate_cars_across_events(
        events, prices_wide, market_returns,
        event_window=PLOT_WINDOW, stock_col="removed_stocks",
    )

    plot_average_car(
        agg_removed,
        save_path=FIG_DIR / "car_removed_stocks_real.png",
        effective_day=EFF_DAY,
        title=("[REAL DATA] 平均 CAR — 00919 調出股 "
               f"（{len(events)} 個事件、61 個股票事件配對）"),
        label="調出股 平均 CAR",
    )
    print(f"  ✓ 儲存 → output/figures/car_removed_stocks_real.png")

    # ── Multi-window t-tests ──────────────────────────────────────────────
    print(f"\n{'='*70}\n  Multi-window t-tests\n{'='*70}")

    windows = [
        ((-5, 0),                "[-5, 0]  公告前一週"),
        ((0, 5),                 "[0, +5]  公告後一週"),
        ((0, EFF_DAY),           f"[0, {EFF_DAY}]   公告→生效"),
        ((EFF_DAY, EFF_DAY+10),  f"[{EFF_DAY}, {EFF_DAY+10}]  生效後 10 交易日"),
        ((EFF_DAY, EFF_DAY+30),  f"[{EFF_DAY}, {EFF_DAY+30}]  生效後 30 交易日"),
    ]

    rows = []
    for w, name in windows:
        for stock_col in ["added_stocks", "removed_stocks"]:
            print(f"\n→ {name}  |  {stock_col}")
            r = t_test_car(events, prices_wide, market_returns,
                           window=w, stock_col=stock_col)
            if r.empty:
                continue
            rec = r.iloc[0].to_dict()
            rec["stock_col"]    = stock_col
            rec["window_label"] = name
            rows.append(rec)

    summary = pd.DataFrame(rows)[
        ["window_label", "stock_col", "window", "N",
         "mean_CAR", "std_CAR", "t_stat", "p_value"]
    ]
    out_csv = TBL_DIR / "car_summary_real.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n✓ Summary 寫入 {out_csv.relative_to(ROOT)}")

    # ── Plain-text summary ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  文字摘要")
    print("=" * 70)
    ann_to_eff = summary[
        (summary["window_label"].str.contains("公告→生效")) &
        (summary["stock_col"] == "added_stocks")
    ].iloc[0]
    print(f"\n【調入股 ANN → EFF 區間】")
    print(f"  視窗            : {ann_to_eff['window']}（trading days）")
    print(f"  N (event×stock) : {int(ann_to_eff['N'])}")
    print(f"  Mean CAR        : {ann_to_eff['mean_CAR']:+.4f}  "
          f"({ann_to_eff['mean_CAR']:.2%})")
    print(f"  t-statistic     : {ann_to_eff['t_stat']:+.4f}")
    print(f"  p-value         : {ann_to_eff['p_value']:.4f}")

    ann_to_eff_rm = summary[
        (summary["window_label"].str.contains("公告→生效")) &
        (summary["stock_col"] == "removed_stocks")
    ].iloc[0]
    print(f"\n【調出股 ANN → EFF 區間（對照）】")
    print(f"  視窗            : {ann_to_eff_rm['window']}")
    print(f"  N (event×stock) : {int(ann_to_eff_rm['N'])}")
    print(f"  Mean CAR        : {ann_to_eff_rm['mean_CAR']:+.4f}  "
          f"({ann_to_eff_rm['mean_CAR']:.2%})")
    print(f"  t-statistic     : {ann_to_eff_rm['t_stat']:+.4f}")
    print(f"  p-value         : {ann_to_eff_rm['p_value']:.4f}")


if __name__ == "__main__":
    main()
