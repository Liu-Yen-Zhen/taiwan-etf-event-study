"""
scripts/run_car_analysis_v2.py
------------------------------
重跑階段 2 CAR 分析，使用清理過的 stock_prices.parquet（OHLC=0 → NaN）。

產出
----
  output/tables/car_summary_real_v2.csv
  output/figures/car_added_stocks_real_v2.png
  output/figures/car_removed_stocks_real_v2.png

並印出 v1 vs v2 對照表（讀取 output/tables/car_summary_real.csv 作為 v1）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np
from data_fetcher import load_events
from event_study import (
    aggregate_cars_across_events,
    plot_average_car,
    t_test_car,
)

PARQUET = ROOT / "data" / "processed" / "stock_prices.parquet"
EVENTS  = ROOT / "data" / "raw" / "events.csv"
V1_CSV  = ROOT / "output" / "tables"  / "car_summary_real.csv"
V2_CSV  = ROOT / "output" / "tables"  / "car_summary_real_v2.csv"
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ── Load cleaned panel ────────────────────────────────────────────────
    prices_long = pd.read_parquet(PARQUET)
    prices_wide = (prices_long
                   .pivot(index="date", columns="stock_id", values="close")
                   .sort_index())
    market_returns = prices_wide["TAIEX"].pct_change()
    events = load_events(EVENTS)

    print(f"Loaded cleaned panel: {prices_wide.shape}  events={len(events)}")
    n_nan_close = int(prices_wide.isna().sum().sum())
    print(f"  close NaN cells: {n_nan_close} (was 0 before cleanup)")

    # Effective day in trading-day space (we confirmed = 1 for all 7 events)
    trading_days = prices_wide.index
    eff_offsets = []
    for _, e in events.iterrows():
        a = trading_days.get_indexer([e["announcement_date"]], method="bfill")[0]
        f = trading_days.get_indexer([e["effective_date"]],   method="bfill")[0]
        eff_offsets.append(int(f - a))
    EFF_DAY = int(pd.Series(eff_offsets).mode().iloc[0])
    PLOT_WINDOW = (-30, EFF_DAY + 30)

    # ── Aggregate CAR for plotting ────────────────────────────────────────
    print("\n[1/2] Aggregating CAR (added stocks)…")
    agg_added = aggregate_cars_across_events(
        events, prices_wide, market_returns,
        event_window=PLOT_WINDOW, stock_col="added_stocks",
    )
    plot_average_car(
        agg_added,
        save_path=FIG_DIR / "car_added_stocks_real_v2.png",
        effective_day=EFF_DAY,
        title=("[REAL DATA — v2 cleaned] 平均 CAR — 00919 調入股 "
               f"（{len(events)} 個事件、71 個股票事件配對）"),
        label="調入股 平均 CAR",
    )
    print(f"  ✓ {FIG_DIR / 'car_added_stocks_real_v2.png'}")

    print("\n[2/2] Aggregating CAR (removed stocks)…")
    agg_removed = aggregate_cars_across_events(
        events, prices_wide, market_returns,
        event_window=PLOT_WINDOW, stock_col="removed_stocks",
    )
    plot_average_car(
        agg_removed,
        save_path=FIG_DIR / "car_removed_stocks_real_v2.png",
        effective_day=EFF_DAY,
        title=("[REAL DATA — v2 cleaned] 平均 CAR — 00919 調出股 "
               f"（{len(events)} 個事件、61 個股票事件配對）"),
        label="調出股 平均 CAR",
    )
    print(f"  ✓ {FIG_DIR / 'car_removed_stocks_real_v2.png'}")

    # ── Multi-window t-tests ──────────────────────────────────────────────
    windows = [
        ((-30, 0),               "[-30, 0]  公告前 30 日"),
        ((-5, 0),                "[-5, 0]   公告前一週"),
        ((0, 5),                 "[0, +5]   公告後一週"),
        ((0, EFF_DAY),           f"[0, {EFF_DAY}]    公告→生效"),
        ((EFF_DAY, EFF_DAY+10),  f"[{EFF_DAY}, {EFF_DAY+10}]   生效後 10 日"),
        ((EFF_DAY, EFF_DAY+30),  f"[{EFF_DAY}, {EFF_DAY+30}]   生效後 30 日"),
    ]

    rows_v2 = []
    for w, name in windows:
        for stock_col in ["added_stocks", "removed_stocks"]:
            r = t_test_car(events, prices_wide, market_returns,
                           window=w, stock_col=stock_col)
            if r.empty:
                continue
            rec = r.iloc[0].to_dict()
            rec["stock_col"]    = stock_col
            rec["window_label"] = name
            rows_v2.append(rec)

    v2 = pd.DataFrame(rows_v2)[
        ["window_label", "stock_col", "window", "N",
         "mean_CAR", "std_CAR", "t_stat", "p_value"]
    ]
    v2.to_csv(V2_CSV, index=False)
    print(f"\n✓ v2 summary → {V2_CSV.relative_to(ROOT)}")

    # ── v1 vs v2 comparison table ─────────────────────────────────────────
    print("\n" + "=" * 102)
    print("  v1 (有 close=0 artifact) vs v2 (close=0 → NaN 清理後) 對照")
    print("=" * 102)

    v1 = pd.read_csv(V1_CSV)
    v1["key"] = v1["window_label"].str.replace(r"\s+", " ", regex=True).str.strip() + "|" + v1["stock_col"]
    v2["key"] = v2["window_label"].str.replace(r"\s+", " ", regex=True).str.strip() + "|" + v2["stock_col"]

    # Normalise keys for cross-version match (just by window numbers + stock_col)
    def short_key(w):
        # w is like "[-30, 0]" — normalise spaces
        return str(w).replace(" ", "")
    v1["short_key"] = v1["window"].map(short_key) + "|" + v1["stock_col"]
    v2["short_key"] = v2["window"].map(short_key) + "|" + v2["stock_col"]
    cmp = v1.merge(v2, on="short_key", suffixes=("_v1", "_v2"))

    # Build header
    print(f"  {'視窗':14s} {'群組':5s} {'N':>4s}   {'舊 CAR':>9s} {'新 CAR':>9s} {'Δ CAR':>8s}     "
          f"{'舊 t':>7s} {'新 t':>7s}    {'舊 p':>7s} {'新 p':>7s}    結論變化?")
    print(f"  {'-'*14} {'-'*5} {'-'*4}   {'-'*9} {'-'*9} {'-'*8}     "
          f"{'-'*7} {'-'*7}    {'-'*7} {'-'*7}    {'-'*16}")

    for _, r in cmp.iterrows():
        win = r["window_v1"]
        col = "調入" if r["stock_col_v1"] == "added_stocks" else "調出"
        n  = int(r["N_v1"])
        cv1 = r["mean_CAR_v1"]; cv2 = r["mean_CAR_v2"]
        d  = cv2 - cv1
        t1 = r["t_stat_v1"]; t2 = r["t_stat_v2"]
        p1 = r["p_value_v1"]; p2 = r["p_value_v2"]

        def sig_label(p):
            return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "—"

        sig1 = sig_label(p1); sig2 = sig_label(p2)
        if sig1 == sig2:
            conclusion = f"無變化 ({sig1})"
        else:
            conclusion = f"{sig1} → {sig2}"

        print(f"  {win:14s} {col:5s} {n:>4d}   {cv1:>+8.2%} {cv2:>+8.2%} {d:>+7.2%}     "
              f"{t1:>+7.2f} {t2:>+7.2f}    {p1:>7.4f} {p2:>7.4f}    {conclusion}")

    # ── Key window summary ────────────────────────────────────────────────
    main_row = cmp[(cmp["short_key"] == "[1,31]|added_stocks")].iloc[0]
    print("\n" + "=" * 70)
    print("  最關鍵視窗摘要：[+1, +31] 調入股 CAR_post_30")
    print("=" * 70)
    print(f"  舊版 CAR : {main_row['mean_CAR_v1']:+.4f}  ({main_row['mean_CAR_v1']:+.2%})")
    print(f"  新版 CAR : {main_row['mean_CAR_v2']:+.4f}  ({main_row['mean_CAR_v2']:+.2%})")
    print(f"  變動     : {main_row['mean_CAR_v2'] - main_row['mean_CAR_v1']:+.4f}  "
          f"({(main_row['mean_CAR_v2'] - main_row['mean_CAR_v1']):+.2%})")
    print(f"  舊 t / p : {main_row['t_stat_v1']:+.4f} / {main_row['p_value_v1']:.6f}")
    print(f"  新 t / p : {main_row['t_stat_v2']:+.4f} / {main_row['p_value_v2']:.6f}")


if __name__ == "__main__":
    main()
