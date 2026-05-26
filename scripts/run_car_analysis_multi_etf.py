"""
scripts/run_car_analysis_multi_etf.py
--------------------------------------
Multi-ETF CAR 事件研究
- AR = r_stock - r_TAIEX（市場調整模型，同 Stage 2）
- 事件視窗：[-30, 0]、[-5, 0]、[0, +5]、[+1, +11]、[+1, +31]
- has_additions=True 事件（35 個）
- is_structural_change=True（0056_2022Dec）含/排除兩個版本

輸出：
  output/figures/car_multi_etf_added_stocks.png   （4 panel：整體 + 各 ETF）
  output/figures/car_multi_etf_excl_structural.png（排除結構性擴張）
  output/tables/car_summary_multi_etf.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from event_study import (
    aggregate_cars_across_events,
    plot_average_car,
    compute_event_car,
)
from _plot_config import apply_chinese_style
apply_chinese_style()

warnings.filterwarnings("ignore")

FIG_DIR    = ROOT / "output" / "figures"
TBL_DIR    = ROOT / "output" / "tables"
EVENTS_CSV = ROOT / "data" / "processed" / "multi_etf_events.csv"
PRICES_PQ  = ROOT / "data" / "processed" / "multi_etf_stock_prices.parquet"

PLOT_WINDOW = (-30, 35)

TEST_WINDOWS = [
    ((-30, 0),  "[-30, 0]   公告前 30 日"),
    ((-5,  0),  "[-5,  0]   公告前一週"),
    ((0,  +5),  "[0,  +5]   公告後一週"),
    ((+1, +11), "[+1, +11]  生效後 10 日"),
    ((+1, +31), "[+1, +31]  生效後 30 日"),
]

# Stage 2（00919 單一 ETF）數值備查
STAGE2 = {
    "[-30, 0]   公告前 30 日": dict(CAR=+0.0486, t=+4.33, p="<0.0001", sig="***", N=71),
    "[-5,  0]   公告前一週":   dict(CAR=+0.0238, t=+4.16, p="0.0001",  sig="***", N=71),
    "[0,  +5]   公告後一週":   dict(CAR=-0.0237, t=-2.77, p="0.007",   sig="***", N=71),
    "[+1, +11]  生效後 10 日": dict(CAR=-0.0328, t=-3.48, p="0.0009",  sig="***", N=71),
    "[+1, +31]  生效後 30 日": dict(CAR=-0.0742, t=-5.54, p="<0.0001", sig="***", N=71),
}


# ── Data loading ────────────────────────────────────────────────────────────
def load_multi_events():
    """Load multi_etf_events.csv; parse added/removed_stocks to lists."""
    df = pd.read_csv(EVENTS_CSV, dtype={"event_id": str, "etf_code": str})
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])
    df["effective_date"]    = pd.to_datetime(df["effective_date"])

    def parse_stocks(s):
        if pd.isna(s) or str(s).strip() == "":
            return []
        return [x.strip() for x in str(s).split("|") if x.strip()]

    df["added_stocks"]   = df["added_stocks"].apply(parse_stocks)
    df["removed_stocks"] = df["removed_stocks"].apply(parse_stocks)
    return df


# ── Statistical tests ────────────────────────────────────────────────────────
def sig_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return "—"


def t_test_events(events, prices_wide, market_returns, window, stock_col="added_stocks"):
    """One-sample t-test on per-(event, stock) terminal CAR for a window."""
    car_values = []
    for _, ev in events.iterrows():
        tickers = ev[stock_col]
        for sid in tickers:
            if sid not in prices_wide.columns:
                continue
            try:
                car_df = compute_event_car(
                    stock_id=sid,
                    event_announcement_date=ev["announcement_date"],
                    prices_wide=prices_wide,
                    market_returns=market_returns,
                    event_window=window,
                )
            except (ValueError, KeyError):
                continue
            if car_df.empty:
                continue
            car_values.append(float(car_df["CAR"].iloc[-1]))

    n = len(car_values)
    if n < 2:
        return None
    arr = np.array(car_values)
    t_stat, p_value = stats.ttest_1samp(arr, popmean=0)
    return {
        "N":       n,
        "mean_CAR": float(arr.mean()),
        "std_CAR":  float(arr.std(ddof=1)),
        "t_stat":   float(t_stat),
        "p_value":  float(p_value),
        "sig":      sig_stars(float(p_value)),
    }


def run_test_battery(label, events, prices_wide, market_returns):
    """Run all TEST_WINDOWS for one event set; return list of row-dicts."""
    rows = []
    for (window, wlabel) in TEST_WINDOWS:
        r = t_test_events(events, prices_wide, market_returns, window)
        if r is None:
            continue
        rows.append({"group": label, "window": wlabel, **r})
    return rows


# ── Effective-day helper ─────────────────────────────────────────────────────
def modal_eff_day(events, trading_days):
    """Modal announcement → effective offset in trading days."""
    offsets = []
    for _, ev in events.iterrows():
        ann_pos = trading_days.get_indexer(
            [pd.Timestamp(ev["announcement_date"])], method="bfill"
        )[0]
        eff_pos = trading_days.get_indexer(
            [pd.Timestamp(ev["effective_date"])], method="bfill"
        )[0]
        if ann_pos >= 0 and eff_pos >= 0:
            offsets.append(int(eff_pos - ann_pos))
    if not offsets:
        return 1
    return int(pd.Series(offsets).mode().iloc[0])


# ── 4-panel CAR chart ────────────────────────────────────────────────────────
def plot_4panel(agg_all, agg_by_etf, events_hadd, trading_days, out_path):
    """2×2 panel: Overall + per-ETF CAR."""
    ETF_COLORS = {
        "整體": "#1f77b4",
        "0056": "#d62728",
        "00713": "#2ca02c",
        "00919": "#ff7f0e",
    }
    panels = [
        ("整體", agg_all,              events_hadd),
        ("0056",  agg_by_etf["0056"],  events_hadd[events_hadd["etf_code"] == "0056"]),
        ("00713", agg_by_etf["00713"], events_hadd[events_hadd["etf_code"] == "00713"]),
        ("00919", agg_by_etf["00919"], events_hadd[events_hadd["etf_code"] == "00919"]),
    ]
    n_events = {
        "整體": len(events_hadd),
        "0056":  (events_hadd["etf_code"] == "0056").sum(),
        "00713": (events_hadd["etf_code"] == "00713").sum(),
        "00919": (events_hadd["etf_code"] == "00919").sum(),
    }
    n_stocks = {
        "整體": events_hadd["added_stocks"].apply(len).sum(),
        "0056":  events_hadd[events_hadd["etf_code"] == "0056"]["added_stocks"].apply(len).sum(),
        "00713": events_hadd[events_hadd["etf_code"] == "00713"]["added_stocks"].apply(len).sum(),
        "00919": events_hadd[events_hadd["etf_code"] == "00919"]["added_stocks"].apply(len).sum(),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, (key, agg_df, ev_sub) in zip(axes.flat, panels):
        if agg_df is None or agg_df.empty:
            ax.text(0.5, 0.5, "資料不足", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(key)
            continue

        color = ETF_COLORS[key]
        df = agg_df.sort_values("relative_day").copy()
        df["ci95"] = 1.96 * df["std_CAR"] / np.sqrt(df["N"].clip(lower=1))

        ax.fill_between(df["relative_day"],
                        df["mean_CAR"] - df["ci95"],
                        df["mean_CAR"] + df["ci95"],
                        alpha=0.18, color=color)
        ax.plot(df["relative_day"], df["mean_CAR"],
                color=color, linewidth=1.8, label="平均 CAR")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axvline(0, color="#d62728", linewidth=1.2, linestyle="--",
                   alpha=0.8, label="公告日（T=0）")

        eff = modal_eff_day(ev_sub, trading_days)
        ax.axvline(eff, color="#ff7f0e", linewidth=1.2, linestyle="--",
                   alpha=0.8, label=f"生效日（T=+{eff}）")

        ne = n_events[key]
        ns = n_stocks[key]
        title_str = {
            "整體": f"整體（{ne} 個事件、{ns} 個配對）",
            "0056": f"0056（{ne} 個事件、{ns} 個配對）",
            "00713": f"00713（{ne} 個事件、{ns} 個配對）",
            "00919": f"00919（{ne} 個事件、{ns} 個配對）",
        }[key]
        ax.set_title(title_str, fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        ax.set_xlabel("相對公告日（交易日）", fontsize=9)
        ax.set_ylabel("平均 CAR", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        n_obs = f"{df['N'].min()}–{df['N'].max()}"
        ax.text(0.02, 0.02, f"觀測值/日：{n_obs}",
                transform=ax.transAxes, fontsize=7, color="gray")

    fig.suptitle("Multi-ETF 調入股平均 CAR（市場調整模型，AR = r − r_TAIEX）",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ 儲存 → {out_path.relative_to(ROOT)}")


# ── Print helpers ────────────────────────────────────────────────────────────
def print_stat_table(results):
    print("\n" + "=" * 88)
    print("  Multi-ETF CAR 統計檢定表（調入股，AR = r_stock − r_TAIEX）")
    print("=" * 88)
    print(f"  顯著性：*** p<0.01  ** p<0.05  * p<0.10  — 不顯著")
    print()

    groups = results["group"].unique()
    for grp in groups:
        sub = results[results["group"] == grp]
        print(f"  ┌─ {grp} {'─'*(60-len(grp))}")
        print(f"  │  {'視窗':<26} {'N':>5} {'Mean CAR':>10} {'t-stat':>8} {'p-value':>8} {'sig':>5}")
        print(f"  │  {'─'*65}")
        for _, r in sub.iterrows():
            print(f"  │  {r['window']:<26} {int(r['N']):>5} "
                  f"{r['mean_CAR']:>+9.2%} {r['t_stat']:>+8.3f} "
                  f"{r['p_value']:>8.4f} {r['sig']:>5}")
        print()


def print_comparison_table(results):
    print("=" * 88)
    print("  與 Stage 2（00919 單一 ETF）對比")
    print("=" * 88)
    print(f"  {'視窗':<26} {'Stage2 CAR':>11} {'sig':>4}  "
          f"{'Multi-ETF CAR':>14} {'sig':>4}  "
          f"{'方向一致':>8}")
    print("  " + "─" * 75)

    me = results[results["group"] == "整體（35 事件）"].set_index("window")
    for wlabel, s2 in STAGE2.items():
        if wlabel in me.index:
            r = me.loc[wlabel]
            me_car = r["mean_CAR"]
            me_sig = r["sig"]
            same_dir = "✓" if (s2["CAR"] * me_car > 0) else "✗"
        else:
            me_car = np.nan
            me_sig = "—"
            same_dir = "?"
        s2_car_str = f"{s2['CAR']:+.2%}"
        me_car_str = f"{me_car:+.2%}" if not np.isnan(me_car) else "N/A"
        print(f"  {wlabel:<26} {s2_car_str:>11} {s2['sig']:>4}  "
              f"{me_car_str:>14} {me_sig:>4}  {same_dir:>8}")

    print()
    print("  Stage 2：00919 7 個事件（僅公告前 1 個事件已確認成分），71 個配對")
    print("  Multi-ETF：35 個事件（has_additions=True），含 0056/00713/00919")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Multi-ETF CAR 事件研究")
    print("=" * 70)

    # Load prices
    prices_long = pd.read_parquet(PRICES_PQ)
    prices_wide = (
        prices_long
        .pivot(index="date", columns="stock_id", values="close")
        .sort_index()
    )
    if "TAIEX" not in prices_wide.columns:
        raise RuntimeError("TAIEX not in prices — 無法計算市場調整 AR")
    market_returns = prices_wide["TAIEX"].pct_change()
    trading_days   = prices_wide.index

    print(f"Price panel : {prices_wide.shape[0]} 交易日 × {prices_wide.shape[1]} 股票")
    print(f"日期範圍    : {prices_wide.index.min().date()} → {prices_wide.index.max().date()}")

    # Load events
    events_all    = load_multi_events()
    events_hadd   = events_all[events_all["has_additions"] == True].copy()
    events_no_str = events_hadd[~events_hadd["is_structural_change"]].copy()

    print(f"\n事件統計：")
    print(f"  全部    : {len(events_all)} 事件")
    print(f"  has_additions=True : {len(events_hadd)} 事件")
    print(f"  排除 structural    : {len(events_no_str)} 事件")
    for etf in ["0056", "00713", "00919"]:
        ev = events_hadd[events_hadd["etf_code"] == etf]
        n_pos = ev["added_stocks"].apply(len).sum()
        print(f"    {etf}: {len(ev)} 事件  {n_pos} 調入配對  "
              f"（生效日 modal +{modal_eff_day(ev, trading_days)} 交易日）")

    # ── [1] Overall CAR ────────────────────────────────────────────────────
    print("\n[1/5] Overall CAR（35 事件）…")
    agg_all = aggregate_cars_across_events(
        events_hadd, prices_wide, market_returns,
        event_window=PLOT_WINDOW, stock_col="added_stocks",
    )

    # ── [2] Per-ETF CAR ────────────────────────────────────────────────────
    print("[2/5] Per-ETF CAR …")
    agg_by_etf = {}
    for etf in ["0056", "00713", "00919"]:
        ev_sub = events_hadd[events_hadd["etf_code"] == etf]
        agg_by_etf[etf] = aggregate_cars_across_events(
            ev_sub, prices_wide, market_returns,
            event_window=PLOT_WINDOW, stock_col="added_stocks",
        )
        print(f"  {etf}: done")

    # ── [3] Overall excl. structural ──────────────────────────────────────
    print("[3/5] Overall excl. 0056_2022Dec（34 事件）…")
    agg_no_str = aggregate_cars_across_events(
        events_no_str, prices_wide, market_returns,
        event_window=PLOT_WINDOW, stock_col="added_stocks",
    )

    # ── [4] Plots ──────────────────────────────────────────────────────────
    print("[4/5] Saving plots …")
    plot_4panel(
        agg_all, agg_by_etf, events_hadd, trading_days,
        FIG_DIR / "car_multi_etf_added_stocks.png",
    )
    eff_day_all = modal_eff_day(events_hadd, trading_days)
    plot_average_car(
        agg_no_str,
        save_path=FIG_DIR / "car_multi_etf_excl_structural.png",
        effective_day=eff_day_all,
        title="Multi-ETF 調入股 CAR（排除 0056_2022Dec 結構性擴張，34 事件）",
        label="調入股 平均 CAR（34 事件）",
    )
    print(f"  ✓ 儲存 → output/figures/car_multi_etf_excl_structural.png")

    # ── [5] Statistical tests ──────────────────────────────────────────────
    print("[5/5] Statistical tests …")
    all_rows = []
    all_rows += run_test_battery("整體（35 事件）",        events_hadd,   prices_wide, market_returns)
    all_rows += run_test_battery("整體（排除 2022Dec，34 事件）", events_no_str, prices_wide, market_returns)
    for etf in ["0056", "00713", "00919"]:
        ev_sub = events_hadd[events_hadd["etf_code"] == etf]
        all_rows += run_test_battery(etf, ev_sub, prices_wide, market_returns)

    results = pd.DataFrame(all_rows)
    results.to_csv(TBL_DIR / "car_summary_multi_etf.csv", index=False)
    print(f"  ✓ 儲存 → output/tables/car_summary_multi_etf.csv")

    # ── Print ──────────────────────────────────────────────────────────────
    print_stat_table(results)
    print_comparison_table(results)
    print("\n  完成。等待確認後再進行下一步。\n")


if __name__ == "__main__":
    main()
