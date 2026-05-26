"""
scripts/run_backtest.py
------------------------
Stage 5：假設策略回測 [HYPOTHETICAL — 僅 3 個 OOS 事件]

訊號：XGBoost 預測機率（重現 Stage 4 相同設定）
買入日：T-14（公告日前 14 個交易日）收盤價
賣出日：公告日 + 1 個交易日收盤價
K 值：5, 8, 10

成本情境：
  1. 無成本（假設上界）
  2. 手續費 + 證交稅（0.285% + 0.3% = 0.585%）
  3. 情境 2 + 雙邊滑點估計 20bps（0.4%，合計 0.985%）

絕對限制：
- 只用 test events 5-7（00919_20241217, 00919_20250603, 00919_20251216）
- 不含任何合成或隨機模擬資料用於報酬計算
- 隨機基準僅用於 bootstrap 比較，不參與策略評估
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────
PANEL_PQ      = ROOT / "data"   / "processed" / "prediction_panel.parquet"
PRICES_PQ     = ROOT / "data"   / "processed" / "stock_prices.parquet"
OUT_TABLE_DIR = ROOT / "output" / "tables"
OUT_FIG_DIR   = ROOT / "output" / "figures"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
TRAIN_EVENTS = ["00919_20230531", "00919_20231218", "00919_20240531"]
TEST_EVENTS  = ["00919_20241217", "00919_20250603", "00919_20251216"]
K_VALUES     = [5, 8, 10]
N_BOOTSTRAP  = 1000
RNG_SEED     = 42

COST_SCENARIOS = {
    "no_cost":        0.0,
    "fee_tax":        0.00585,   # 0.285% 手續費 + 0.3% 證交稅
    "fee_tax_slippage": 0.00985, # 0.585% + 0.4% 雙邊滑點 (20bps × 2)
}

# Stage 4 same feature lists
CONT_FEATURES = [
    "dividend_yield_ttm",
    "dividend_yield_rank_in_pool",
    "log_market_cap",
    "market_cap_rank_in_pool",
    "log_avg_turnover_60d",
    "turnover_ratio_60d",
    "volatility_60d",
    "beta_60d",
    "momentum_60d_pre",
]
BIN_FEATURES = ["was_in_00919_previous"]
ALL_FEATURES = CONT_FEATURES + BIN_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
# 1. Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    df = pd.read_parquet(PANEL_PQ)
    df["was_in_00919_previous"] = df["was_in_00919_previous"].fillna(0)
    df = df.dropna(subset=["market_cap"])
    return df


def get_trading_calendar(prices: pd.DataFrame) -> pd.Series:
    taiex = prices[prices["stock_id"] == "TAIEX"][["date"]].sort_values("date")
    return taiex["date"].reset_index(drop=True)


def next_trading_day(ann_date: pd.Timestamp, calendar: pd.Series) -> pd.Timestamp:
    """First trading day strictly after ann_date."""
    after = calendar[calendar > ann_date]
    if len(after) == 0:
        raise ValueError(f"No trading day after {ann_date}")
    return after.iloc[0]


def get_close_price(prices: pd.DataFrame, stock_id: str,
                    date: pd.Timestamp) -> float:
    row = prices[(prices["stock_id"] == stock_id) & (prices["date"] == date)]
    if row.empty:
        raise ValueError(f"No close price for {stock_id} on {date.date()}")
    close = row["close"].iloc[0]
    if pd.isna(close) or close == 0:
        raise ValueError(f"Invalid close price ({close}) for {stock_id} on {date.date()}")
    return float(close)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Re-train XGBoost (same hyperparams as Stage 4)
# ═══════════════════════════════════════════════════════════════════════════

def train_xgb_and_predict(df: pd.DataFrame):
    """Returns test_df with column '_xgb_proba'."""
    train_df = df[df["event_id"].isin(TRAIN_EVENTS)].copy()
    test_df  = df[df["event_id"].isin(TEST_EVENTS)].copy()

    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(train_df[CONT_FEATURES])
    X_test_cont  = scaler.transform(test_df[CONT_FEATURES])

    X_train = np.hstack([X_train_cont, train_df[BIN_FEATURES].values])
    X_test  = np.hstack([X_test_cont,  test_df[BIN_FEATURES].values])
    y_train = train_df["y"].values

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        scale_pos_weight=9.5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    proba = xgb.predict_proba(X_test)[:, 1]

    test_df = test_df.copy()
    test_df["_xgb_proba"] = proba
    print(f"XGBoost re-trained: train {len(train_df)} rows, "
          f"test {len(test_df)} rows")
    return test_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. Compute stock-level returns
# ═══════════════════════════════════════════════════════════════════════════

def compute_stock_returns(test_df: pd.DataFrame,
                          prices: pd.DataFrame,
                          calendar: pd.Series) -> pd.DataFrame:
    """
    For each stock in test events:
      gross_return = (close_at_ann+1 - close_at_t14) / close_at_t14
    """
    records = []
    for ev in TEST_EVENTS:
        sub = test_df[test_df["event_id"] == ev].copy()
        t14     = pd.to_datetime(sub["t14_date"].iloc[0])
        ann     = pd.to_datetime(sub["ann_date"].iloc[0])
        ann_p1  = next_trading_day(ann, calendar)

        print(f"\n  [{ev}] t14={t14.date()}  ann={ann.date()}  "
              f"sell={ann_p1.date()}  pool={len(sub)}")

        for _, row in sub.iterrows():
            sid = row["stock_id"]
            try:
                buy  = get_close_price(prices, sid, t14)
                sell = get_close_price(prices, sid, ann_p1)
                gross = (sell - buy) / buy
            except ValueError as e:
                print(f"    WARNING: {e} — skipping {sid}")
                continue

            records.append({
                "event_id":    ev,
                "ann_date":    ann,
                "t14_date":    t14,
                "sell_date":   ann_p1,
                "stock_id":    sid,
                "y":           int(row["y"]),
                "_xgb_proba":  row["_xgb_proba"],
                "buy_price":   buy,
                "sell_price":  sell,
                "gross_return": gross,
            })

    ret_df = pd.DataFrame(records)
    print(f"\n  Stock-return records: {len(ret_df)} "
          f"({ret_df['y'].sum()} positives)")
    return ret_df


# ═══════════════════════════════════════════════════════════════════════════
# 4. Strategy: top-K by XGB proba
# ═══════════════════════════════════════════════════════════════════════════

def strategy_event_return(ev_df: pd.DataFrame, k: int) -> dict:
    """
    Select top-K stocks by XGB proba within a single event.
    Return dict with event-level stats.
    """
    ranked = ev_df.sort_values("_xgb_proba", ascending=False)
    top_k  = ranked.head(k)

    n_correct    = int(top_k["y"].sum())
    gross_avg    = float(top_k["gross_return"].mean())
    individual_r = top_k["gross_return"].tolist()

    return {
        "n_predicted_correct": n_correct,
        "gross_event_return":  gross_avg,
        "individual_returns":  individual_r,
        "top_k_stocks":        top_k["stock_id"].tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_random_baseline(ret_df: pd.DataFrame,
                               k: int, n_iter: int, rng) -> np.ndarray:
    """
    1000 × average-of-3-events random-top-K gross returns.
    Returns array shape (n_iter,).
    """
    iter_returns = []
    for _ in range(n_iter):
        ev_rets = []
        for ev in TEST_EVENTS:
            ev_df = ret_df[ret_df["event_id"] == ev]
            if len(ev_df) < k:
                chosen_r = ev_df["gross_return"].values
            else:
                chosen_r = rng.choice(ev_df["gross_return"].values, k, replace=False)
            ev_rets.append(chosen_r.mean())
        iter_returns.append(np.mean(ev_rets))
    return np.array(iter_returns)


def perfect_prediction_return(ret_df: pd.DataFrame) -> float:
    """
    Buy ALL actual positives in each test event at T-14, sell at ann+1.
    Event return = equal-weight avg of positives.
    Strategy return = equal-weight avg across 3 events.
    """
    ev_rets = []
    for ev in TEST_EVENTS:
        ev_df = ret_df[(ret_df["event_id"] == ev) & (ret_df["y"] == 1)]
        if ev_df.empty:
            continue
        ev_rets.append(float(ev_df["gross_return"].mean()))
    return float(np.mean(ev_rets))


# ═══════════════════════════════════════════════════════════════════════════
# 6. Build result tables
# ═══════════════════════════════════════════════════════════════════════════

def build_results(ret_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_detail  = []   # per-event rows → backtest_results.csv
    rows_summary = []   # per-k-scenario rows → backtest_summary.csv

    rng = np.random.default_rng(RNG_SEED)

    for k in K_VALUES:
        print(f"\n── K = {k} ──")

        # ── per-event gross returns ──────────────────────────────────────
        ev_gross = {}
        for ev in TEST_EVENTS:
            ev_df  = ret_df[ret_df["event_id"] == ev]
            result = strategy_event_return(ev_df, k)
            ev_gross[ev] = result["gross_event_return"]
            print(f"  {ev}: top-{k} correct={result['n_predicted_correct']}, "
                  f"gross={result['gross_event_return']:.4f}")

        # ── cost scenarios → per-event net returns ───────────────────────
        for scen_name, cost in COST_SCENARIOS.items():
            cumulative = 1.0
            for ev in TEST_EVENTS:
                ev_df  = ret_df[ret_df["event_id"] == ev]
                result = strategy_event_return(ev_df, k)
                net    = result["gross_event_return"] - cost

                rows_detail.append({
                    "k":                   k,
                    "scenario":            scen_name,
                    "event_id":            ev,
                    "n_predicted_correct": result["n_predicted_correct"],
                    "gross_event_return":  round(result["gross_event_return"], 6),
                    "cost":                round(cost, 6),
                    "net_event_return":    round(net, 6),
                    "cost_bps":            round(cost * 10000, 1),
                })
                cumulative *= (1 + net)

            net_event_rets = [
                strategy_event_return(ret_df[ret_df["event_id"] == ev], k)[
                    "gross_event_return"
                ] - cost
                for ev in TEST_EVENTS
            ]
            avg_ret   = float(np.mean(net_event_rets))
            std_ret   = float(np.std(net_event_rets, ddof=1))
            worst_ret = float(np.min(net_event_rets))
            cum_ret   = float(np.prod([1 + r for r in net_event_rets]) - 1)

            # bootstrap random baseline (gross, no cost deducted)
            boot = bootstrap_random_baseline(ret_df, k, N_BOOTSTRAP, rng)

            rows_summary.append({
                "k":               k,
                "scenario":        scen_name,
                "cost_bps":        round(cost * 10000, 1),
                "avg_net_return":  round(avg_ret, 6),
                "std_net_return":  round(std_ret, 6),
                "worst_event_net": round(worst_ret, 6),
                "cum_3event_net":  round(cum_ret, 6),
                "random_boot_median_gross": round(float(np.median(boot)), 6),
                "random_boot_p10_gross":   round(float(np.percentile(boot, 10)), 6),
                "random_boot_p90_gross":   round(float(np.percentile(boot, 90)), 6),
            })
            print(f"    [{scen_name}] avg_net={avg_ret:.4f}  "
                  f"std={std_ret:.4f}  cum={cum_ret:.4f}")

        # perfect prediction once per K
        perf = perfect_prediction_return(ret_df)
        print(f"  Perfect-pred avg gross: {perf:.4f}")

    detail_df  = pd.DataFrame(rows_detail)
    summary_df = pd.DataFrame(rows_summary)

    # add perfect_pred_gross to summary (same for all scenarios of same K)
    # compute once per K
    perf_by_k = {}
    for k in K_VALUES:
        perf_by_k[k] = perfect_prediction_return(ret_df)
    summary_df["perfect_pred_gross"] = summary_df["k"].map(perf_by_k)

    return detail_df, summary_df


# ═══════════════════════════════════════════════════════════════════════════
# 7. Figure
# ═══════════════════════════════════════════════════════════════════════════

SCEN_LABELS = {
    "no_cost":           "No Cost",
    "fee_tax":           "Fee+Tax\n(58.5bps)",
    "fee_tax_slippage":  "Fee+Tax+Slip\n(98.5bps)",
}
SCEN_COLORS = {
    "no_cost":           "#4C72B0",
    "fee_tax":           "#55A868",
    "fee_tax_slippage":  "#C44E52",
}
PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linestyle":   "--",
    "font.family":      "sans-serif",
}


def plot_backtest_comparison(summary_df: pd.DataFrame,
                             ret_df: pd.DataFrame,
                             out_path: Path) -> None:
    """
    Bar chart: 3 K values × 3 scenarios (avg net return).
    Horizontal lines: bootstrap random median (no cost) + perfect prediction.
    """
    k_vals    = K_VALUES
    scenarios = list(COST_SCENARIOS.keys())
    n_k       = len(k_vals)
    n_scen    = len(scenarios)

    bar_width  = 0.22
    group_gap  = 0.10
    group_width = n_scen * bar_width + group_gap

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))

        x_centers = np.arange(n_k) * group_width

        for si, scen in enumerate(scenarios):
            offsets = (si - n_scen / 2 + 0.5) * bar_width
            for ki, k in enumerate(k_vals):
                row = summary_df[(summary_df["k"] == k) &
                                 (summary_df["scenario"] == scen)].iloc[0]
                height = row["avg_net_return"]
                color  = SCEN_COLORS[scen]

                bar = ax.bar(x_centers[ki] + offsets, height,
                             width=bar_width * 0.9,
                             color=color,
                             edgecolor="white",
                             alpha=0.88,
                             label=SCEN_LABELS[scen] if ki == 0 else "")

                # value label on bar
                va = "bottom" if height >= 0 else "top"
                ypos = height + 0.001 * (1 if height >= 0 else -1)
                ax.text(x_centers[ki] + offsets, ypos,
                        f"{height*100:.1f}%",
                        ha="center", va=va, fontsize=7.5, color="#333")

        # ── benchmark lines (no-cost scenario, K=10) ──────────────────
        # random bootstrap median (gross, no cost)
        rand_row  = summary_df[(summary_df["k"] == 10) &
                               (summary_df["scenario"] == "no_cost")].iloc[0]
        rand_med  = rand_row["random_boot_median_gross"]
        rand_p10  = rand_row["random_boot_p10_gross"]
        rand_p90  = rand_row["random_boot_p90_gross"]

        # perfect prediction
        perf = perfect_prediction_return(ret_df)

        ax.axhline(rand_med, color="#888888", linestyle=":", lw=1.6,
                   label=f"Random baseline (median gross, K=10): {rand_med*100:.1f}%")
        ax.axhline(perf, color="#FF8800", linestyle="--", lw=1.8,
                   label=f"Perfect prediction (gross avg): {perf*100:.1f}%")
        ax.axhline(0, color="black", lw=0.8)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)

        # ── formatting ─────────────────────────────────────────────────
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"Top-{k}" for k in k_vals], fontsize=12)
        ax.set_ylabel("Avg Event Return (3 events, equal-weight)", fontsize=11)
        ax.set_title(
            "HYPOTHETICAL Strategy Backtest — XGBoost Signal\n"
            "(Buy T−14 close / Sell Ann+1 close, OOS Events 5-7 only, N=3)",
            fontsize=12, pad=12
        )

        # legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=8.5, loc="upper right",
                  framealpha=0.9)

        # annotation box
        ax.text(0.01, 0.02,
                "⚠ HYPOTHETICAL | 3 events only | No market impact | Long-only",
                transform=ax.transAxes, fontsize=8,
                color="#CC0000", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0",
                          edgecolor="#CC0000", alpha=0.8))

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Stage 5: HYPOTHETICAL Strategy Backtest")
    print("  [Signal: XGBoost top-K | Buy T-14 | Sell Ann+1]")
    print("=" * 65)

    # ── Load ─────────────────────────────────────────────────────────────
    df     = load_and_prepare()
    prices = pd.read_parquet(PRICES_PQ)
    prices["date"] = pd.to_datetime(prices["date"])
    calendar = get_trading_calendar(prices)

    # ── Train XGBoost & predict ──────────────────────────────────────────
    test_df = train_xgb_and_predict(df)

    # ── Compute per-stock gross returns ──────────────────────────────────
    print("\n[Computing stock-level returns]")
    ret_df = compute_stock_returns(test_df, prices, calendar)

    # ── Print per-event positive stock returns ────────────────────────────
    print("\n[Actual positive stock gross returns (ann+1 / T-14 - 1)]")
    for ev in TEST_EVENTS:
        pos = ret_df[(ret_df["event_id"] == ev) & (ret_df["y"] == 1)]
        print(f"\n  {ev} positives ({len(pos)} stocks):")
        for _, r in pos.sort_values("_xgb_proba", ascending=False).iterrows():
            print(f"    {r['stock_id']}  proba={r['_xgb_proba']:.4f}  "
                  f"gross={r['gross_return']:+.4f}  "
                  f"buy={r['buy_price']:.2f}  sell={r['sell_price']:.2f}")

    # ── Build results ─────────────────────────────────────────────────────
    print("\n[Building result tables]")
    detail_df, summary_df = build_results(ret_df)

    # ── Save tables ───────────────────────────────────────────────────────
    detail_path  = OUT_TABLE_DIR / "backtest_results.csv"
    summary_path = OUT_TABLE_DIR / "backtest_summary.csv"
    detail_df.to_csv(detail_path,  index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved: {detail_path.relative_to(ROOT)}")
    print(f"  Saved: {summary_path.relative_to(ROOT)}")

    # ── Plot ──────────────────────────────────────────────────────────────
    print("\n[Generating figure]")
    plot_backtest_comparison(
        summary_df, ret_df,
        OUT_FIG_DIR / "backtest_comparison.png"
    )

    # ── Summary printout ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BACKTEST SUMMARY  [HYPOTHETICAL — 3 OOS events only]")
    print("=" * 65)

    perf = perfect_prediction_return(ret_df)
    print(f"  Perfect prediction (buy all positives) gross avg: "
          f"{perf*100:+.2f}%")

    for k in K_VALUES:
        print(f"\n  ── K = {k} ──")
        for scen_name, cost in COST_SCENARIOS.items():
            row = summary_df[(summary_df["k"] == k) &
                             (summary_df["scenario"] == scen_name)].iloc[0]
            print(f"    [{scen_name:20s}]  "
                  f"avg={row['avg_net_return']*100:+6.2f}%  "
                  f"std={row['std_net_return']*100:.2f}%  "
                  f"worst={row['worst_event_net']*100:+.2f}%  "
                  f"cum(3ev)={row['cum_3event_net']*100:+.2f}%")

    # random baseline for K=10
    rng_row = summary_df[(summary_df["k"] == 10) &
                         (summary_df["scenario"] == "no_cost")].iloc[0]
    print(f"\n  Random baseline (bootstrap median, K=10, gross): "
          f"{rng_row['random_boot_median_gross']*100:+.2f}%  "
          f"[p10={rng_row['random_boot_p10_gross']*100:.2f}%  "
          f"p90={rng_row['random_boot_p90_gross']*100:.2f}%]")

    print("\n" + "=" * 65)
    print("  ⚠ 限制宣告")
    print("=" * 65)
    print("  1. 僅 3 個 OOS 事件，統計不穩定（std 可能超過 avg）")
    print("  2. 買賣皆以收盤價假設完全成交（無市場衝擊模型）")
    print("  3. 無市場 beta 對沖（純多頭暴露）")
    print("  4. 資本假設等額部署於每事件（無複利滾動）")
    print("  5. 滑點 20bps 為估計值，實際視市值與成交量而定")

    print("\n  Stage 5 完成。等待審核後進 Stage 6。")


if __name__ == "__main__":
    main()
