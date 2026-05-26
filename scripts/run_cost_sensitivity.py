"""
scripts/run_cost_sensitivity.py
---------------------------------
Stage 6：成本敏感度分析

輸入：output/tables/backtest_results.csv（Stage 5 已產出）
      data/processed/prediction_panel.parquet（容量估計用）

不重跑任何模型或回測。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKTEST_CSV  = ROOT / "output" / "tables" / "backtest_results.csv"
PANEL_PQ      = ROOT / "data"   / "processed" / "prediction_panel.parquet"
OUT_TABLE_DIR = ROOT / "output" / "tables"
OUT_FIG_DIR   = ROOT / "output" / "figures"

TEST_EVENTS = ["00919_20241217", "00919_20250603", "00919_20251216"]

# Perfect-prediction ceilings (computed in Stage 5 supplementary checks)
PERFECT_T14_GROSS_BPS = 15.43   # +0.15% from Stage 5 main backtest
PERFECT_T5_GROSS_BPS  = -108.0  # -1.08% supplementary T-5 check
PERFECT_T7_MAR_BPS    = -148.0  # -1.48% supplementary T-7 MAR check

# Cost scenarios to display in sensitivity table
# columns: label, cost_bps
COST_ROWS = [
    ("無成本（假設上界）",              0.0),
    ("手續費 + 證交稅",                58.5),
    ("+ 滑點 20bps（雙邊）",           98.5),
    ("+ 滑點 50bps（雙邊）",          158.5),
    ("+ 滑點 100bps（雙邊）",         258.5),
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load backtest results & compute gross averages
# ═══════════════════════════════════════════════════════════════════════════

def load_gross_avg(k: int) -> float:
    """Average gross event return (no cost) across 3 test events for given K."""
    br = pd.read_csv(BACKTEST_CSV)
    rows = br[(br["k"] == k) & (br["scenario"] == "no_cost")]
    return float(rows["gross_event_return"].mean())


# ═══════════════════════════════════════════════════════════════════════════
# 2. Cost sensitivity table builder
# ═══════════════════════════════════════════════════════════════════════════

def build_cost_table(k: int, gross_avg: float,
                     perf_gross_bps: float) -> pd.DataFrame:
    """
    For a given K and gross average return, sweep cost scenarios.
    Also include perfect-prediction ceiling row.
    """
    rows = []
    for label, cost_bps in COST_ROWS:
        cost = cost_bps / 10_000
        net  = gross_avg - cost
        rows.append({
            "情境":         label,
            "成本 (bps)":   cost_bps,
            "假設平均事件報酬": round(gross_avg * 100, 3),
            "成本 (%)":     round(cost * 100, 3),
            "淨報酬 (%)":   round(net * 100, 3),
            "淨報酬 (bps)": round(net * 10_000, 1),
            "是否 > 0":     "✓" if net > 0 else "✗",
        })

    # Break-even row
    be_bps = gross_avg * 10_000
    rows.append({
        "情境":          f"Break-even（淨報酬 = 0）",
        "成本 (bps)":    round(be_bps, 1),
        "假設平均事件報酬": round(gross_avg * 100, 3),
        "成本 (%)":      round(gross_avg * 100, 3),
        "淨報酬 (%)":    0.0,
        "淨報酬 (bps)":  0.0,
        "是否 > 0":      "=",
    })

    # Perfect prediction ceiling row
    perf_gross = perf_gross_bps / 10_000
    perf_net_at_fee = perf_gross - 58.5 / 10_000
    rows.append({
        "情境":          "【完美預測 ceiling，手續費+稅】",
        "成本 (bps)":    58.5,
        "假設平均事件報酬": round(perf_gross * 100, 3),
        "成本 (%)":      round(58.5 / 100, 3),
        "淨報酬 (%)":    round(perf_net_at_fee * 100, 3),
        "淨報酬 (bps)":  round(perf_net_at_fee * 10_000, 1),
        "是否 > 0":      "✓" if perf_net_at_fee > 0 else "✗",
    })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Capacity estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_capacity(k: int, impact_pct: float = 0.10) -> dict:
    """
    For top-K stocks in test events:
    Capacity per stock = avg_turnover_60d × impact_pct
    Total capacity     = sum of per-stock capacity across K stocks
    Use median pool turnover as conservative proxy (don't need to know exact top-K).
    Also report percentile breakpoints.
    """
    panel = pd.read_parquet(PANEL_PQ)
    test  = panel[panel["event_id"].isin(TEST_EVENTS)].copy()

    # Per-event capacity estimates
    ev_results = []
    for ev in TEST_EVENTS:
        ev_df = test[test["event_id"] == ev].copy()
        # Sort by avg_turnover_60d desc as proxy for top-K (liquid stocks)
        # (not exact XGB top-K, but gives realistic upper-bound estimate)
        top_k_df = ev_df.nlargest(k, "avg_turnover_60d")
        bottom_k_df = ev_df.nsmallest(k, "avg_turnover_60d")
        pool_median = ev_df["avg_turnover_60d"].median()

        cap_top    = float((top_k_df["avg_turnover_60d"]    * impact_pct).sum())
        cap_bottom = float((bottom_k_df["avg_turnover_60d"] * impact_pct).sum())
        cap_median = pool_median * impact_pct * k

        ev_results.append({
            "event_id":           ev,
            "pool_size":          len(ev_df),
            "pool_median_turnover_M": round(pool_median / 1e6, 1),
            "cap_top_k_M":        round(cap_top    / 1e6, 1),
            "cap_bottom_k_M":     round(cap_bottom / 1e6, 1),
            "cap_median_proxy_M": round(cap_median / 1e6, 1),
        })

    # Overall pool stats (across all test events)
    pool_p25   = test["avg_turnover_60d"].quantile(0.25)
    pool_p50   = test["avg_turnover_60d"].quantile(0.50)
    pool_p75   = test["avg_turnover_60d"].quantile(0.75)
    pool_mean  = test["avg_turnover_60d"].mean()

    return {
        "impact_pct":      impact_pct,
        "k":               k,
        "per_event":       ev_results,
        "pool_p25_M":      round(pool_p25  / 1e6, 1),
        "pool_p50_M":      round(pool_p50  / 1e6, 1),
        "pool_p75_M":      round(pool_p75  / 1e6, 1),
        "pool_mean_M":     round(pool_mean / 1e6, 1),
        "conservative_cap_M":    round(pool_p25 * impact_pct * k / 1e6, 1),
        "median_cap_M":          round(pool_p50 * impact_pct * k / 1e6, 1),
        "optimistic_cap_M":      round(pool_p75 * impact_pct * k / 1e6, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Figure
# ═══════════════════════════════════════════════════════════════════════════

def plot_cost_sensitivity(k: int, gross_avg: float, out_path: Path) -> None:
    labels   = [r[0] for r in COST_ROWS]
    costs    = [r[1] for r in COST_ROWS]
    net_rets = [(gross_avg - c / 10_000) * 100 for c in costs]

    # Shorten labels for display (English to avoid CJK font issues)
    short_labels = [
        "No Cost\n(0 bps)",
        "Fee+Tax\n(58.5 bps)",
        "+Slip 20bps\n(98.5 bps)",
        "+Slip 50bps\n(158.5 bps)",
        "+Slip 100bps\n(258.5 bps)",
    ]

    colors = ["#4C72B0" if v > 0 else "#C44E52" for v in net_rets]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.set_facecolor("white")

    bars = ax.bar(short_labels, net_rets, color=colors,
                  edgecolor="white", width=0.55)

    # Break-even line
    ax.axhline(0, color="#CC0000", linestyle="--", lw=1.8,
               label="Break-even (0%)")

    # Perfect prediction ceiling reference
    perf_gross_pct = PERFECT_T14_GROSS_BPS / 100
    ax.axhline(perf_gross_pct, color="#FF8800", linestyle=":", lw=1.5,
               label=f"完美預測 ceiling (gross {perf_gross_pct:.2f}%)")

    # Value labels on bars
    for bar, v in zip(bars, net_rets):
        va  = "bottom" if v >= 0 else "top"
        off = 0.01 if v >= 0 else -0.01
        ax.text(bar.get_x() + bar.get_width() / 2, v + off,
                f"{v:+.2f}%", ha="center", va=va, fontsize=9.5,
                fontweight="bold", color="#111")

    ax.set_ylabel("Avg Event Net Return (3 OOS events, equal-weight)", fontsize=10)
    ax.set_title(
        f"HYPOTHETICAL Strategy Cost Sensitivity (Top-K={k})\n"
        f"3 OOS events avg | Close-price fill | No market impact model",
        fontsize=12, pad=10
    )
    # Re-label legend items to English
    ax.get_legend_handles_labels()
    handles, lbls = ax.get_legend_handles_labels()
    eng_lbls = [
        l.replace("完美預測 ceiling (gross", "Perfect-pred ceiling (gross")
         .replace("%)", "%)") for l in lbls
    ]
    ax.legend(handles, eng_lbls, fontsize=9, loc="lower left")
    ax.yaxis.grid(True, color="#e0e0e0", linestyle="--")
    ax.set_axisbelow(True)

    # Warning annotation
    ax.text(0.99, 0.97,
            "HYPOTHETICAL | N=3 events | Not statistically significant",
            transform=ax.transAxes, fontsize=8,
            color="#CC0000", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0",
                      edgecolor="#CC0000", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Stage 6: 成本敏感度分析")
    print("=" * 60)

    # ── Gross averages from Stage 5 ──────────────────────────────────
    gross_k10 = load_gross_avg(10)
    gross_k5  = load_gross_avg(5)
    print(f"\nStage 5 gross avg（無成本）:")
    print(f"  K=10: {gross_k10*100:+.4f}%  ({gross_k10*10000:.1f} bps)")
    print(f"  K=5:  {gross_k5*100:+.4f}%  ({gross_k5*10000:.1f} bps)")
    print(f"  完美預測 ceiling (T-14→ann+1): {PERFECT_T14_GROSS_BPS:.2f} bps")

    # ── Cost sensitivity tables ──────────────────────────────────────
    tbl_k10 = build_cost_table(10, gross_k10, PERFECT_T14_GROSS_BPS)
    tbl_k5  = build_cost_table(5,  gross_k5,  PERFECT_T14_GROSS_BPS)

    print("\n[K=10 成本敏感度表]")
    print(tbl_k10[["情境","成本 (bps)","淨報酬 (bps)","是否 > 0"]].to_string(index=False))

    print("\n[K=5 成本敏感度表]")
    print(tbl_k5[["情境","成本 (bps)","淨報酬 (bps)","是否 > 0"]].to_string(index=False))

    # Save combined CSV
    tbl_k10["K"] = 10
    tbl_k5["K"]  = 5
    combined = pd.concat([tbl_k10, tbl_k5], ignore_index=True)
    out_csv = OUT_TABLE_DIR / "cost_sensitivity.csv"
    combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Saved: {out_csv.relative_to(ROOT)}")

    # ── Capacity estimation ──────────────────────────────────────────
    print("\n[容量估計 (K=10, 衝擊上限 = 日均成交額 10%)]")
    cap = estimate_capacity(10, impact_pct=0.10)

    print(f"  候選池日均成交額 (NT$ M):")
    print(f"    P25 = {cap['pool_p25_M']} M  P50 = {cap['pool_p50_M']} M  "
          f"P75 = {cap['pool_p75_M']} M  Mean = {cap['pool_mean_M']} M")
    print(f"\n  單事件容量估計（K={cap['k']}，衝擊上限 {cap['impact_pct']*100:.0f}%）:")
    print(f"    保守（P25 代表）: {cap['conservative_cap_M']} M NTD")
    print(f"    中位（P50 代表）: {cap['median_cap_M']} M NTD")
    print(f"    樂觀（P75 代表）: {cap['optimistic_cap_M']} M NTD")
    print(f"\n  逐事件詳細（以日均成交量排名前 K 作為樂觀上界）:")
    for ev in cap["per_event"]:
        print(f"    {ev['event_id']}: "
              f"pool={ev['pool_size']}  "
              f"pool_median={ev['pool_median_turnover_M']} M  "
              f"cap(top-K)={ev['cap_top_k_M']} M  "
              f"cap(mid proxy)={ev['cap_median_proxy_M']} M")

    # ── Break-even summary ───────────────────────────────────────────
    be_k10 = gross_k10 * 10_000
    be_k5  = gross_k5  * 10_000
    be_perf = PERFECT_T14_GROSS_BPS

    print(f"\n[Break-even 成本對比]")
    print(f"  K=10 break-even:          {be_k10:.1f} bps")
    print(f"  K=5  break-even:          {be_k5:.1f} bps")
    print(f"  完美預測 break-even:      {be_perf:.1f} bps")
    print(f"  台股實際最低成本（手續費+稅）: 58.5 bps")
    print(f"  結論: K=10 break-even ({be_k10:.1f} bps) < 58.5 bps → 現實成本即虧損")
    print(f"        完美預測 break-even ({be_perf:.1f} bps) < 58.5 bps → 天花板亦不夠覆蓋")

    # ── Figure ───────────────────────────────────────────────────────
    print("\n[生成圖表]")
    plot_cost_sensitivity(10, gross_k10, OUT_FIG_DIR / "cost_sensitivity.png")

    # ── Summary for findings ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 6 SUMMARY")
    print("=" * 60)

    return {
        "gross_k10": gross_k10, "gross_k5": gross_k5,
        "be_k10": be_k10, "be_k5": be_k5,
        "cap": cap, "tbl_k10": tbl_k10, "tbl_k5": tbl_k5,
    }


if __name__ == "__main__":
    main()
