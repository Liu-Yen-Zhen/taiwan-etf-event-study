"""
scripts/diagnose_stage3.py
--------------------------
階段 3 方法論修正診斷：

1. CAR_post_30 outlier 查證（< -50% 的觀測）
3. 重新設計動能變數，避免時間重疊（並列舊版 vs 新版迴歸）
4. 產業精確分布報告

不剔除任何 outlier、不重新製圖、不更新桌面資料夾。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import requests

from data_fetcher import load_events
from event_study import compute_event_car
from cross_section import _prior_value

PRICES_PQ  = ROOT / "data" / "processed" / "stock_prices.parquet"
EVENTS_CSV = ROOT / "data" / "raw"       / "events.csv"
PANEL_CSV  = ROOT / "output" / "tables"  / "cross_section_dataset.csv"
SHARES_PQ  = ROOT / "data" / "processed" / "stock_shares_cache.parquet"
INFO_PQ    = ROOT / "data" / "processed" / "stock_info_cache.parquet"


def _sig(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ════════════════════════════════════════════════════════════════════════
# 問題 1：CAR_post_30 outlier 查證
# ════════════════════════════════════════════════════════════════════════


def investigate_outliers(panel, prices_wide, volume_wide, events):
    print("=" * 78)
    print("  問題 1：CAR_post_30 < -50% outlier 查證")
    print("=" * 78)

    outliers = panel[panel["CAR_post_30"] < -0.50].copy()
    print(f"\n  CAR_post_30 < -50% 觀測數：{len(outliers)}")
    print(f"  全部 71 觀測中佔比：{len(outliers)/len(panel):.1%}")

    if outliers.empty:
        print("  ✓ 無極端 outlier，跳過詳細查證")
        return

    print(f"\n  Outlier 清單：")
    print(outliers[["event_id", "stock_id", "stock_name",
                    "CAR_pre_30", "CAR_post_30"]].to_string(index=False))

    # 對每個 outlier 印出生效日後 30 個交易日的 close + volume
    for _, row in outliers.iterrows():
        sid = row["stock_id"]
        ev = events[events["event_id"] == row["event_id"]].iloc[0]
        eff_date = pd.Timestamp(ev["effective_date"])

        print(f"\n  ── {sid} ({row['stock_name']}) | event {row['event_id']} ──")
        print(f"     生效日: {eff_date.date()}  CAR_post_30 = {row['CAR_post_30']:.2%}")

        # 找出 eff_date 開始的 31 個交易日
        trading_days = prices_wide.index
        eff_pos_arr = trading_days.get_indexer([eff_date], method="bfill")
        if eff_pos_arr[0] == -1:
            print("     ⚠ 找不到生效日對應的交易日")
            continue
        eff_pos = int(eff_pos_arr[0])
        window = trading_days[eff_pos : eff_pos + 32]   # eff_date 起 32 個交易日

        if sid not in prices_wide.columns:
            print(f"     ⚠ {sid} 不在價格資料中")
            continue

        close = prices_wide.loc[window, sid]
        vol = volume_wide.loc[window, sid] if sid in volume_wide.columns else None

        # 簡潔輸出：日期、close、volume、單日報酬
        df_show = pd.DataFrame({
            "close": close.values,
            "ret_d": close.pct_change().values,
            "volume": vol.values if vol is not None else np.nan,
        }, index=window)

        # 找出 NaN、停牌 (volume=0)、和異常單日跌幅 < -8%
        df_show["flag"] = ""
        df_show.loc[df_show["close"].isna(), "flag"] = "❌ 無價"
        df_show.loc[df_show["volume"] == 0, "flag"] += " ⚠ 停牌"
        df_show.loc[df_show["ret_d"] < -0.08, "flag"] += " ⚠ 急跌"
        df_show.loc[df_show["ret_d"] > 0.08, "flag"] += " ⚠ 急漲"

        # 統計
        n_nan_close = int(df_show["close"].isna().sum())
        n_zero_vol = int((df_show["volume"] == 0).sum())
        last_valid_idx = df_show["close"].last_valid_index()
        first_valid_idx = df_show["close"].first_valid_index()

        print(f"     視窗交易日: {len(df_show)}  缺價: {n_nan_close}  零成交量: {n_zero_vol}")
        print(f"     首/末有價日: {first_valid_idx.date() if first_valid_idx else 'NA'} → "
              f"{last_valid_idx.date() if last_valid_idx else 'NA'}")
        if first_valid_idx is not None and last_valid_idx is not None:
            p0, p1 = df_show.loc[first_valid_idx, "close"], df_show.loc[last_valid_idx, "close"]
            print(f"     首/末有價收盤: {p0:.2f} → {p1:.2f}  ({(p1/p0-1):+.2%} 真實價格變動)")

        # 印出有 flag 的列 + 前 3 + 後 3
        flagged = df_show[df_show["flag"].str.strip() != ""]
        if not flagged.empty:
            print(f"     異常列 ({len(flagged)} 筆)：")
            print(flagged[["close", "ret_d", "volume", "flag"]].to_string(
                float_format=lambda x: f"{x:.3f}" if abs(x) < 1000 else f"{x:.0f}"))


# ════════════════════════════════════════════════════════════════════════
# 問題 3：重新設計動能變數
# ════════════════════════════════════════════════════════════════════════


def compute_momentum_pre(prices_wide, stock_id, ref_date):
    """[-90, -31] 累積報酬（公告前第 90 到第 31 日，60 天，結束於 CAR_pre_30 視窗開始之前）。

    使用嚴格 pre = trading_days < ref_date 的位置切片。
    """
    if stock_id not in prices_wide.columns:
        return np.nan
    pre = prices_wide.index[prices_wide.index < ref_date]
    if len(pre) < 91:
        return np.nan
    window = pre[-90:-30]              # 60 個交易日，最末為 -31
    p = prices_wide.loc[window, stock_id].dropna()
    if len(p) < 2:
        return np.nan
    rets = p.pct_change().dropna()
    return float((1 + rets).prod() - 1)


def compute_volume_anomaly(volume_wide, stock_id, ref_date):
    """volume_anomaly = mean(volume[-30:-25]) / mean(volume[-90:-30])。

    依使用者規格：分子 = 5 個交易日（位置 -30 至 -26），分母 = 60 個交易日。
    注意：分子時間窗在 CAR_pre_30 視窗之內，但 volume 不是 return 本身、
    不會造成 CAR 與自變數的自相關。
    """
    if stock_id not in volume_wide.columns:
        return np.nan
    pre = volume_wide.index[volume_wide.index < ref_date]
    if len(pre) < 90:
        return np.nan
    num_w = pre[-30:-25]      # 5 個交易日（位置 -30 至 -26）
    den_w = pre[-90:-30]      # 60 個交易日（位置 -90 至 -31）
    num_vol = volume_wide.loc[num_w, stock_id].dropna().mean()
    den_vol = volume_wide.loc[den_w, stock_id].dropna().mean()
    if pd.isna(num_vol) or pd.isna(den_vol) or den_vol <= 0:
        return np.nan
    return float(num_vol / den_vol)


def rerun_regressions(panel, prices_wide, volume_wide, events):
    print("\n" + "=" * 78)
    print("  問題 3：動能變數重設計 — 舊版 vs 新版迴歸比較")
    print("=" * 78)

    # 為每筆觀測計算新動能變數（用 ann_date）
    momentum_pre = []
    volume_anomaly = []
    for _, row in panel.iterrows():
        ev = events[events["event_id"] == row["event_id"]].iloc[0]
        ann = pd.Timestamp(ev["announcement_date"])
        sid = row["stock_id"]
        momentum_pre.append(compute_momentum_pre(prices_wide, sid, ann))
        volume_anomaly.append(compute_volume_anomaly(volume_wide, sid, ann))

    panel2 = panel.copy()
    panel2["momentum_pre"] = momentum_pre
    panel2["volume_anomaly_pre_5"] = volume_anomaly

    print(f"\n  新變數描述性統計：")
    print(panel2[["momentum_pre", "volume_anomaly_pre_5"]].describe().to_string())
    print(f"  缺失：momentum_pre={panel2['momentum_pre'].isna().sum()},  "
          f"volume_anomaly_pre_5={panel2['volume_anomaly_pre_5'].isna().sum()}")

    # ── 舊版迴歸（重做、相同 dataset） ──
    OLD = ["log_market_cap", "log_avg_turnover", "turnover_ratio_60d",
           "dividend_yield", "momentum_60d", "momentum_recent_5d"]
    # ── 新版迴歸 ──
    NEW = ["log_market_cap", "log_avg_turnover", "turnover_ratio_60d",
           "dividend_yield", "momentum_pre", "volume_anomaly_pre_5"]

    def fit(features, outcome, df):
        sub = df[[outcome] + features].dropna()
        X = sm.add_constant(sub[features])
        y = sub[outcome]
        return sm.OLS(y, X).fit(), len(sub)

    print(f"\n  ╔═════════════════════════════════════════════════════════════╗")
    print(f"  ║   CAR_pre_30 ~ features                                     ║")
    print(f"  ╚═════════════════════════════════════════════════════════════╝")
    for label, feats in [("OLD (含重疊 momentum)", OLD), ("NEW (無重疊)", NEW)]:
        m, n = fit(feats, "CAR_pre_30", panel2)
        print(f"\n  ── {label}  ──  N={n}  R²={m.rsquared:.3f}  adj-R²={m.rsquared_adj:.3f}")
        for f in feats:
            print(f"    {f:24s}  β={m.params[f]:+.4f}  t={m.tvalues[f]:+.3f}  "
                  f"p={m.pvalues[f]:.4f}  {_sig(m.pvalues[f])}")

    print(f"\n  ╔═════════════════════════════════════════════════════════════╗")
    print(f"  ║   CAR_post_30 ~ features                                    ║")
    print(f"  ╚═════════════════════════════════════════════════════════════╝")
    for label, feats in [("OLD", OLD), ("NEW", NEW)]:
        m, n = fit(feats, "CAR_post_30", panel2)
        print(f"\n  ── {label}  ──  N={n}  R²={m.rsquared:.3f}  adj-R²={m.rsquared_adj:.3f}")
        for f in feats:
            print(f"    {f:24s}  β={m.params[f]:+.4f}  t={m.tvalues[f]:+.3f}  "
                  f"p={m.pvalues[f]:.4f}  {_sig(m.pvalues[f])}")

    # 把新變數寫回 panel（不覆蓋既有 CSV，先供後續決定）
    out_path = ROOT / "output" / "tables" / "cross_section_dataset_v2.csv"
    panel2.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n  ✓ 新版 panel（含新動能變數）→ {out_path.relative_to(ROOT)}")


# ════════════════════════════════════════════════════════════════════════
# 問題 4：產業精確分布
# ════════════════════════════════════════════════════════════════════════


def industry_breakdown(panel):
    print("\n" + "=" * 78)
    print("  問題 4：產業精確分布")
    print("=" * 78)

    sub = panel.dropna(subset=["CAR_pre_30", "CAR_post_30"]).copy()
    print(f"\n  總觀測：{len(sub)}")

    counts = sub["industry"].value_counts()
    print(f"  獨立產業數：{len(counts)}")

    agg = (sub.groupby("industry")
           .agg(N=("CAR_pre_30", "count"),
                mean_pre=("CAR_pre_30", "mean"),
                std_pre=("CAR_pre_30", "std"),
                mean_post=("CAR_post_30", "mean"),
                std_post=("CAR_post_30", "std"))
           .sort_values("N", ascending=False))
    agg["mean_pre_%"]  = agg["mean_pre"] * 100
    agg["mean_post_%"] = agg["mean_post"] * 100

    # N ≥ 5 與 N < 5 分開報告
    big = agg[agg["N"] >= 5].copy()
    small = agg[agg["N"] < 5].copy()

    print(f"\n  ── N ≥ 5 的產業（{len(big)} 個）──")
    print(big[["N", "mean_pre_%", "mean_post_%"]].to_string(
        float_format=lambda x: f"{x:+.2f}"))

    print(f"\n  ── N < 5 的產業（{len(small)} 個，合併為 Other 之前的狀態）──")
    print(small[["N", "mean_pre_%", "mean_post_%"]].to_string(
        float_format=lambda x: f"{x:+.2f}"))

    print(f"\n  ── 整體分布摘要 ──")
    print(f"    N ≥ 5 產業：{len(big)} 個，共 {big['N'].sum()} 觀測（占 {big['N'].sum()/len(sub):.1%}）")
    print(f"    N < 5 產業：{len(small)} 個，共 {small['N'].sum()} 觀測（占 {small['N'].sum()/len(sub):.1%}）")


# ════════════════════════════════════════════════════════════════════════
# 主程式
# ════════════════════════════════════════════════════════════════════════


def main():
    # Load data
    print("Loading panel + price data …")
    panel = pd.read_csv(PANEL_CSV, parse_dates=["ann_date"],
                        dtype={"stock_id": str, "event_id": str})
    long = pd.read_parquet(PRICES_PQ)
    prices_wide = long.pivot(index="date", columns="stock_id", values="close").sort_index()
    volume_wide = long.pivot(index="date", columns="stock_id", values="volume").sort_index()
    events = load_events(EVENTS_CSV)
    print(f"  panel rows = {len(panel)}, events = {len(events)}")
    print()

    # 問題 1
    investigate_outliers(panel, prices_wide, volume_wide, events)

    # 問題 3
    rerun_regressions(panel, prices_wide, volume_wide, events)

    # 問題 4
    industry_breakdown(panel)


if __name__ == "__main__":
    main()
