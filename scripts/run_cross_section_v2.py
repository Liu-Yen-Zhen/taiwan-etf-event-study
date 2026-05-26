"""
scripts/run_cross_section_v2.py
-------------------------------
階段 3 v2：用清理過的 stock_prices.parquet（OHLC=0 → NaN）+ 新動能變數
（避免與 CAR_pre_30 視窗重疊）重跑橫斷面分析。

新自變數
--------
  log_market_cap, log_avg_turnover, turnover_ratio_60d,
  dividend_yield,
  momentum_pre [-90, -31]            (取代 momentum_60d)
  volume_anomaly_pre_5 [-30,-25]/[-90,-31]  (取代 momentum_recent_5d)

產出
----
  output/tables/cross_section_dataset_v2.csv
  output/tables/cross_section_describe_v2.csv
  output/tables/cross_section_regression_v2.txt
  output/figures/group_car_<feature>_v2.png  (6 張)
  output/figures/regression_diagnostics_pre_v2.png
  output/figures/regression_diagnostics_post_v2.png
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm

from _plot_config import apply_chinese_style
apply_chinese_style()

from data_fetcher import load_events
from event_study import compute_event_car
from cross_section import _prior_value, _winsorize

PRICES_PQ   = ROOT / "data" / "processed" / "stock_prices.parquet"
EVENTS_CSV  = ROOT / "data" / "raw"       / "events.csv"
PER_CACHE   = ROOT / "data" / "processed" / "stock_per_cache.parquet"
SHARES_CACHE= ROOT / "data" / "processed" / "stock_shares_cache.parquet"
INFO_CACHE  = ROOT / "data" / "processed" / "stock_info_cache.parquet"

OUT_DATASET = ROOT / "output" / "tables"  / "cross_section_dataset_v2.csv"
OUT_DESC    = ROOT / "output" / "tables"  / "cross_section_describe_v2.csv"
OUT_REG     = ROOT / "output" / "tables"  / "cross_section_regression_v2.txt"
FIG_DIR     = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature helpers ──────────────────────────────────────────────────────


def _last_car(stock_id, ann_date, prices_wide, market_returns, window):
    try:
        df = compute_event_car(stock_id, ann_date, prices_wide, market_returns,
                               event_window=window)
        if df.empty:
            return np.nan
        return float(df["CAR"].iloc[-1])
    except (ValueError, KeyError):
        return np.nan


def _momentum_pre(prices_wide, stock_id, ref_date):
    """Cumulative return over [-90, -31] (60 trading days, ends BEFORE CAR_pre_30 window)."""
    if stock_id not in prices_wide.columns:
        return np.nan
    pre = prices_wide.index[prices_wide.index < ref_date]
    if len(pre) < 91:
        return np.nan
    window = pre[-90:-30]
    p = prices_wide.loc[window, stock_id].dropna()
    if len(p) < 2:
        return np.nan
    rets = p.pct_change().dropna()
    return float((1 + rets).prod() - 1)


def _volume_anomaly_pre_5(volume_wide, stock_id, ref_date):
    """mean(volume[-30:-25]) / mean(volume[-90:-30])."""
    if stock_id not in volume_wide.columns:
        return np.nan
    pre = volume_wide.index[volume_wide.index < ref_date]
    if len(pre) < 90:
        return np.nan
    num = volume_wide.loc[pre[-30:-25], stock_id].dropna().mean()
    den = volume_wide.loc[pre[-90:-30], stock_id].dropna().mean()
    if pd.isna(num) or pd.isna(den) or den <= 0:
        return np.nan
    return float(num / den)


def _turnover_ratio_60d(volume_wide, shares_df, stock_id, ref_date, lookback=60):
    if stock_id not in volume_wide.columns:
        return np.nan
    pre = volume_wide.index[volume_wide.index < ref_date]
    if len(pre) < 10:
        return np.nan
    window = pre[-lookback:]
    avg_vol = float(volume_wide.loc[window, stock_id].mean())
    if pd.isna(avg_vol) or avg_vol <= 0:
        return np.nan
    shares = _prior_value(shares_df, stock_id, ref_date, "shares_issued")
    if pd.isna(shares) or shares <= 0:
        return np.nan
    return avg_vol / shares


# ── Build panel ──────────────────────────────────────────────────────────


def build_panel_v2(events, prices_wide, volume_wide, turnover_wide,
                   market_returns, per_df, shares_df, info_df):
    industry_map = (info_df.drop_duplicates(subset=["stock_id"], keep="last")
                    .set_index("stock_id")["industry_category"].to_dict())
    name_map = (info_df.drop_duplicates(subset=["stock_id"], keep="last")
                .set_index("stock_id")["stock_name"].to_dict())

    records = []
    for _, ev in events.iterrows():
        ann = pd.Timestamp(ev["announcement_date"])
        for sid in ev["added_stocks"]:
            row = {
                "event_id":   ev["event_id"],
                "ann_date":   ann,
                "stock_id":   sid,
                "stock_name": name_map.get(sid, ""),
                "industry":   industry_map.get(sid, "Unknown"),
            }
            # CARs (with cleaned prices)
            row["CAR_pre_30"]  = _last_car(sid, ann, prices_wide, market_returns, (-30, 0))
            row["CAR_pre_5"]   = _last_car(sid, ann, prices_wide, market_returns, (-5, 0))
            row["CAR_post_5"]  = _last_car(sid, ann, prices_wide, market_returns, (0, 5))
            row["CAR_post_30"] = _last_car(sid, ann, prices_wide, market_returns, (1, 31))

            # Existing features
            pre = prices_wide.index[prices_wide.index < ann]
            if len(pre) == 0 or sid not in prices_wide.columns:
                price_prev = np.nan
            else:
                p = prices_wide.loc[pre[-1], sid]
                price_prev = float(p) if pd.notna(p) and p > 0 else np.nan

            shares = _prior_value(shares_df, sid, ann, "shares_issued")
            row["market_cap"] = (price_prev * shares
                                 if (pd.notna(price_prev) and pd.notna(shares)) else np.nan)

            if sid in turnover_wide.columns:
                av = float(turnover_wide.loc[pre[-60:], sid].mean())
                row["avg_turnover_60d"] = av if pd.notna(av) else np.nan
            else:
                row["avg_turnover_60d"] = np.nan

            row["turnover_ratio_60d"] = _turnover_ratio_60d(volume_wide, shares_df, sid, ann)
            row["dividend_yield"] = _prior_value(per_df, sid, ann, "dividend_yield")

            # NEW: non-overlapping momentum + volume anomaly
            row["momentum_pre"]         = _momentum_pre(prices_wide, sid, ann)
            row["volume_anomaly_pre_5"] = _volume_anomaly_pre_5(volume_wide, sid, ann)

            mc, av = row["market_cap"], row["avg_turnover_60d"]
            row["log_market_cap"]   = float(np.log(mc)) if (pd.notna(mc) and mc > 0) else np.nan
            row["log_avg_turnover"] = float(np.log(av)) if (pd.notna(av) and av > 0) else np.nan

            records.append(row)

    return pd.DataFrame(records)


# ── Plotting ─────────────────────────────────────────────────────────────


def plot_group_car_v2(panel, feature, outcomes, save_path, label_x):
    sub = panel[[feature, *outcomes]].dropna(subset=[feature]).copy()
    if len(sub) < 9:
        print(f"  ⚠ {feature}: N={len(sub)} too few; skip")
        return
    try:
        sub["tercile"] = pd.qcut(sub[feature], q=3,
                                  labels=["Low", "Mid", "High"], duplicates="drop")
    except ValueError:
        print(f"  ⚠ {feature}: qcut failed; skip")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#5b9bd5", "#a5a5a5", "#ed7d31"]
    for ax, outcome in zip(axes, outcomes):
        s = sub.dropna(subset=[outcome])
        if s.empty:
            ax.set_title(f"{outcome} (no data)")
            continue
        agg = (s.groupby("tercile", observed=True)[outcome]
               .agg(["mean", "std", "count"]).reindex(["Low", "Mid", "High"]))
        agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
        x = np.arange(len(agg))
        ax.bar(x, agg["mean"] * 100, yerr=agg["ci"] * 100,
               color=colors, alpha=0.85, capsize=5, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lbl}\nN={int(n)}" for lbl, n in zip(agg.index, agg["count"])],
                           fontsize=9)
        ax.set_ylabel("平均 CAR (%)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
        ax.set_title({"CAR_pre_30": "公告前 30 日 CAR",
                       "CAR_post_30": "生效後 30 日 CAR"}.get(outcome, outcome), fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(f"[REAL DATA — v2] 分組分析: {label_x}（低/中/高三分位）",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path.name}")


def plot_diagnostics_v2(model, sub, outcome, save_path):
    fitted = model.fittedvalues
    resid = model.resid
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].scatter(fitted, resid, alpha=0.6, color="#1f77b4", edgecolor="white")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fitted")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].grid(linestyle=":", alpha=0.5)

    sm.qqplot(resid, line="45", fit=True, ax=axes[1])
    axes[1].set_title("Normal Q-Q")
    axes[1].grid(linestyle=":", alpha=0.5)

    cooks = model.get_influence().cooks_distance[0]
    axes[2].stem(range(len(cooks)), cooks, linefmt="C0-", markerfmt="C0o", basefmt=" ")
    threshold = 4 / len(cooks)
    axes[2].axhline(threshold, color="red", linestyle="--", linewidth=1,
                    label=f"4/N = {threshold:.3f}")
    axes[2].set_title(f"Cook's d (top {int((cooks>threshold).sum())} flagged)")
    axes[2].legend(fontsize=8)
    axes[2].grid(linestyle=":", alpha=0.5)

    fig.suptitle(f"[REAL DATA — v2] Diagnostics — {outcome} (N={len(sub)})",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    top_idx = np.argsort(cooks)[-5:][::-1]
    return sub.iloc[top_idx].assign(cooks_d=cooks[top_idx])


# ── Main ─────────────────────────────────────────────────────────────────


REG_FEATURES = [
    "log_market_cap", "log_avg_turnover", "turnover_ratio_60d",
    "dividend_yield", "momentum_pre", "volume_anomaly_pre_5",
]


def fit(features, outcome, df):
    cols = ["event_id", "stock_id", outcome] + features
    sub = df[cols].dropna(subset=[outcome] + features).reset_index(drop=True)
    X = sm.add_constant(sub[features])
    y = sub[outcome]
    return sm.OLS(y, X).fit(), sub


def main():
    print("=" * 70)
    print("  Stage 3 v2 — Cross-section with cleaned data + non-overlap features")
    print("=" * 70)

    long = pd.read_parquet(PRICES_PQ)
    prices_wide = long.pivot(index="date", columns="stock_id",
                             values="close").sort_index()
    volume_wide = long.pivot(index="date", columns="stock_id",
                             values="volume").sort_index()
    turnover_wide = long.pivot(index="date", columns="stock_id",
                                values="turnover").sort_index()
    market_returns = prices_wide["TAIEX"].pct_change()
    events = load_events(EVENTS_CSV)

    per_df    = pd.read_parquet(PER_CACHE)
    shares_df = pd.read_parquet(SHARES_CACHE)
    info_df   = pd.read_parquet(INFO_CACHE)

    print(f"  cleaned panel : {prices_wide.shape}, NaN close cells = {int(prices_wide.isna().sum().sum())}")
    print(f"  events        : {len(events)}")
    print(f"  PER cache     : {len(per_df):,} rows")

    # Build v2 panel
    panel = build_panel_v2(events, prices_wide, volume_wide, turnover_wide,
                            market_returns, per_df, shares_df, info_df)
    panel.to_csv(OUT_DATASET, index=False, float_format="%.6f")
    print(f"\n  ✓ panel ({len(panel)} rows) → {OUT_DATASET.relative_to(ROOT)}")

    # Describe
    feats = ["CAR_pre_30", "CAR_pre_5", "CAR_post_5", "CAR_post_30",
             "log_market_cap", "log_avg_turnover", "turnover_ratio_60d",
             "dividend_yield", "momentum_pre", "volume_anomaly_pre_5"]
    desc = panel[feats].describe().T
    desc["missing"] = len(panel) - panel[feats].count()
    desc = desc[["count", "missing", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc.to_csv(OUT_DESC, float_format="%.4f")
    print(f"  ✓ describe → {OUT_DESC.relative_to(ROOT)}")

    # Group plots (6)
    print("\n  [Group plots]")
    plots = [
        ("log_market_cap",       "log(市值)"),
        ("log_avg_turnover",     "log(60 日平均成交金額)"),
        ("turnover_ratio_60d",   "60 日周轉率"),
        ("dividend_yield",       "殖利率 (%)"),
        ("momentum_pre",         "[-90,-31] 動能"),
        ("volume_anomaly_pre_5", "[-30,-25] vs [-90,-31] 成交量比"),
    ]
    for feat, lbl in plots:
        plot_group_car_v2(panel, feat, ("CAR_pre_30", "CAR_post_30"),
                          FIG_DIR / f"group_car_{feat}_v2.png", lbl)

    # Regressions
    m_pre,  sub_pre  = fit(REG_FEATURES, "CAR_pre_30", panel)
    m_post, sub_post = fit(REG_FEATURES, "CAR_post_30", panel)

    print(f"\n  CAR_pre_30  : N={len(sub_pre)}, R²={m_pre.rsquared:.4f}, adj-R²={m_pre.rsquared_adj:.4f}")
    print(f"  CAR_post_30 : N={len(sub_post)}, R²={m_post.rsquared:.4f}, adj-R²={m_post.rsquared_adj:.4f}")

    # Save regression text
    with open(OUT_REG, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  Cross-Sectional Regression v2  (cleaned data + non-overlap features)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"N panel : {len(panel)} (event × added_stock)\n")
        f.write(f"Predictors: " + ", ".join(REG_FEATURES) + "\n\n")
        f.write("─" * 70 + "\n")
        f.write("MODEL 1: CAR_pre_30\n")
        f.write("─" * 70 + "\n")
        f.write(str(m_pre.summary()) + "\n\n")
        f.write("─" * 70 + "\n")
        f.write("MODEL 2: CAR_post_30\n")
        f.write("─" * 70 + "\n")
        f.write(str(m_post.summary()) + "\n\n")
    print(f"  ✓ regression → {OUT_REG.relative_to(ROOT)}")

    # Diagnostics
    top_pre  = plot_diagnostics_v2(m_pre, sub_pre, "CAR_pre_30",
                                    FIG_DIR / "regression_diagnostics_pre_v2.png")
    top_post = plot_diagnostics_v2(m_post, sub_post, "CAR_post_30",
                                    FIG_DIR / "regression_diagnostics_post_v2.png")

    print("\n  Top-5 Cook's distance (CAR_pre_30):")
    print(top_pre[["event_id", "stock_id", "cooks_d"]].to_string(index=False))
    print("\n  Top-5 Cook's distance (CAR_post_30):")
    print(top_post[["event_id", "stock_id", "cooks_d"]].to_string(index=False))

    # ── v1 vs v2 comparison ───────────────────────────────────────────────
    # v1 values from diagnose_stage3.py output (uncleaned data + new features)
    v1_pre = {
        "log_market_cap":       (+0.0096, +0.292, 0.7711),
        "log_avg_turnover":     (-0.0243, -0.738, 0.4635),
        "turnover_ratio_60d":   (+1.7498, +0.819, 0.4164),
        "dividend_yield":       (-0.0111, -1.115, 0.2699),
        "momentum_pre":         (-0.0518, -0.618, 0.5391),
        "volume_anomaly_pre_5": (+0.0148, +1.147, 0.2564),
    }
    v1_post = {
        "log_market_cap":       (+0.0719, +1.447, 0.1537),
        "log_avg_turnover":     (-0.1274, -2.558, 0.0134),
        "turnover_ratio_60d":   (+8.0633, +2.498, 0.0156),
        "dividend_yield":       (-0.0099, -0.656, 0.5144),
        "momentum_pre":         (+0.0109, +0.086, 0.9315),
        "volume_anomaly_pre_5": (+0.0089, +0.458, 0.6489),
    }
    v1_overall = {"N_pre": 61, "N_post": 61,
                  "R²_pre": 0.072, "adjR²_pre": -0.031,
                  "R²_post": 0.214, "adjR²_post": 0.127}

    def sig(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "—"

    def fmt_change(old_p, new_p):
        s1, s2 = sig(old_p), sig(new_p)
        return f"{s1} → {s2}" if s1 != s2 else f"無變化 ({s1})"

    print("\n" + "=" * 110)
    print("  v1 (含 artifact) vs v2 (清理後) 對照")
    print("=" * 110)

    for label, model, v1_vals in [("CAR_pre_30", m_pre, v1_pre),
                                   ("CAR_post_30", m_post, v1_post)]:
        print(f"\n  ── {label} ──")
        print(f"  {'變數':22s} {'v1 β':>9s} {'v1 t':>7s} {'v1 p':>7s}    "
              f"{'v2 β':>9s} {'v2 t':>7s} {'v2 p':>7s}    結論變化")
        print(f"  {'-'*22} {'-'*9} {'-'*7} {'-'*7}    {'-'*9} {'-'*7} {'-'*7}    {'-'*15}")
        for feat in REG_FEATURES:
            v1b, v1t, v1p = v1_vals[feat]
            v2b = model.params[feat]; v2t = model.tvalues[feat]; v2p = model.pvalues[feat]
            print(f"  {feat:22s} {v1b:>+9.4f} {v1t:>+7.2f} {v1p:>7.4f}    "
                  f"{v2b:>+9.4f} {v2t:>+7.2f} {v2p:>7.4f}    {fmt_change(v1p, v2p)}")

    print("\n  ── 整體模型 ──")
    print(f"  {'指標':22s} {'v1':>10s}    {'v2':>10s}")
    print(f"  {'-'*22} {'-'*10}    {'-'*10}")
    print(f"  {'N (CAR_pre_30)':22s} {v1_overall['N_pre']:>10d}    {len(sub_pre):>10d}")
    print(f"  {'N (CAR_post_30)':22s} {v1_overall['N_post']:>10d}    {len(sub_post):>10d}")
    print(f"  {'R² CAR_pre_30':22s} {v1_overall['R²_pre']:>10.4f}    {m_pre.rsquared:>10.4f}")
    print(f"  {'adj-R² CAR_pre_30':22s} {v1_overall['adjR²_pre']:>10.4f}    {m_pre.rsquared_adj:>10.4f}")
    print(f"  {'R² CAR_post_30':22s} {v1_overall['R²_post']:>10.4f}    {m_post.rsquared:>10.4f}")
    print(f"  {'adj-R² CAR_post_30':22s} {v1_overall['adjR²_post']:>10.4f}    {m_post.rsquared_adj:>10.4f}")


if __name__ == "__main__":
    main()
