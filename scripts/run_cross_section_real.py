"""
scripts/run_cross_section_real.py
---------------------------------
Stage 3 — Cross-sectional analysis of 00919 added stocks.

For each added stock event pair (N ≤ 71):
  • Compute CAR_pre_30, CAR_post_30, CAR_pre_5, CAR_post_5
  • Compute pre-announcement features (no look-ahead)
  • Univariate group analysis (low/mid/high tertile per feature)
  • Multivariate OLS regression
  • Residual diagnostics
  • Industry effect

Data sources (all FinMind, real data only):
  • Prices  : data/processed/stock_prices.parquet (already fetched)
  • PER     : FinMind TaiwanStockPER (dividend_yield daily)
  • Shares  : FinMind TaiwanStockDividend (ParticipateDistributionOfTotalShares)
  • Industry: FinMind TaiwanStockInfo

Dropped from spec
-----------------
  • impact_ratio : NO ETF AUM data in FinMind free tier; 00919 also lacks
                   shares-outstanding data for an AUM proxy. Documented as
                   limitation.
  • target_weight: Without true allocation data, equal-weight assumption
                   (0.30 / N_added) collapses to event-size dummy and is
                   not informative. Excluded.

Outputs
-------
  output/tables/cross_section_dataset.csv    full per-stock-event panel
  output/tables/cross_section_describe.csv   describe() per feature
  output/tables/cross_section_regression.txt OLS summaries (pre & post)
  output/figures/group_car_<feature>.png      6 figures
  output/figures/regression_diagnostics_pre.png
  output/figures/regression_diagnostics_post.png
  output/figures/group_car_by_industry.png
  data/processed/cross_section_missing.csv   if any feature missing
"""

import sys
import time
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as sstats
import statsmodels.api as sm

from _plot_config import apply_chinese_style
apply_chinese_style()

from data_fetcher import load_events
from event_study import compute_event_car
from cross_section import (
    fetch_per_multiple,
    fetch_shares_multiple,
    _prior_value,
    _winsorize,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ── Paths ─────────────────────────────────────────────────────────────────
EVENTS_CSV  = ROOT / "data" / "raw"       / "events.csv"
PRICES_PQ   = ROOT / "data" / "processed" / "stock_prices.parquet"
PER_CACHE   = ROOT / "data" / "processed" / "stock_per_cache.parquet"
SHARES_CACHE= ROOT / "data" / "processed" / "stock_shares_cache.parquet"
INFO_CACHE  = ROOT / "data" / "processed" / "stock_info_cache.parquet"
OUT_DATASET = ROOT / "output" / "tables"  / "cross_section_dataset.csv"
OUT_DESC    = ROOT / "output" / "tables"  / "cross_section_describe.csv"
OUT_REG     = ROOT / "output" / "tables"  / "cross_section_regression.txt"
OUT_MISS    = ROOT / "data" / "processed" / "cross_section_missing.csv"
FIG_DIR     = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "output" / "tables").mkdir(parents=True, exist_ok=True)

# ── Data fetch window (covers all events with buffer) ─────────────────────
DATA_START = "2022-01-01"
DATA_END   = "2026-01-31"


# ── Helpers ──────────────────────────────────────────────────────────────


def _last_car(stock_id, ann_date, prices_wide, market_returns, window):
    """Return cumulative CAR at end of window, or NaN if data insufficient."""
    try:
        df = compute_event_car(stock_id, ann_date, prices_wide, market_returns,
                               event_window=window)
        if df.empty:
            return np.nan
        return float(df["CAR"].iloc[-1])
    except (ValueError, KeyError):
        return np.nan


def _compute_momentum(prices_wide, stock_id, ref_date,
                      lookback=60, skip=5):
    """Cumulative return over [ref - lookback - skip, ref - skip - 1] trading days.

    Excludes the most-recent `skip` days to avoid contamination with the
    'recent_5d' window. No look-ahead: all data strictly before ref_date.
    """
    if stock_id not in prices_wide.columns:
        return np.nan
    pre = prices_wide.index[prices_wide.index < ref_date]
    if len(pre) < lookback + skip:
        return np.nan
    window_days = pre[-(lookback + skip):-skip]
    p = prices_wide.loc[window_days, stock_id].dropna()
    if len(p) < 2:
        return np.nan
    rets = p.pct_change().dropna()
    return float((1 + rets).prod() - 1)


def _compute_recent_momentum(prices_wide, stock_id, ref_date, n=5):
    """Cumulative return over last `n` trading days strictly before ref_date."""
    if stock_id not in prices_wide.columns:
        return np.nan
    pre = prices_wide.index[prices_wide.index < ref_date]
    if len(pre) < n + 1:
        return np.nan
    window_days = pre[-(n + 1):]    # need n+1 prices for n returns
    p = prices_wide.loc[window_days, stock_id].dropna()
    if len(p) < 2:
        return np.nan
    rets = p.pct_change().dropna()
    return float((1 + rets).prod() - 1)


def _compute_turnover_ratio(volume_wide, shares_df, stock_id, ref_date,
                            lookback=60):
    """avg_daily_volume_60d / shares_issued (most recent before ref_date)."""
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


# ── 1. Load price panel ──────────────────────────────────────────────────


def load_panels():
    print("Loading stock_prices.parquet …")
    long = pd.read_parquet(PRICES_PQ)
    prices_wide = long.pivot(index="date", columns="stock_id",
                             values="close").sort_index()
    volume_wide = long.pivot(index="date", columns="stock_id",
                             values="volume").sort_index()
    turnover_wide = long.pivot(index="date", columns="stock_id",
                                values="turnover").sort_index()
    market_returns = prices_wide["TAIEX"].pct_change()
    print(f"  prices: {prices_wide.shape}  volume: {volume_wide.shape}")
    return prices_wide, volume_wide, turnover_wide, market_returns


# ── 2. Fetch PER (dividend_yield) + shares + industry, with caches ──────


def fetch_or_load_per(stock_ids):
    if PER_CACHE.exists():
        print(f"  loading PER cache → {PER_CACHE.name}")
        return pd.read_parquet(PER_CACHE)
    print(f"  fetching PER for {len(stock_ids)} stocks (rate-limited)…")
    df = fetch_per_multiple(stock_ids, DATA_START, DATA_END, sleep_seconds=0.6)
    if df.empty:
        raise RuntimeError("PER fetch returned empty — FinMind error")
    df.to_parquet(PER_CACHE)
    return df


def fetch_or_load_shares(stock_ids):
    if SHARES_CACHE.exists():
        print(f"  loading shares cache → {SHARES_CACHE.name}")
        return pd.read_parquet(SHARES_CACHE)
    print(f"  fetching shares for {len(stock_ids)} stocks (rate-limited)…")
    df = fetch_shares_multiple(stock_ids, DATA_START, DATA_END, sleep_seconds=0.6)
    if df.empty:
        raise RuntimeError("Shares fetch returned empty — FinMind error")
    df.to_parquet(SHARES_CACHE)
    return df


def fetch_or_load_info():
    if INFO_CACHE.exists():
        print(f"  loading info cache → {INFO_CACHE.name}")
        return pd.read_parquet(INFO_CACHE)
    print("  fetching TaiwanStockInfo (one API call)…")
    resp = requests.get("https://api.finmindtrade.com/api/v4/data",
        params={"dataset": "TaiwanStockInfo"}, timeout=30)
    j = resp.json()
    data = j.get("data", [])
    if not data:
        raise RuntimeError(f"TaiwanStockInfo fetch failed: {j.get('msg', '?')}")
    df = pd.DataFrame(data)
    df.to_parquet(INFO_CACHE)
    return df


# ── 3. Build cross-section dataset ───────────────────────────────────────


def build_panel(events, prices_wide, volume_wide, turnover_wide,
                market_returns, per_df, shares_df, info_df):
    industry_map = (
        info_df.drop_duplicates(subset=["stock_id"], keep="last")
        .set_index("stock_id")["industry_category"].to_dict()
    )
    name_map = (
        info_df.drop_duplicates(subset=["stock_id"], keep="last")
        .set_index("stock_id")["stock_name"].to_dict()
    )

    records = []
    missing = []

    for _, ev in events.iterrows():
        ann = pd.Timestamp(ev["announcement_date"])
        eff = pd.Timestamp(ev["effective_date"])
        added = ev["added_stocks"]
        n_add = len(added)

        for sid in added:
            row = {
                "event_id": ev["event_id"],
                "ann_date": ann,
                "stock_id": sid,
                "stock_name": name_map.get(sid, ""),
                "industry": industry_map.get(sid, "Unknown"),
            }

            # ── CARs (the four target outcomes) ──
            row["CAR_pre_30"]  = _last_car(sid, ann, prices_wide,
                                            market_returns, (-30, 0))
            row["CAR_pre_5"]   = _last_car(sid, ann, prices_wide,
                                            market_returns, (-5, 0))
            row["CAR_post_5"]  = _last_car(sid, ann, prices_wide,
                                            market_returns, (0, 5))
            row["CAR_post_30"] = _last_car(sid, ann, prices_wide,
                                            market_returns, (1, 31))

            # ── Features ──
            # Prior trading day close
            pre = prices_wide.index[prices_wide.index < ann]
            if len(pre) == 0 or sid not in prices_wide.columns:
                price_prev = np.nan
            else:
                p = prices_wide.loc[pre[-1], sid]
                price_prev = float(p) if pd.notna(p) and p > 0 else np.nan
            row["price_prev"] = price_prev

            # Market cap = close × shares
            shares = _prior_value(shares_df, sid, ann, "shares_issued")
            row["shares_issued"] = shares
            row["market_cap"] = (
                price_prev * shares
                if (pd.notna(price_prev) and pd.notna(shares)) else np.nan
            )

            # Avg daily turnover (TWD) over 60 trading days
            if sid in turnover_wide.columns:
                w = pre[-60:]
                av = float(turnover_wide.loc[w, sid].mean())
                row["avg_turnover_60d"] = av if pd.notna(av) else np.nan
            else:
                row["avg_turnover_60d"] = np.nan

            # Turnover ratio = avg daily shares traded / shares outstanding
            row["turnover_ratio_60d"] = _compute_turnover_ratio(
                volume_wide, shares_df, sid, ann
            )

            # Dividend yield (last published, strictly before ann)
            row["dividend_yield"] = _prior_value(per_df, sid, ann, "dividend_yield")

            # Momentum
            row["momentum_60d"] = _compute_momentum(prices_wide, sid, ann,
                                                    lookback=60, skip=5)
            row["momentum_recent_5d"] = _compute_recent_momentum(
                prices_wide, sid, ann, n=5
            )

            # Log transforms
            mc = row["market_cap"]
            av = row["avg_turnover_60d"]
            row["log_market_cap"]   = float(np.log(mc)) if (pd.notna(mc) and mc > 0) else np.nan
            row["log_avg_turnover"] = float(np.log(av)) if (pd.notna(av) and av > 0) else np.nan

            # Track missingness
            for feat in ["market_cap", "avg_turnover_60d", "turnover_ratio_60d",
                         "dividend_yield", "momentum_60d", "momentum_recent_5d"]:
                if pd.isna(row.get(feat)):
                    missing.append({
                        "event_id": ev["event_id"],
                        "stock_id": sid,
                        "feature": feat,
                    })

            records.append(row)

    panel = pd.DataFrame(records)

    # ── is_high_yield: top 30 by dividend_yield within each event ──
    panel["is_high_yield"] = False
    for eid in panel["event_id"].unique():
        sub = panel[panel["event_id"] == eid].copy()
        # Within the event's added stocks, mark top-N by dividend_yield as 1
        # (if N_added > 5, top 30%; otherwise all = True effectively)
        n_top = max(1, int(np.ceil(len(sub) * 0.30)))
        top_idx = sub.nlargest(n_top, "dividend_yield").index
        panel.loc[top_idx, "is_high_yield"] = True

    # Winsorize extreme variables (preserve NaN)
    for col in ["market_cap", "avg_turnover_60d", "turnover_ratio_60d",
                "dividend_yield", "momentum_60d", "momentum_recent_5d"]:
        if col in panel.columns:
            panel[col + "_w"] = _winsorize(panel[col], 0.01, 0.99)

    return panel, pd.DataFrame(missing)


# ── 4. Describe ─────────────────────────────────────────────────────────


def write_describe(panel):
    feats = ["CAR_pre_30", "CAR_pre_5", "CAR_post_5", "CAR_post_30",
             "market_cap", "log_market_cap",
             "avg_turnover_60d", "log_avg_turnover",
             "turnover_ratio_60d",
             "dividend_yield",
             "momentum_60d", "momentum_recent_5d"]
    feats = [f for f in feats if f in panel.columns]
    desc = panel[feats].describe().T
    desc["missing"] = len(panel) - panel[feats].count()
    desc = desc[["count", "missing", "mean", "std", "min",
                 "25%", "50%", "75%", "max"]]
    desc.to_csv(OUT_DESC, float_format="%.4f")
    print(f"\n✓ Describe table → {OUT_DESC.relative_to(ROOT)}")
    print(desc.to_string())
    return desc


# ── 5. Group analysis (low/mid/high tertile) ─────────────────────────────


def plot_group_car(panel, feature, outcomes=("CAR_pre_30", "CAR_post_30"),
                   save_path=None, label_x=None):
    sub = panel[[feature, *outcomes]].dropna(subset=[feature]).copy()
    if len(sub) < 9:
        print(f"  ⚠ {feature}: N={len(sub)} too few for tertiles; skipping")
        return

    try:
        sub["tercile"] = pd.qcut(sub[feature], q=3,
                                  labels=["Low", "Mid", "High"], duplicates="drop")
    except ValueError:
        print(f"  ⚠ {feature}: qcut failed (duplicate bin edges); skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    colors = ["#5b9bd5", "#a5a5a5", "#ed7d31"]

    for ax, outcome in zip(axes, outcomes):
        s = sub.dropna(subset=[outcome])
        if s.empty:
            ax.set_title(f"{outcome} (no data)")
            continue
        agg = (s.groupby("tercile", observed=True)[outcome]
               .agg(["mean", "std", "count"])
               .reindex(["Low", "Mid", "High"]))
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
        title_kor = {"CAR_pre_30": "公告前 30 日 CAR",
                     "CAR_post_30": "生效後 30 日 CAR",
                     "CAR_pre_5": "公告前 5 日 CAR",
                     "CAR_post_5": "公告後 5 日 CAR"}.get(outcome, outcome)
        ax.set_title(title_kor, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    xlabel = label_x or feature
    fig.suptitle(f"[REAL DATA] 分組分析: {xlabel} (低 / 中 / 高三分位)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {feature} → {save_path.name}")


# ── 6. Industry analysis ─────────────────────────────────────────────────


def plot_industry(panel, save_path):
    sub = panel.dropna(subset=["CAR_pre_30", "CAR_post_30"]).copy()
    # Industries with N < 5 → "Other"
    counts = sub["industry"].value_counts()
    sub["industry_grp"] = sub["industry"].where(
        sub["industry"].isin(counts[counts >= 5].index), "Other"
    )

    agg = (sub.groupby("industry_grp")
           .agg(N=("CAR_pre_30", "count"),
                mean_pre=("CAR_pre_30", "mean"),
                std_pre=("CAR_pre_30", "std"),
                mean_post=("CAR_post_30", "mean"),
                std_post=("CAR_post_30", "std"))
           .sort_values("N", ascending=False))
    agg["ci_pre"]  = 1.96 * agg["std_pre"]  / np.sqrt(agg["N"])
    agg["ci_post"] = 1.96 * agg["std_post"] / np.sqrt(agg["N"])

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.4 * len(agg) + 1.5)))
    y = np.arange(len(agg))

    for ax, (mean_col, ci_col, title) in zip(
        axes,
        [("mean_pre", "ci_pre", "公告前 30 日 CAR"),
         ("mean_post", "ci_post", "生效後 30 日 CAR")]
    ):
        colors = ["#2ca02c" if m > 0 else "#d62728" for m in agg[mean_col]]
        ax.barh(y, agg[mean_col] * 100, xerr=agg[ci_col] * 100,
                color=colors, alpha=0.8, capsize=4, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{ind} (N={n})" for ind, n in zip(agg.index, agg["N"])],
                           fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.set_xlabel("CAR (%)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.invert_yaxis()

    fig.suptitle("[REAL DATA] 產業分組 — CAR 比較（N ≥ 5 的產業；其餘合併為 Other）",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ industry → {save_path.name}")
    return agg


# ── 7. Regression ────────────────────────────────────────────────────────


REG_FEATURES = [
    "log_market_cap",
    "log_avg_turnover",
    "turnover_ratio_60d",
    "dividend_yield",
    "momentum_60d",
    "momentum_recent_5d",
]


def run_regression(panel, outcome):
    cols = ["event_id", "stock_id", outcome] + REG_FEATURES
    sub = panel[cols].dropna(subset=[outcome] + REG_FEATURES).reset_index(drop=True)
    X = sm.add_constant(sub[REG_FEATURES])
    y = sub[outcome]
    model = sm.OLS(y, X).fit()
    return model, sub


def plot_diagnostics(model, sub, outcome, save_path):
    fitted = model.fittedvalues
    resid = model.resid

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Residuals vs Fitted
    axes[0].scatter(fitted, resid, alpha=0.6, color="#1f77b4", edgecolor="white")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].grid(linestyle=":", alpha=0.5)

    # Q-Q plot
    sm.qqplot(resid, line="45", fit=True, ax=axes[1])
    axes[1].set_title("Normal Q-Q")
    axes[1].grid(linestyle=":", alpha=0.5)

    # Cook's distance
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]
    axes[2].stem(range(len(cooks)), cooks,
                 linefmt="C0-", markerfmt="C0o", basefmt=" ")
    threshold = 4 / len(cooks)
    axes[2].axhline(threshold, color="red", linestyle="--", linewidth=1,
                    label=f"4/N = {threshold:.3f}")
    axes[2].set_xlabel("Observation index")
    axes[2].set_ylabel("Cook's distance")
    axes[2].set_title(f"Cook's distance (top {sum(cooks > threshold)} flagged)")
    axes[2].legend(fontsize=8)
    axes[2].grid(linestyle=":", alpha=0.5)

    fig.suptitle(f"[REAL DATA] Regression Diagnostics — {outcome}  (N = {len(sub)})",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # List top-5 Cook's
    top_idx = np.argsort(cooks)[-5:][::-1]
    top_rows = sub.iloc[top_idx].copy()
    top_rows["cooks_d"] = cooks[top_idx]
    return top_rows


# ── 8. Main ──────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  Stage 3 — Cross-sectional Analysis (REAL DATA)")
    print("=" * 70)

    # Load price panels
    prices_wide, volume_wide, turnover_wide, market_returns = load_panels()

    # Load events
    events = load_events(EVENTS_CSV)
    print(f"Events: {len(events)}")

    # Universe of added stocks (deduplicated)
    all_added = sorted({sid for lst in events["added_stocks"] for sid in lst})
    print(f"Unique added stocks: {len(all_added)}  (event×stock = {sum(len(l) for l in events['added_stocks'])})")

    # ── Pre-fetch auxiliary data ──
    print("\n[Aux fetch]")
    per_df    = fetch_or_load_per(all_added)
    shares_df = fetch_or_load_shares(all_added)
    info_df   = fetch_or_load_info()
    print(f"  PER rows    : {len(per_df):,}")
    print(f"  shares rows : {len(shares_df):,}")
    print(f"  info rows   : {len(info_df):,}")

    # ── Build cross-section panel ──
    print("\n[Build panel]")
    panel, missing = build_panel(events, prices_wide, volume_wide, turnover_wide,
                                  market_returns, per_df, shares_df, info_df)
    panel.to_csv(OUT_DATASET, index=False, float_format="%.6f")
    print(f"✓ Panel rows = {len(panel)} → {OUT_DATASET.relative_to(ROOT)}")
    if not missing.empty:
        missing.to_csv(OUT_MISS, index=False)
        miss_summary = missing.groupby("feature").size().sort_values(ascending=False)
        print(f"  Missing report → {OUT_MISS.relative_to(ROOT)}")
        print(f"  Missing counts by feature:\n{miss_summary.to_string()}")

    # ── Describe ──
    print("\n[Describe]")
    desc = write_describe(panel)

    # ── Group analyses ──
    print("\n[Group analysis (tertile)]")
    plots = [
        ("log_market_cap",      "log(市值)"),
        ("log_avg_turnover",    "log(60 日平均成交金額)"),
        ("turnover_ratio_60d",  "60 日周轉率"),
        ("dividend_yield",      "殖利率 (%)"),
        ("momentum_60d",        "60 日動能（剔除最近 5 日）"),
        ("momentum_recent_5d",  "公告前 5 日動能"),
    ]
    for feat, label in plots:
        save = FIG_DIR / f"group_car_{feat}.png"
        plot_group_car(panel, feat, outcomes=("CAR_pre_30", "CAR_post_30"),
                       save_path=save, label_x=label)

    # ── Industry analysis ──
    print("\n[Industry analysis]")
    ind_agg = plot_industry(panel, FIG_DIR / "group_car_by_industry.png")

    # ── Regression ──
    print("\n[Regression]")
    model_pre,  sub_pre  = run_regression(panel, "CAR_pre_30")
    model_post, sub_post = run_regression(panel, "CAR_post_30")

    print(f"\n  CAR_pre_30 regression : N = {len(sub_pre)}")
    print(f"  CAR_post_30 regression: N = {len(sub_post)}")
    if min(len(sub_pre), len(sub_post)) < 40:
        print(f"  ⚠ WARNING: post-dropna N < 40; reduce predictors recommended")

    with open(OUT_REG, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  Cross-Sectional Regression  (REAL DATA, Stage 3)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Window      : ann_date-30 to eff_date+30\n")
        f.write(f"N_events    : {len(events)}\n")
        f.write(f"N_pairs     : {len(panel)} (event × added_stock)\n\n")
        f.write("Predictors  : " + ", ".join(REG_FEATURES) + "\n")
        f.write("Excluded    : impact_ratio (no AUM data), target_weight (no allocation data)\n\n")

        f.write("─" * 70 + "\n")
        f.write("MODEL 1: CAR_pre_30 = pre-announcement run-up\n")
        f.write("─" * 70 + "\n")
        f.write(str(model_pre.summary()))
        f.write("\n\n")

        f.write("─" * 70 + "\n")
        f.write("MODEL 2: CAR_post_30 = post-effective drift\n")
        f.write("─" * 70 + "\n")
        f.write(str(model_post.summary()))
        f.write("\n\n")

        f.write("─" * 70 + "\n")
        f.write("INDUSTRY MEAN CAR (N >= 5)\n")
        f.write("─" * 70 + "\n")
        f.write(ind_agg.to_string())
        f.write("\n")

    print(f"\n✓ Regression report → {OUT_REG.relative_to(ROOT)}")

    # ── Diagnostics ──
    print("\n[Diagnostics]")
    top_pre  = plot_diagnostics(model_pre,  sub_pre,  "CAR_pre_30",
                                FIG_DIR / "regression_diagnostics_pre.png")
    top_post = plot_diagnostics(model_post, sub_post, "CAR_post_30",
                                FIG_DIR / "regression_diagnostics_post.png")
    print(f"  Top-5 Cook's distance (pre):\n{top_pre[['event_id','stock_id','cooks_d']].to_string(index=False)}")
    print(f"\n  Top-5 Cook's distance (post):\n{top_post[['event_id','stock_id','cooks_d']].to_string(index=False)}")

    # ── Text summary ──
    print("\n" + "=" * 70)
    print("  TEXT SUMMARY")
    print("=" * 70)

    def _sig(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    for name, m, n in [("CAR_pre_30", model_pre, len(sub_pre)),
                       ("CAR_post_30", model_post, len(sub_post))]:
        print(f"\n{name}  (N = {n}, R² = {m.rsquared:.3f}, adj-R² = {m.rsquared_adj:.3f})")
        for feat in REG_FEATURES:
            coef = m.params[feat]
            t    = m.tvalues[feat]
            p    = m.pvalues[feat]
            print(f"  {feat:24s}  β = {coef:+.4f}  t = {t:+.3f}  p = {p:.4f}  {_sig(p)}")


if __name__ == "__main__":
    main()
