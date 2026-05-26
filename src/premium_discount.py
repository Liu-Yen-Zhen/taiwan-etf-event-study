"""
src/premium_discount.py
-----------------------
ETF premium / discount structural observations.

POSITIONING
-----------
This module produces **descriptive observations only**.
  - No strategy backtest, no P&L calculation, no Sharpe ratio.
  - All findings are described as structural phenomena, NOT trading signals.
  - Data may be real (FinMind) or synthetic (clearly labelled).

Three structural questions
--------------------------
1. Premium distribution — how wide is the premium/discount band for
   high-dividend ETFs (00919, 00878)?  Are they systematically above NAV?
2. Dividend cycle — does the premium expand before ex-dividend dates and
   contract afterwards?  (Dividend-chasing behaviour hypothesis.)
3. Volatility ETF asymmetry — does 00632R (富邦 VIX) trade at larger premium
   on high-stress days (VIX-spike / market-drop)?

All charts are saved to output/figures/ with filenames containing the ETF code.
All statistics are printed in-notebook for transparency.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
from _plot_config import apply_chinese_style
apply_chinese_style()
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FINMIND_URL   = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")


# ── FinMind helpers ────────────────────────────────────────────────────────────


def _fm_get(dataset: str, stock_id: str, start_date: str, end_date: str) -> list[dict]:
    """Single FinMind API request.  Returns raw data list or [] on failure."""
    params = {
        "dataset":    dataset,
        "data_id":    stock_id,
        "start_date": start_date,
        "end_date":   end_date,
    }
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN
    try:
        resp = requests.get(FINMIND_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != 200:
            logger.warning("FinMind %s/%s → status %s: %s",
                           dataset, stock_id, payload.get("status"), payload.get("msg"))
            return []
        time.sleep(0.5)
        return payload.get("data", [])
    except Exception as exc:
        logger.warning("FinMind fetch failed (%s/%s): %s", dataset, stock_id, exc)
        return []


def _fetch_close_prices(etf_id: str, start_date: str, end_date: str) -> pd.Series:
    """Fetch daily close price from FinMind TaiwanStockPrice.  Returns empty Series on failure."""
    rows = _fm_get("TaiwanStockPrice", etf_id, start_date, end_date)
    if not rows:
        return pd.Series(dtype=float, name="close")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df["close"].rename("close")


def _fetch_etf_dividends(etf_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch ETF dividend records from FinMind TaiwanETFDividend.

    Returns DataFrame with columns: [date, stock_id, dividend, nav, ...].
    Returns empty DataFrame if data unavailable (free-tier limitation).
    """
    rows = _fm_get("TaiwanETFDividend", etf_id, start_date, end_date)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


# ── Synthetic data helpers ─────────────────────────────────────────────────────
# ⛔  DEPRECATED — 以下三個函式僅供舊 notebook 示範用，禁止在真實研究流程中呼叫。
#     呼叫前請先確認使用的是真實 NAV 資料。


def _synthetic_premium_series(
    trading_days: pd.DatetimeIndex,
    mean_premium: float = 0.005,
    ar_coef: float = 0.85,
    sigma: float = 0.003,
    rng: Optional[np.random.Generator] = None,
) -> pd.Series:
    """⛔ DEPRECATED — 產生合成折溢價序列，禁止用於真實研究。

    Generate a synthetic AR(1) premium series.
    premium_t = mean_premium + ar_coef * (premium_{t-1} - mean_premium) + eps_t
    """
    raise RuntimeError(
        "_synthetic_premium_series 已停用。請改用真實每日淨值（NAV）資料計算折溢價。"
    )
    # ── 以下為原始合成資料程式碼，僅供參考 ──
    if rng is None:
        rng = np.random.default_rng()
    n = len(trading_days)
    eps = rng.normal(0, sigma, size=n)
    prem = np.zeros(n)
    prem[0] = mean_premium + eps[0]
    for i in range(1, n):
        prem[i] = mean_premium + ar_coef * (prem[i - 1] - mean_premium) + eps[i]
    return pd.Series(prem, index=trading_days, name="premium_pct")


def _inject_dividend_spikes(
    premium: pd.Series,
    ex_div_dates: list,
    trading_days: pd.DatetimeIndex,
    pre_window: int = 10,
    spike_size: float = 0.008,
    decay_rate: float = 0.7,
) -> pd.Series:
    """⛔ DEPRECATED — 注入合成除息衝擊，禁止用於真實研究。"""
    raise RuntimeError(
        "_inject_dividend_spikes 已停用。請改用真實除息日與 NAV 資料進行分析。"
    )
    # ── 以下為原始合成資料程式碼，僅供參考 ──
    prem = premium.copy()
    for ex_date in ex_div_dates:
        ex_td = ex_date if ex_date in trading_days else (
            trading_days[trading_days >= ex_date][0]
            if (trading_days >= ex_date).any() else None
        )
        if ex_td is None:
            continue
        idx = trading_days.get_loc(ex_td)
        for d in range(-pre_window, 0):
            i = idx + d
            if 0 <= i < len(trading_days):
                fade = (pre_window + d) / pre_window   # 0→1 approaching ex-date
                prem.iloc[i] += spike_size * fade
        # Post ex-div: brief discount
        for d in range(0, 5):
            i = idx + d
            if 0 <= i < len(trading_days):
                prem.iloc[i] -= spike_size * 0.4 * (decay_rate ** d)
    return prem


def _inject_stress_spikes(
    premium: pd.Series,
    market_returns: pd.Series,
    stress_threshold: float = -0.015,
    spike_scale: float = 8.0,
) -> pd.Series:
    """⛔ DEPRECATED — 注入合成壓力日衝擊，禁止用於真實研究。"""
    raise RuntimeError(
        "_inject_stress_spikes 已停用。請改用真實折溢價與市場報酬資料進行分析。"
    )
    # ── 以下為原始合成資料程式碼，僅供參考 ──
    prem = premium.copy()
    aligned = market_returns.reindex(prem.index)
    stress_mask = aligned < stress_threshold
    # Positive premium spike proportional to how negative the market return is
    prem[stress_mask] += -aligned[stress_mask] * spike_scale
    return prem


# ── Public API ─────────────────────────────────────────────────────────────────


def fetch_etf_premium_data(
    etf_id: str,
    start_date: str,
    end_date: str,
    nav_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch ETF daily close price (FinMind) and NAV (caller-supplied CSV).

    ⛔  合成 NAV 路徑已移除。NAV 必須由呼叫者提供真實資料。

    Parameters
    ----------
    etf_id : str
        FinMind stock_id, e.g. "00919", "00878", "00632R".
    start_date, end_date : str
        ISO date strings "YYYY-MM-DD".
    nav_csv_path : str or None
        Path to a CSV with columns [date, nav].  date = ISO string or datetime.
        If None, raises RuntimeError.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex.  Columns: close, nav, premium_pct (%).
    """
    # ── Close price (真實 FinMind 資料) ──
    close_real = _fetch_close_prices(etf_id, start_date, end_date)
    if close_real.empty:
        raise RuntimeError(
            f"{etf_id}: FinMind 回傳空資料。請確認 stock_id 正確、日期範圍有交易日、"
            "且網路/token 正常。"
        )

    # ── NAV (呼叫者必須提供) ──
    if nav_csv_path is None:
        raise RuntimeError(
            f"{etf_id}: 未提供 nav_csv_path。每日 NAV 資料無法從 FinMind 免費層取得，"
            "請自行從 TWSE/投信公告下載後傳入 CSV 路徑。\n"
            "CSV 格式：date(YYYY-MM-DD), nav(float)"
        )

    nav_df = pd.read_csv(nav_csv_path, parse_dates=["date"]).set_index("date").sort_index()
    if "nav" not in nav_df.columns:
        raise ValueError(f"nav_csv_path={nav_csv_path} 缺少 'nav' 欄位")

    nav_series = pd.to_numeric(nav_df["nav"], errors="coerce")
    if nav_series.dropna().empty:
        raise ValueError(f"nav_csv_path={nav_csv_path} 的 nav 欄位全為 NaN")

    # ── 合併，對齊到 close 的交易日 ──
    close = close_real
    nav   = nav_series.reindex(close.index).interpolate(method="time")

    missing_nav = nav.isna().sum()
    if missing_nav > 0:
        logger.warning("%s: %d 個交易日 NAV 缺值（線性內插無法填補），已設為 NaN",
                       etf_id, missing_nav)

    df = pd.DataFrame({"close": close, "nav": nav}).dropna()
    if df.empty:
        raise RuntimeError(
            f"{etf_id}: close 與 NAV 合併後無交集，請確認日期範圍一致。"
        )

    df["premium_pct"] = (df["close"] - df["nav"]) / df["nav"] * 100
    df.index.name = "date"

    logger.info("%s: %d 交易日  premium mean=%.3f%%  std=%.3f%%",
                etf_id, len(df), df["premium_pct"].mean(), df["premium_pct"].std())
    return df


# ── Analysis functions ─────────────────────────────────────────────────────────


def analyze_premium_distribution(
    premium_df: pd.DataFrame,
    etf_id: str,
    save_dir: str | Path = "output/figures",
    title_prefix: str = "",
) -> dict:
    """Plot premium/discount time series and distribution; compute descriptive stats.

    Parameters
    ----------
    premium_df : pd.DataFrame
        Output of fetch_etf_premium_data().  Must have ``premium_pct`` column.
    etf_id : str
        Used for file naming and chart titles.
    save_dir : str or Path
        Destination directory.
    title_prefix : str
        Prepended to chart title (e.g. ETF display name).

    Returns
    -------
    dict
        Descriptive statistics: mean, median, std, pct5, pct95, n_premium, n_discount.
    """
    df = premium_df.dropna(subset=["premium_pct"]).copy()
    pct = df["premium_pct"]

    stats = {
        "mean":        float(pct.mean()),
        "median":      float(pct.median()),
        "std":         float(pct.std()),
        "pct5":        float(pct.quantile(0.05)),
        "pct95":       float(pct.quantile(0.95)),
        "n_premium":   int((pct > 0).sum()),
        "n_discount":  int((pct <= 0).sum()),
        "pct_days_premium": float((pct > 0).mean() * 100),
    }

    name = title_prefix or etf_id
    fig, (ax_ts, ax_hist) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # ── Time series ──
    pos = pct.where(pct > 0)
    neg = pct.where(pct <= 0)
    ax_ts.plot(df.index, pct, color="#7f7f7f", linewidth=0.7, alpha=0.5)
    ax_ts.fill_between(df.index, 0, pos, alpha=0.30, color="#2ca02c", label="溢價（高於淨值）")
    ax_ts.fill_between(df.index, 0, neg, alpha=0.30, color="#d62728", label="折價（低於淨值）")
    ax_ts.axhline(0, color="black", linewidth=0.9)
    ax_ts.axhline(stats["mean"], color="#1f77b4", linewidth=1.2, linestyle="--",
                  label=f"均值 {stats['mean']:+.3f}%")
    ax_ts.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:+.2f}%"))
    ax_ts.set_ylabel("折溢價率 (%)", fontsize=11)
    ax_ts.set_title(f"{name} — 折溢價時序（觀察，非策略）", fontsize=12, fontweight="bold")
    ax_ts.legend(fontsize=9)
    ax_ts.grid(axis="y", linestyle=":", alpha=0.5)

    # ── Histogram ──
    ax_hist.hist(pct, bins=60, color="#1f77b4", alpha=0.65, edgecolor="white")
    ax_hist.axvline(0, color="black", linewidth=0.9)
    ax_hist.axvline(stats["mean"],   color="#ff7f0e", linewidth=1.5,
                    linestyle="--", label=f"均值 {stats['mean']:+.3f}%")
    ax_hist.axvline(stats["pct5"],   color="#9467bd", linewidth=1.2, linestyle=":",
                    label=f"5th pct {stats['pct5']:+.2f}%")
    ax_hist.axvline(stats["pct95"],  color="#9467bd", linewidth=1.2, linestyle=":",
                    label=f"95th pct {stats['pct95']:+.2f}%")
    ax_hist.set_xlabel("折溢價率 (%)", fontsize=10)
    ax_hist.set_ylabel("天數", fontsize=10)
    ax_hist.set_title(
        f"折溢價分布  std={stats['std']:.3f}%  "
        f"溢價日占比={stats['pct_days_premium']:.1f}%",
        fontsize=10,
    )
    ax_hist.legend(fontsize=8)
    ax_hist.grid(axis="y", linestyle=":", alpha=0.5)

    fig.text(
        0.5, 0.01,
        "純觀察性分析 — 非交易訊號 — 不構成投資建議",
        ha="center", fontsize=8, color="gray", style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"premium_dist_{etf_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Premium distribution chart saved -> %s", out)
    return stats


def analyze_dividend_cycle(
    premium_df: pd.DataFrame,
    ex_dividend_dates: list,
    etf_id: str = "",
    window: int = 20,
    save_dir: str | Path = "output/figures",
) -> pd.DataFrame:
    """Observe average premium/discount path around ex-dividend dates.

    For each ex-dividend date, extract the [-window, +window] trading-day
    window of premium_pct.  Average across all events.

    The resulting chart shows whether the premium systematically expands
    before ex-dividend and contracts (or turns to discount) after.

    Parameters
    ----------
    premium_df : pd.DataFrame
        Output of fetch_etf_premium_data().
    ex_dividend_dates : list of date-like
        Ex-dividend dates to centre the event windows on.
    etf_id : str
        Used in file naming.
    window : int
        Half-window in trading days (−window to +window).
    save_dir : str or Path
        Destination directory.

    Returns
    -------
    pd.DataFrame
        Columns: relative_day, mean_premium, std_premium, n.
    """
    df = premium_df.dropna(subset=["premium_pct"]).copy()
    trading_days = df.index

    records = []
    for ex_raw in ex_dividend_dates:
        ex_ts   = pd.Timestamp(ex_raw)
        future  = trading_days[trading_days >= ex_ts]
        if future.empty:
            continue
        ex_td  = future[0]
        base_i = trading_days.get_loc(ex_td)

        for d in range(-window, window + 1):
            i = base_i + d
            if 0 <= i < len(trading_days):
                records.append({
                    "relative_day": d,
                    "premium_pct":  float(df["premium_pct"].iloc[i]),
                })

    if not records:
        logger.warning("analyze_dividend_cycle: no valid event windows found")
        return pd.DataFrame()

    event_df = pd.DataFrame(records)
    agg = (
        event_df.groupby("relative_day")["premium_pct"]
        .agg(mean_premium="mean", std_premium="std", n="count")
        .reset_index()
    )

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(agg["relative_day"], agg["mean_premium"], "o-",
            color="#1f77b4", linewidth=2, markersize=4, label="平均折溢價率")
    ci = 1.96 * agg["std_premium"] / np.sqrt(agg["n"])
    ax.fill_between(
        agg["relative_day"],
        agg["mean_premium"] - ci,
        agg["mean_premium"] + ci,
        alpha=0.20, color="#1f77b4", label="95% CI",
    )
    ax.axvline(0, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label="除息日")
    ax.axhline(0, color="black", linewidth=0.8)

    # Shade pre-dividend region
    ax.axvspan(-window, 0, alpha=0.04, color="#2ca02c", label="除息前")
    ax.axvspan(0, window,  alpha=0.04, color="#d62728", label="除息後")

    # Annotate pre/post means
    pre_mean  = agg.loc[agg["relative_day"] < 0,  "mean_premium"].mean()
    post_mean = agg.loc[agg["relative_day"] > 0,  "mean_premium"].mean()
    ax.annotate(f"除息前均值\n{pre_mean:+.3f}%",
                xy=(-window // 2, pre_mean),
                fontsize=9, color="#2ca02c", ha="center")
    ax.annotate(f"除息後均值\n{post_mean:+.3f}%",
                xy=(window // 2, post_mean),
                fontsize=9, color="#d62728", ha="center")

    ax.set_xlabel("相對除息日交易日數（負=除息前）", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:+.3f}%"))
    ax.set_ylabel("平均折溢價率 (%)", fontsize=11)
    etf_label = f" ({etf_id})" if etf_id else ""
    ax.set_title(
        f"除息週期折溢價路徑{etf_label}  [結構性觀察，非策略建議]",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.text(
        0.5, 0.01,
        "純觀察性分析 — 非交易訊號 — 不構成投資建議",
        ha="center", fontsize=8, color="gray", style="italic",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "premium_dividend_cycle.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Dividend cycle chart saved -> %s", out)
    return agg


def analyze_inverse_etf_asymmetry(
    inverse_etf_premium: pd.Series,
    market_returns: pd.Series,
    stress_threshold: float = -0.015,
    etf_id: str = "00632R",
    save_dir: str | Path = "output/figures",
) -> dict:
    """Observe premium asymmetry between stress days and normal days.

    Compares the premium/discount distribution on days when market returns
    fall below ``stress_threshold`` vs all other days.  This tests whether
    there is a structural tendency for the premium to expand during market stress.

    Parameters
    ----------
    inverse_etf_premium : pd.Series
        Daily premium_pct (%) with DatetimeIndex.
    market_returns : pd.Series
        Daily market returns (fraction, not %) with DatetimeIndex.
    stress_threshold : float
        Market return below which a day is labelled "stress day".
        Default −1.5 %.
    etf_id : str
        Used for file naming and titles.
    save_dir : str or Path
        Destination directory.

    Returns
    -------
    dict
        Keys: stress_days_n, normal_days_n, stress_mean, normal_mean,
              stress_std, normal_std, stress_pct95, normal_pct95.
    """
    prem = inverse_etf_premium.dropna()
    mkt  = market_returns.reindex(prem.index).dropna()
    prem = prem.reindex(mkt.index)

    stress_mask  = mkt < stress_threshold
    stress_prem  = prem[stress_mask]
    normal_prem  = prem[~stress_mask]

    result = {
        "stress_days_n":  int(stress_mask.sum()),
        "normal_days_n":  int((~stress_mask).sum()),
        "stress_mean":    float(stress_prem.mean()),
        "normal_mean":    float(normal_prem.mean()),
        "stress_std":     float(stress_prem.std()),
        "normal_std":     float(normal_prem.std()),
        "stress_pct95":   float(stress_prem.quantile(0.95)),
        "normal_pct95":   float(normal_prem.quantile(0.95)),
        "stress_threshold_pct": stress_threshold * 100,
    }

    # ── Plot: dual-panel ──
    fig, (ax_box, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

    # Box plot comparison
    ax_box.boxplot(
        [stress_prem.values, normal_prem.values],
        labels=[f"壓力日\n(市場≤{stress_threshold*100:.1f}%)\nn={result['stress_days_n']}",
                f"一般日\nn={result['normal_days_n']}"],
        patch_artist=True,
        boxprops=dict(facecolor="#ff7f0e", alpha=0.6),
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
        widths=0.4,
    )
    ax_box.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_box.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:+.2f}%"))
    ax_box.set_ylabel("折溢價率 (%)", fontsize=11)
    ax_box.set_title(f"{etf_id} 折溢價分布：壓力日 vs 一般日", fontsize=11, fontweight="bold")
    ax_box.grid(axis="y", linestyle=":", alpha=0.5)

    # Annotate means
    for x, val, clr in [(1, result["stress_mean"], "#d62728"),
                        (2, result["normal_mean"], "#2ca02c")]:
        ax_box.annotate(
            f"均值 {val:+.3f}%",
            xy=(x, val), xytext=(x + 0.25, val),
            fontsize=9, color=clr,
            arrowprops=dict(arrowstyle="->", color=clr, lw=0.8),
        )

    # Overlapping histogram
    bins = np.linspace(
        min(stress_prem.min(), normal_prem.min()),
        max(stress_prem.max(), normal_prem.max()),
        40,
    )
    ax_hist.hist(normal_prem, bins=bins, color="#1f77b4", alpha=0.55,
                 label=f"一般日 (n={result['normal_days_n']})", edgecolor="white")
    ax_hist.hist(stress_prem, bins=bins, color="#ff7f0e", alpha=0.65,
                 label=f"壓力日 (n={result['stress_days_n']})", edgecolor="white")
    ax_hist.axvline(result["normal_mean"], color="#1f77b4", linewidth=1.5,
                    linestyle="--", label=f"一般日均值 {result['normal_mean']:+.3f}%")
    ax_hist.axvline(result["stress_mean"], color="#ff7f0e", linewidth=1.5,
                    linestyle="--", label=f"壓力日均值 {result['stress_mean']:+.3f}%")
    ax_hist.set_xlabel("折溢價率 (%)", fontsize=10)
    ax_hist.set_ylabel("天數", fontsize=10)
    ax_hist.set_title("折溢價分布密度比較", fontsize=11, fontweight="bold")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        f"{etf_id} 波動型 ETF：市場壓力日折溢價結構性觀察",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.02,
        "純觀察性分析 — 非交易訊號 — 不構成投資建議",
        ha="center", fontsize=8, color="gray", style="italic",
    )
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"premium_inverse_asymmetry_{etf_id}.png".replace("/", "_")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Inverse ETF asymmetry chart saved -> %s", out)
    return result


def plot_premium_comparison(
    premium_dict: dict[str, pd.DataFrame],
    save_dir: str | Path = "output/figures",
    title: str = "高股息 ETF 折溢價比較（結構性觀察）",
) -> None:
    """Side-by-side premium time series for multiple ETFs.

    Useful for a single slide comparing 00919 vs 00878 side by side.

    Parameters
    ----------
    premium_dict : dict[str, pd.DataFrame]
        Mapping of ETF code → DataFrame with ``premium_pct`` column.
    save_dir : str or Path
        Destination directory.
    title : str
        Overall figure title.
    """
    n = len(premium_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for ax, (etf_id, df), color in zip(axes, premium_dict.items(), palette):
        pct = df["premium_pct"].dropna()
        pos = pct.where(pct > 0)
        neg = pct.where(pct <= 0)
        ax.plot(df.index, pct, color=color, linewidth=0.8, alpha=0.7)
        ax.fill_between(df.index, 0, pos, alpha=0.25, color="#2ca02c")
        ax.fill_between(df.index, 0, neg, alpha=0.25, color="#d62728")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(pct.mean(), color=color, linewidth=1.3, linestyle="--",
                   alpha=0.8, label=f"均值 {pct.mean():+.3f}%")
        ax.set_title(
            f"{etf_id}\n均值={pct.mean():+.3f}%  std={pct.std():.3f}%",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("日期", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:+.2f}%"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel("折溢價率 (%)", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.text(
        0.5, -0.02,
        "純觀察性分析 — 非交易訊號 — 不構成投資建議",
        ha="center", fontsize=8, color="gray", style="italic",
    )
    plt.tight_layout()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "premium_etf_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ETF comparison chart saved -> %s", out)
