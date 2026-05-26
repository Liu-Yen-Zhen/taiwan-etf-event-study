"""
scripts/build_prediction_panel.py
---------------------------------
Step A3：構造階段 4 預測模型用的 panel（不做建模，只做資料準備）。

設計
----
- 跳過 event 1（00919_20221216），用 events 2-7（避免「event 1 前 00919 初始成分股」
  的不可知假設）
- T-14：每事件公告日前 14 個交易日
- Universe：data/raw/static_universe.csv（137 檔）
- 候選池篩選：
    (a) T-14 之前 12 個月有配息（TaiwanStockPER.dividend_yield > 0）
    (b) T-14 仍在市場（delisting 之後排除）
    (c) 排除「已在 00919 持股」（events.csv 累積追蹤，下界估計）

特徵（全部 T-14 可觀察）
------------------------
A. dividend_yield_ttm, dividend_yield_rank_in_pool
B. log_market_cap, market_cap_rank_in_pool
C. log_avg_turnover_60d, turnover_ratio_60d
D. volatility_60d (annualized), beta_60d
E. was_in_00919_previous (0/1)
F. momentum_60d_pre  (累積報酬 over [T-90, T-30])

產出
----
  data/processed/prediction_panel.parquet
  output/tables/prediction_panel_describe.csv
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from data_fetcher import load_events

logging.basicConfig(level=logging.WARNING)

# ── Paths ────────────────────────────────────────────────────────────────
EVENTS_CSV   = ROOT / "data" / "raw"       / "events.csv"
UNIVERSE_CSV = ROOT / "data" / "raw"       / "static_universe.csv"
PRICES_PQ    = ROOT / "data" / "processed" / "stock_prices.parquet"
SHARES_PQ    = ROOT / "data" / "processed" / "shareholding_v2.parquet"
PER_PQ       = ROOT / "data" / "processed" / "stock_per_cache.parquet"
DELIST_PQ    = ROOT / "data" / "processed" / "delisting.parquet"
OUT_PANEL    = ROOT / "data" / "processed" / "prediction_panel.parquet"
OUT_DESC     = ROOT / "output" / "tables"  / "prediction_panel_describe.csv"


# ── Helpers ──────────────────────────────────────────────────────────────


def _t_minus_n_trading_day(ann_date, trading_days, n=14):
    """Return the trading day that is exactly n trading days before ann_date."""
    pre = trading_days[trading_days < ann_date]
    if len(pre) < n:
        return None
    return pre[-n]


def _prior_value(per_df, stock_id, date, col):
    """Most recent non-null value of `col` strictly before `date`."""
    mask = (per_df["stock_id"] == stock_id) & (per_df.index < date)
    s = per_df.loc[mask, col].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def _at_or_before(shares_df, stock_id, date):
    """Most recent NumberOfSharesIssued at or before `date`."""
    s = shares_df[(shares_df["stock_id"] == stock_id) & (shares_df["date"] <= date)]
    if s.empty:
        return np.nan
    return float(s.sort_values("date").iloc[-1]["NumberOfSharesIssued"])


def _is_delisted_by(stock_id, date, delist_df):
    sub = delist_df[delist_df["stock_id"] == stock_id]
    if sub.empty:
        return False
    return (sub["date"].min() <= date)


def _had_dividend_past_12mo(per_df, stock_id, t14_date):
    """True if any dividend_yield > 0 in the past 365 calendar days before t14."""
    start = t14_date - pd.Timedelta(days=365)
    mask = ((per_df["stock_id"] == stock_id) &
            (per_df.index >= start) & (per_df.index < t14_date))
    s = per_df.loc[mask, "dividend_yield"].dropna()
    return bool((s > 0).any())


def _build_holdings_tracker(events):
    """Best-effort cumulative tracking of 00919 holdings starting from event 1.

    Returns a dict: event_id (str) → set of stock_ids known to be in 00919 just
    before that event. For event 1 this is empty (initial 30 unknown).
    """
    tracked = {}
    cur = set()
    for _, ev in events.iterrows():
        tracked[ev["event_id"]] = set(cur)   # snapshot BEFORE this event applies
        cur = (cur | set(ev["added_stocks"])) - set(ev["removed_stocks"])
    return tracked


def _annualized_vol(ret):
    if len(ret) < 5:
        return np.nan
    return float(ret.std(ddof=1) * np.sqrt(252))


def _beta_vs_market(stock_ret, market_ret):
    df = pd.concat([stock_ret, market_ret], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    cov = df.cov().iloc[0, 1]
    var = df.iloc[:, 1].var()
    if var == 0 or pd.isna(var):
        return np.nan
    return float(cov / var)


# ── Build panel ──────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  Step A3 — Build prediction panel")
    print("=" * 70)

    events    = load_events(EVENTS_CSV)
    universe  = pd.read_csv(UNIVERSE_CSV, dtype={"stock_id": str})
    universe_ids = set(universe["stock_id"])

    prices_long = pd.read_parquet(PRICES_PQ)
    prices_wide = prices_long.pivot(index="date", columns="stock_id",
                                     values="close").sort_index()
    turnover_wide = prices_long.pivot(index="date", columns="stock_id",
                                       values="turnover").sort_index()
    volume_wide = prices_long.pivot(index="date", columns="stock_id",
                                     values="volume").sort_index()
    market_returns = prices_wide["TAIEX"].pct_change()

    shares = pd.read_parquet(SHARES_PQ)
    shares["date"] = pd.to_datetime(shares["date"])
    per = pd.read_parquet(PER_PQ)
    if per.index.name != "date":
        # ensure DatetimeIndex named "date"
        if "date" in per.columns:
            per["date"] = pd.to_datetime(per["date"])
            per = per.set_index("date")
    delist = pd.read_parquet(DELIST_PQ)
    delist["date"] = pd.to_datetime(delist["date"])

    print(f"\nUniverse: {len(universe_ids)} stocks")
    print(f"Events  : {len(events)} (will use events 2-{len(events)})")

    # 00919 holdings tracker (skip event 1)
    holdings = _build_holdings_tracker(events)

    trading_days = prices_wide.index
    records = []
    hit_stats = []   # candidate-pool hit-rate per event

    for ev_idx, (_, ev) in enumerate(events.iterrows(), start=1):
        if ev_idx == 1:
            print(f"\n[Event {ev_idx}/7] {ev['event_id']} — SKIPPED (event 1, unknown initial holdings)")
            continue

        ann = pd.Timestamp(ev["announcement_date"])
        t14 = _t_minus_n_trading_day(ann, trading_days, n=14)
        if t14 is None:
            print(f"  ⚠ {ev['event_id']}: insufficient pre-history, skipping")
            continue

        current_holdings = holdings[ev["event_id"]]
        added_in_this_event = set(ev["added_stocks"])

        # ── Candidate pool filtering ────────────────────────────────────
        pool = []
        excluded_reasons = {"delisted": 0, "no_dividend": 0, "in_00919": 0}
        for sid in sorted(universe_ids):
            if _is_delisted_by(sid, t14, delist):
                excluded_reasons["delisted"] += 1
                continue
            if not _had_dividend_past_12mo(per, sid, t14):
                excluded_reasons["no_dividend"] += 1
                continue
            if sid in current_holdings:
                excluded_reasons["in_00919"] += 1
                continue
            pool.append(sid)

        n_added_in_pool = sum(1 for sid in added_in_this_event if sid in pool)
        n_added_total   = len(added_in_this_event)
        hit_stats.append({
            "event_id": ev["event_id"],
            "ann_date": ann.date(),
            "t14_date": t14.date(),
            "pool_size": len(pool),
            "y1_in_pool": n_added_in_pool,
            "y1_total": n_added_total,
            "hit_rate": n_added_in_pool / max(n_added_total, 1),
            "excluded_delisted": excluded_reasons["delisted"],
            "excluded_no_dividend": excluded_reasons["no_dividend"],
            "excluded_in_00919": excluded_reasons["in_00919"],
        })
        print(f"\n[Event {ev_idx}/7] {ev['event_id']} | ann={ann.date()} | T-14={t14.date()}")
        print(f"  Pool size: {len(pool)} (excluded: delisted={excluded_reasons['delisted']}, "
              f"no_div={excluded_reasons['no_dividend']}, in_00919={excluded_reasons['in_00919']})")
        print(f"  y=1 in pool: {n_added_in_pool}/{n_added_total} = {n_added_in_pool/max(n_added_total,1):.0%}")

        # ── Pre-compute pool-wide rankings before per-stock loop ────────
        # First pass: yields and market caps
        pool_yield = {}
        pool_mcap  = {}
        for sid in pool:
            yld = _prior_value(per, sid, t14, "dividend_yield")
            pool_yield[sid] = yld
            # Market cap at T-14
            if sid in prices_wide.columns:
                pre = prices_wide.index[prices_wide.index <= t14]
                price = float(prices_wide.loc[pre[-1], sid]) if len(pre) and pd.notna(prices_wide.loc[pre[-1], sid]) else np.nan
            else:
                price = np.nan
            sh = _at_or_before(shares, sid, t14)
            mcap = price * sh if (pd.notna(price) and pd.notna(sh)) else np.nan
            pool_mcap[sid] = mcap

        # Convert to rank (1 = highest)
        yield_series = pd.Series(pool_yield).dropna()
        mcap_series  = pd.Series(pool_mcap).dropna()
        yield_ranks  = yield_series.rank(ascending=False, method="min").to_dict()
        mcap_ranks   = mcap_series.rank(ascending=False, method="min").to_dict()

        # ── Per-stock features ────────────────────────────────────────
        prev_event_holdings = holdings[ev["event_id"]]   # = current_holdings (set just before this event)
        # was_in_00919_previous means: was the stock in 00919 just before THIS event?
        # But we excluded those from pool. So for pool stocks, this is always 0.
        # Reinterpret: was_in_00919_previous = was in 00919 BEFORE the PREVIOUS event (event ev_idx-2 indexing).
        # Practically: the cumulative set BEFORE the previous event.
        prev_prev_holdings = (holdings[events.iloc[ev_idx - 2]["event_id"]]
                              if ev_idx >= 2 else set())

        for sid in pool:
            row = {
                "event_id": ev["event_id"],
                "ann_date": ann,
                "t14_date": t14,
                "stock_id": sid,
                "y": 1 if sid in added_in_this_event else 0,
            }

            # ── A. Dividend yield ──────────────────────────────────────
            yld = pool_yield[sid]
            row["dividend_yield_ttm"] = yld
            row["dividend_yield_rank_in_pool"] = yield_ranks.get(sid, np.nan)

            # ── B. Size ────────────────────────────────────────────────
            mcap = pool_mcap[sid]
            row["market_cap"] = mcap
            row["log_market_cap"] = float(np.log(mcap)) if (pd.notna(mcap) and mcap > 0) else np.nan
            row["market_cap_rank_in_pool"] = mcap_ranks.get(sid, np.nan)

            # ── C. Liquidity ──────────────────────────────────────────
            if sid in turnover_wide.columns:
                pre = turnover_wide.index[turnover_wide.index <= t14]
                window = pre[-60:]
                avg_to = float(turnover_wide.loc[window, sid].mean())
                row["avg_turnover_60d"] = avg_to if pd.notna(avg_to) else np.nan
                row["log_avg_turnover_60d"] = (float(np.log(avg_to))
                                               if pd.notna(avg_to) and avg_to > 0 else np.nan)
            else:
                row["avg_turnover_60d"] = np.nan
                row["log_avg_turnover_60d"] = np.nan

            sh_at = _at_or_before(shares, sid, t14)
            if sid in volume_wide.columns and pd.notna(sh_at) and sh_at > 0:
                pre = volume_wide.index[volume_wide.index <= t14]
                window = pre[-60:]
                avg_vol = float(volume_wide.loc[window, sid].mean())
                row["turnover_ratio_60d"] = (avg_vol / sh_at) if pd.notna(avg_vol) and avg_vol > 0 else np.nan
            else:
                row["turnover_ratio_60d"] = np.nan

            # ── D. Volatility & Beta ───────────────────────────────────
            if sid in prices_wide.columns:
                pre = prices_wide.index[prices_wide.index <= t14]
                window = pre[-60:]
                p = prices_wide.loc[window, sid].dropna()
                ret = p.pct_change().dropna()
                row["volatility_60d"] = _annualized_vol(ret) if len(ret) >= 5 else np.nan
                mkt = market_returns.loc[window].dropna()
                row["beta_60d"] = _beta_vs_market(ret, mkt) if len(ret) >= 10 else np.nan
            else:
                row["volatility_60d"] = np.nan
                row["beta_60d"] = np.nan

            # ── E. History ────────────────────────────────────────────
            # was_in_00919_previous: was the stock in 00919 just before the previous event?
            row["was_in_00919_previous"] = int(sid in prev_prev_holdings) if ev_idx >= 3 else np.nan

            # ── F. Momentum [T-90, T-30] ──────────────────────────────
            if sid in prices_wide.columns:
                pre = prices_wide.index[prices_wide.index <= t14]
                if len(pre) >= 90:
                    window = pre[-90:-30]   # 60 days, ending T-30
                    p = prices_wide.loc[window, sid].dropna()
                    if len(p) >= 2:
                        rets = p.pct_change().dropna()
                        row["momentum_60d_pre"] = float((1 + rets).prod() - 1)
                    else:
                        row["momentum_60d_pre"] = np.nan
                else:
                    row["momentum_60d_pre"] = np.nan
            else:
                row["momentum_60d_pre"] = np.nan

            records.append(row)

    panel = pd.DataFrame(records)
    panel.to_parquet(OUT_PANEL, index=False)
    print(f"\n✓ Panel saved → {OUT_PANEL.relative_to(ROOT)}  ({len(panel)} rows)")

    # ── Hit-rate summary ─────────────────────────────────────────────────
    hits = pd.DataFrame(hit_stats)
    print("\n" + "=" * 80)
    print("  Per-event candidate pool")
    print("=" * 80)
    print(hits.to_string(index=False))

    overall_added_in_pool = hits["y1_in_pool"].sum()
    overall_added_total   = hits["y1_total"].sum()
    overall_pool_size     = hits["pool_size"].sum()
    print(f"\nOverall hit rate: {overall_added_in_pool}/{overall_added_total} "
          f"= {overall_added_in_pool/overall_added_total:.1%}")
    print(f"Overall pool rows: {overall_pool_size}")

    # ── Class balance ────────────────────────────────────────────────────
    pos = int((panel["y"] == 1).sum())
    neg = int((panel["y"] == 0).sum())
    print(f"\nClass balance: pos={pos}, neg={neg}, pos_rate={pos/(pos+neg):.2%}")

    # ── Describe ─────────────────────────────────────────────────────────
    feat_cols = ["dividend_yield_ttm", "dividend_yield_rank_in_pool",
                 "market_cap", "log_market_cap", "market_cap_rank_in_pool",
                 "avg_turnover_60d", "log_avg_turnover_60d", "turnover_ratio_60d",
                 "volatility_60d", "beta_60d",
                 "was_in_00919_previous", "momentum_60d_pre"]
    desc = panel[feat_cols].describe().T
    desc["missing"] = len(panel) - panel[feat_cols].count()
    desc["missing_rate"] = desc["missing"] / len(panel)
    desc = desc[["count", "missing", "missing_rate", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc.to_csv(OUT_DESC, float_format="%.4f")
    print(f"\n✓ Describe → {OUT_DESC.relative_to(ROOT)}")
    print(desc[["count", "missing", "missing_rate", "mean", "std", "min", "max"]].to_string())

    # ── Pos vs neg quick compare ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  pos (y=1) vs neg (y=0) — mean comparison")
    print("=" * 80)
    cmp = panel.groupby("y")[feat_cols].mean().T
    cmp.columns = ["neg (y=0)", "pos (y=1)"]
    cmp["diff_pos_minus_neg"] = cmp["pos (y=1)"] - cmp["neg (y=0)"]
    print(cmp.to_string())


if __name__ == "__main__":
    main()
