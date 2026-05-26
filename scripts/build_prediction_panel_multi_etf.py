"""
scripts/build_prediction_panel_multi_etf.py
-------------------------------------------
Build prediction_panel_multi_etf.parquet for all 38 events across
0056, 00713, and 00919 (events 2-7).

Design
------
- Universe : expanded static universe (0050 ∪ 0056 ∪ 00713 ∪ 00919 history) = 234 stocks
- T-14     : 14 trading days before each event's announcement_date
- Candidate pool per event:
    (a) had dividend (dividend_yield > 0) in past 12 months at T-14
    (b) not delisted by T-14
    (c) not already in the ETF whose event we are processing
- y=1: stock added in this event; y=0: others in pool

Features (all observable at T-14, no look-ahead)
-------------------------------------------------
  dividend_yield_ttm           most recent dividend_yield before T-14
  dividend_yield_rank_in_pool  rank within pool (1=highest)
  log_market_cap               log(price × shares) at T-14
  market_cap_rank_in_pool      rank within pool (1=largest)
  log_avg_turnover_60d         log(mean daily turnover, 60 td window)
  turnover_ratio_60d           avg daily volume / shares issued
  volatility_60d               annualised std of daily returns (60 td)
  beta_60d                     OLS beta vs TAIEX (60 td)
  momentum_60d_pre             cumulative return [T-90, T-30]
  was_in_etf_previous          1 if stock was in THIS ETF before previous event
  etf_code                     ETF identifier (categorical)

Extra flags (for downstream filtering, not used as features by default)
-----------------------------------------------------------------------
  has_concurrent_announcement  True if another ETF announced same day
  is_dual_treatment            True for stock 2606 on 2024-12-17 only
  is_structural_change         True for 0056_2022Dec
"""

import sys
import logging
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
EVENTS_CSV   = ROOT / "data" / "processed" / "multi_etf_events.csv"
UNIVERSE_CSV = ROOT / "data" / "raw"       / "static_universe.csv"
PRICES_PQ    = ROOT / "data" / "processed" / "multi_etf_stock_prices.parquet"
SHARES_PQ    = ROOT / "data" / "processed" / "shareholding_multi_etf.parquet"
PER_PQ       = ROOT / "data" / "processed" / "per_cache_multi_etf.parquet"
DELIST_PQ    = ROOT / "data" / "processed" / "delisting.parquet"
OUT_PANEL    = ROOT / "data" / "processed" / "prediction_panel_multi_etf.parquet"
OUT_DESC     = ROOT / "output" / "tables"  / "prediction_panel_multi_etf_describe.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _t_minus_n(ann_date, trading_days, n=14):
    pre = trading_days[trading_days < ann_date]
    return pre[-n] if len(pre) >= n else None


def _prior_dividend_yield(per_df, stock_id, date):
    """Most recent non-null dividend_yield strictly before date."""
    mask = (per_df["stock_id"] == stock_id) & (per_df["date"] < date)
    s = per_df.loc[mask, "dividend_yield"].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def _had_dividend_12mo(per_df, stock_id, t14):
    start = t14 - pd.Timedelta(days=365)
    mask = (per_df["stock_id"] == stock_id) & (per_df["date"] >= start) & (per_df["date"] < t14)
    s = per_df.loc[mask, "dividend_yield"].dropna()
    return bool((s > 0).any())


def _shares_at(shares_df, stock_id, date):
    s = shares_df[(shares_df["stock_id"] == stock_id) & (shares_df["date"] <= date)]
    if s.empty:
        return np.nan
    return float(s.sort_values("date").iloc[-1]["NumberOfSharesIssued"])


def _is_delisted(stock_id, date, delist_df):
    sub = delist_df[delist_df["stock_id"] == stock_id]
    return bool(not sub.empty and sub["date"].min() <= date)


def _annualized_vol(ret):
    return float(ret.std(ddof=1) * np.sqrt(252)) if len(ret) >= 5 else np.nan


def _beta(stock_ret, mkt_ret):
    df = pd.concat([stock_ret, mkt_ret], axis=1).dropna()
    if len(df) < 10:
        return np.nan
    cov = df.cov().iloc[0, 1]
    var = float(df.iloc[:, 1].var())
    return float(cov / var) if var > 0 else np.nan


def _build_etf_holdings_tracker(events_list, etf_code):
    """Cumulative holdings tracker for a single ETF.

    Returns dict: event_id -> set of stock_ids known in ETF *before* this event.
    """
    tracked = {}
    cur = set()
    for ev in events_list:
        tracked[ev["event_id"]] = set(cur)
        cur = (cur | set(s for s in ev["added_stocks"].split("|") if s)) \
            - set(s for s in ev["removed_stocks"].split("|") if s)
    return tracked


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Build prediction_panel_multi_etf")
    print("=" * 70)

    # ── Load events
    with open(EVENTS_CSV) as f:
        all_events = list(csv.DictReader(f))
    print(f"Events loaded: {len(all_events)}")

    # ── Build per-ETF holdings trackers
    etf_codes = ["0056", "00713", "00919"]
    trackers = {}
    for etf in etf_codes:
        ev_for_etf = [e for e in all_events if e["etf_code"] == etf]
        trackers[etf] = _build_etf_holdings_tracker(ev_for_etf, etf)

    # ── Expanded universe
    uni_df = pd.read_csv(UNIVERSE_CSV, dtype={"stock_id": str})
    event_stocks = set()
    for e in all_events:
        for col in ["added_stocks", "removed_stocks"]:
            for s in e[col].split("|"):
                if s.strip():
                    event_stocks.add(s.strip())
    universe_ids = sorted(set(uni_df["stock_id"]) | event_stocks)
    print(f"Expanded universe: {len(universe_ids)} stocks")

    # ── Price data
    prices_long = pd.read_parquet(PRICES_PQ)
    prices_long["date"] = pd.to_datetime(prices_long["date"])
    prices_wide   = prices_long.pivot(index="date", columns="stock_id", values="close").sort_index()
    turnover_wide = prices_long.pivot(index="date", columns="stock_id", values="turnover").sort_index()
    volume_wide   = prices_long.pivot(index="date", columns="stock_id", values="volume").sort_index()
    market_ret    = prices_wide["TAIEX"].pct_change()
    trading_days  = prices_wide.index
    print(f"Price data: {len(prices_long):,} rows, trading days {trading_days[0].date()} ~ {trading_days[-1].date()}")

    # ── Ancillary data
    per = pd.read_parquet(PER_PQ)
    per["date"] = pd.to_datetime(per["date"])
    per = per.sort_values(["stock_id", "date"]).reset_index(drop=True)

    shares = pd.read_parquet(SHARES_PQ)
    shares["date"] = pd.to_datetime(shares["date"])

    delist = pd.read_parquet(DELIST_PQ)
    delist["date"] = pd.to_datetime(delist["date"])

    # ── Build panel ────────────────────────────────────────────────────────
    records = []
    hit_stats = []

    # For was_in_etf_previous: need the previous event per ETF
    # Build per-ETF ordered event list
    etf_event_lists = {}
    for etf in etf_codes:
        etf_event_lists[etf] = [e for e in all_events if e["etf_code"] == etf]

    for ev_idx, ev in enumerate(all_events):
        event_id   = ev["event_id"]
        etf        = ev["etf_code"]
        ann        = pd.Timestamp(ev["announcement_date"])
        has_conc   = ev.get("has_concurrent_announcement", "False") == "True"
        is_struct  = ev.get("is_structural_change", "False") == "True"

        t14 = _t_minus_n(ann, trading_days, n=14)
        if t14 is None:
            print(f"  ⚠ {event_id}: insufficient history, skip")
            continue

        # holdings just BEFORE this event (for exclusion)
        current_holdings = trackers[etf].get(event_id, set())
        added_this = set(s for s in ev["added_stocks"].split("|") if s)

        # ── Candidate pool ─────────────────────────────────────────────
        pool = []
        excl = {"delisted": 0, "no_div": 0, "in_etf": 0}
        for sid in universe_ids:
            if _is_delisted(sid, t14, delist):
                excl["delisted"] += 1; continue
            if not _had_dividend_12mo(per, sid, t14):
                excl["no_div"] += 1; continue
            if sid in current_holdings:
                excl["in_etf"] += 1; continue
            pool.append(sid)

        n_pos = sum(1 for s in added_this if s in pool)
        hit_stats.append({
            "event_id": event_id, "etf": etf,
            "ann_date": ann.date(), "t14_date": t14.date(),
            "pool_size": len(pool),
            "y1_in_pool": n_pos, "y1_total": len(added_this),
            "hit_rate": n_pos / max(len(added_this), 1),
            **{f"excl_{k}": v for k, v in excl.items()},
        })
        print(f"\n[{ev_idx+1:2d}/38] {event_id:<24} ann={ann.date()} T-14={t14.date()}")
        print(f"  pool={len(pool):3d}  y1={n_pos}/{len(added_this)}  "
              f"excl: delisted={excl['delisted']} no_div={excl['no_div']} in_etf={excl['in_etf']}")

        # ── Pool-wide pre-computation ──────────────────────────────────
        pool_yield, pool_mcap = {}, {}
        for sid in pool:
            pool_yield[sid] = _prior_dividend_yield(per, sid, t14)
            price = np.nan
            if sid in prices_wide.columns:
                pre = prices_wide.index[prices_wide.index <= t14]
                if len(pre):
                    v = prices_wide.loc[pre[-1], sid]
                    price = float(v) if pd.notna(v) else np.nan
            sh = _shares_at(shares, sid, t14)
            pool_mcap[sid] = price * sh if (pd.notna(price) and pd.notna(sh)) else np.nan

        yield_s = pd.Series(pool_yield).dropna()
        mcap_s  = pd.Series(pool_mcap).dropna()
        yield_ranks = yield_s.rank(ascending=False, method="min").to_dict()
        mcap_ranks  = mcap_s.rank(ascending=False, method="min").to_dict()

        # For was_in_etf_previous: the set BEFORE the previous event of this ETF
        etf_ev_list = etf_event_lists[etf]
        this_pos_in_etf = next((i for i, e in enumerate(etf_ev_list) if e["event_id"] == event_id), None)
        if this_pos_in_etf is not None and this_pos_in_etf >= 1:
            prev_event_id = etf_ev_list[this_pos_in_etf - 1]["event_id"]
            prev_prev_holdings = trackers[etf].get(prev_event_id, set())
            prev_prev_valid = True
        else:
            prev_prev_holdings = set()
            prev_prev_valid = False  # first event of this ETF → NaN

        # ── Per-stock features ─────────────────────────────────────────
        for sid in pool:
            row = {
                "event_id": event_id, "etf_code": etf,
                "ann_date": ann, "t14_date": t14, "stock_id": sid,
                "y": 1 if sid in added_this else 0,
                "has_concurrent_announcement": has_conc,
                "is_dual_treatment": (sid == "2606" and str(ann.date()) == "2024-12-17"),
                "is_structural_change": is_struct,
            }

            # A. Dividend
            yld = pool_yield[sid]
            row["dividend_yield_ttm"] = yld
            row["dividend_yield_rank_in_pool"] = yield_ranks.get(sid, np.nan)

            # B. Size
            mcap = pool_mcap[sid]
            row["market_cap"] = mcap
            row["log_market_cap"] = float(np.log(mcap)) if (pd.notna(mcap) and mcap > 0) else np.nan
            row["market_cap_rank_in_pool"] = mcap_ranks.get(sid, np.nan)

            # C. Liquidity
            if sid in turnover_wide.columns:
                pre = turnover_wide.index[turnover_wide.index <= t14]
                w   = pre[-60:]
                avg_to = float(turnover_wide.loc[w, sid].mean())
                row["avg_turnover_60d"]     = avg_to if pd.notna(avg_to) else np.nan
                row["log_avg_turnover_60d"] = float(np.log(avg_to)) if (pd.notna(avg_to) and avg_to > 0) else np.nan
            else:
                row["avg_turnover_60d"] = row["log_avg_turnover_60d"] = np.nan

            sh = _shares_at(shares, sid, t14)
            if sid in volume_wide.columns and pd.notna(sh) and sh > 0:
                pre = volume_wide.index[volume_wide.index <= t14]
                avg_vol = float(volume_wide.loc[pre[-60:], sid].mean())
                row["turnover_ratio_60d"] = avg_vol / sh if pd.notna(avg_vol) else np.nan
            else:
                row["turnover_ratio_60d"] = np.nan

            # D. Volatility & Beta
            if sid in prices_wide.columns:
                pre = prices_wide.index[prices_wide.index <= t14]
                w   = pre[-60:]
                p   = prices_wide.loc[w, sid].dropna()
                if len(p) >= 2:
                    ret = p.pct_change().dropna()
                    row["volatility_60d"] = _annualized_vol(ret)
                    row["beta_60d"]       = _beta(ret, market_ret.loc[w].dropna())
                else:
                    row["volatility_60d"] = row["beta_60d"] = np.nan
            else:
                row["volatility_60d"] = row["beta_60d"] = np.nan

            # E. was_in_etf_previous (per-ETF tracking)
            if prev_prev_valid:
                row["was_in_etf_previous"] = int(sid in prev_prev_holdings)
            else:
                row["was_in_etf_previous"] = np.nan

            # F. Momentum [T-90, T-30]
            if sid in prices_wide.columns:
                pre = prices_wide.index[prices_wide.index <= t14]
                if len(pre) >= 90:
                    w  = pre[-90:-30]
                    p  = prices_wide.loc[w, sid].dropna()
                    if len(p) >= 2:
                        row["momentum_60d_pre"] = float((1 + p.pct_change().dropna()).prod() - 1)
                    else:
                        row["momentum_60d_pre"] = np.nan
                else:
                    row["momentum_60d_pre"] = np.nan
            else:
                row["momentum_60d_pre"] = np.nan

            records.append(row)

    panel = pd.DataFrame(records)
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PANEL, index=False)
    print(f"\n✓ Saved → {OUT_PANEL.name}  ({len(panel):,} rows)")

    # ── Hit-rate table ─────────────────────────────────────────────────────
    hits = pd.DataFrame(hit_stats)
    print("\n" + "=" * 90)
    print("  Per-event candidate pool")
    print("=" * 90)
    print(hits.to_string(index=False))

    # ── Statistics ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Class balance by ETF")
    print("=" * 70)
    for etf in etf_codes:
        sub = panel[panel["etf_code"] == etf]
        if sub.empty:
            continue
        pos = int((sub["y"] == 1).sum())
        neg = int((sub["y"] == 0).sum())
        nevt = sub["event_id"].nunique()
        pool_range = f"{hits[hits['etf']==etf]['pool_size'].min()}~{hits[hits['etf']==etf]['pool_size'].max()}"
        print(f"  {etf}: {nevt} events | pos={pos} neg={neg} | pos_rate={pos/(pos+neg):.2%} | pool {pool_range}")

    total_pos = int((panel["y"] == 1).sum())
    total_neg = int((panel["y"] == 0).sum())
    print(f"\n  Total: pos={total_pos} neg={total_neg} pos_rate={total_pos/(total_pos+total_neg):.2%}")

    # ── Feature missing rates ───────────────────────────────────────────────
    feat_cols = [
        "dividend_yield_ttm", "dividend_yield_rank_in_pool",
        "log_market_cap", "market_cap_rank_in_pool",
        "log_avg_turnover_60d", "turnover_ratio_60d",
        "volatility_60d", "beta_60d",
        "was_in_etf_previous", "momentum_60d_pre",
    ]
    print("\n" + "=" * 70)
    print("  Feature missing rates")
    print("=" * 70)
    for col in feat_cols:
        if col not in panel.columns:
            continue
        miss = panel[col].isna().sum()
        print(f"  {col:<35s}: {miss:5d} / {len(panel):5d} = {miss/len(panel):.1%}")

    # ── Pos vs Neg mean comparison ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  pos (y=1) vs neg (y=0) — mean comparison")
    print("=" * 70)
    cmp = panel.groupby("y")[feat_cols].mean().T
    cmp.columns = ["neg (y=0)", "pos (y=1)"]
    cmp["diff"] = cmp["pos (y=1)"] - cmp["neg (y=0)"]
    print(cmp.to_string(float_format="{:.4f}".format))

    # ── Describe CSV ────────────────────────────────────────────────────────
    OUT_DESC.parent.mkdir(parents=True, exist_ok=True)
    desc = panel[feat_cols].describe().T
    desc["missing"] = len(panel) - panel[feat_cols].count()
    desc["missing_rate"] = desc["missing"] / len(panel)
    desc.to_csv(OUT_DESC, float_format="%.4f")
    print(f"\n✓ Describe → {OUT_DESC.name}")

    # ── Train/Test split preview ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Train/Test split (2017-2023 train | 2024-2025 test)")
    print("=" * 70)
    panel["year"] = panel["ann_date"].dt.year
    train = panel[panel["ann_date"].dt.year <= 2023]
    test  = panel[panel["ann_date"].dt.year >= 2024]
    for label, sub in [("Train (≤2023)", train), ("Test  (≥2024)", test)]:
        pos = int((sub["y"] == 1).sum())
        neg = int((sub["y"] == 0).sum())
        nevt = sub["event_id"].nunique()
        print(f"  {label}: {nevt:2d} events | {len(sub):5d} rows | pos={pos} neg={neg} rate={pos/(pos+neg):.2%}")


if __name__ == "__main__":
    main()
