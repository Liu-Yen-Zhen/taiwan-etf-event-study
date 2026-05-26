"""
src/prediction.py
-----------------
Rule-based prediction of ETF constituent additions for 00919.

00919 tracks **臺灣精選高息指數 (IX0170)**.
Official rules (per taiwanindex.com.tw, last revised 2026-03-25):

  Universe
  ────────
  • TWSE + TPEx listed stocks, top ~300 by market cap
  • 60-day average daily 成交金額 (turnover) above threshold [~8e7 TWD/day]
  • At least ~120 trading days of history (proxy for listing-age requirement)
  • ROE (trailing 4 quarters) > 0  ← SKIPPED: no quarterly earnings data
    available in our dataset; acknowledged as an approximation gap

  Ranking / Selection
  ───────────────────
  • May review   → ranked by **declared dividend yield** (宣告股利率)
  • December review → ranked by **estimated yield** = trailing_12m_div / price
                      × (1 + YTD EPS growth)  ← APPROXIMATED as trailing yield
                      (quarterly EPS not in scope); noted explicitly in output
  • Top 40 constituents selected

  Buffer mechanism (minimise turnover)
  ─────────────────────────────────────
  • Rank ≤ 15  → auto-include regardless of incumbent status
  • Rank > 46  → auto-exclude regardless of incumbent status
  • Rank 16-46 → incumbent stocks stay in; new entrants excluded

  Weighting (not modelled; relevant only to portfolio construction)
  ──────────────────────────────────────────────────────────────────
  • Dividend-yield weighted, adjusted by liquidity
  • Individual cap: 15 %;  floor: 0.5 %

Look-ahead policy
─────────────────
Every function in this module accepts a ``ref_date`` (= prediction date).
All data accesses filter with ``< ref_date`` or ``<= prev_trading_day``.
No information from the announcement date or later is used.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Rule constants ─────────────────────────────────────────────────────────────

MARKET_CAP_UNIVERSE_SIZE = 300   # top-N by mkt cap to form the initial pool
DEFAULT_TOP_K = 40               # constituents per rebalancing
DEFAULT_MIN_AVG_TURNOVER = 8e7   # TWD/day (rough threshold from public rules)
DEFAULT_MIN_TRADING_DAYS = 120   # proxy for listing-age requirement (~6 months)
BUFFER_AUTO_IN_RANK = 15         # auto-include if rank ≤ this
BUFFER_AUTO_OUT_RANK = 46        # auto-exclude if rank > this


# ── Private helpers ────────────────────────────────────────────────────────────


def _prev_trading_days(ref_date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """All trading days strictly before ref_date."""
    return trading_days[trading_days < ref_date]


def _last_value_before(
    long_df: pd.DataFrame,
    stock_id: str,
    ref_date: pd.Timestamp,
    col: str,
) -> float:
    """Most recent non-null value of ``col`` strictly before ref_date."""
    mask = (long_df.index < ref_date) & (long_df["stock_id"] == stock_id)
    s = long_df.loc[mask, col].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def _compute_market_cap(
    stock_id: str,
    ref_date: pd.Timestamp,
    prices_wide: pd.DataFrame,
    shares_df: Optional[pd.DataFrame],
) -> float:
    """Market cap at previous trading day.  Falls back to price if shares unavailable."""
    pre = _prev_trading_days(ref_date, prices_wide.index)
    if pre.empty or stock_id not in prices_wide.columns:
        return np.nan
    price = float(prices_wide.loc[pre[-1], stock_id])
    if pd.isna(price) or price <= 0:
        return np.nan
    if shares_df is not None:
        shares = _last_value_before(shares_df, stock_id, ref_date, "shares_issued")
        if not pd.isna(shares) and shares > 0:
            return price * shares
    # Fallback: use price alone (ordinal proxy for size; biased but usable)
    return price


def _compute_avg_turnover(
    stock_id: str,
    ref_date: pd.Timestamp,
    turnover_wide: pd.DataFrame,
    n_days: int = 60,
) -> float:
    """Mean daily 成交金額 over last ``n_days`` trading days before ref_date."""
    pre = _prev_trading_days(ref_date, turnover_wide.index)[-n_days:]
    if pre.empty or stock_id not in turnover_wide.columns:
        return np.nan
    return float(turnover_wide.loc[pre, stock_id].mean())


def _count_trading_days_available(
    stock_id: str,
    ref_date: pd.Timestamp,
    prices_wide: pd.DataFrame,
) -> int:
    """Number of non-NaN price observations strictly before ref_date."""
    pre = _prev_trading_days(ref_date, prices_wide.index)
    if pre.empty or stock_id not in prices_wide.columns:
        return 0
    return int(prices_wide.loc[pre, stock_id].notna().sum())


def _apply_buffer(
    ranked_df: pd.DataFrame,
    current_constituents: Optional[set],
    top_k: int,
) -> list[str]:
    """Apply the incumbent-buffer mechanism and return the predicted constituent list.

    Rules (mirroring the 00919 index methodology):
      • rank ≤ BUFFER_AUTO_IN_RANK  → auto-include
      • rank > BUFFER_AUTO_OUT_RANK → auto-exclude
      • rank in (AUTO_IN, AUTO_OUT] → incumbent stays; newcomer excluded

    Parameters
    ----------
    ranked_df : pd.DataFrame
        Must have columns [stock_id, rank] (1-based), sorted ascending by rank.
    current_constituents : set or None
        Current ETF constituents.  If None, no buffer is applied.
    top_k : int
        Maximum number of constituents to select.

    Returns
    -------
    list[str]  — predicted constituent stock_ids
    """
    constituents = set(current_constituents) if current_constituents else set()
    selected = []

    for _, row in ranked_df.iterrows():
        sid = row["stock_id"]
        rnk = row["rank"]

        if rnk <= BUFFER_AUTO_IN_RANK:
            selected.append(sid)
        elif rnk > BUFFER_AUTO_OUT_RANK:
            break  # ranked_df is sorted; remaining are worse
        else:
            # Buffer zone: only keep if currently a constituent
            if sid in constituents:
                selected.append(sid)

        if len(selected) >= top_k:
            break

    return selected


# ── Public API ─────────────────────────────────────────────────────────────────


def build_eligible_universe(
    ref_date: pd.Timestamp | str,
    prices_wide: pd.DataFrame,
    turnover_wide: Optional[pd.DataFrame] = None,
    shares_df: Optional[pd.DataFrame] = None,
    market_cap_top_n: int = MARKET_CAP_UNIVERSE_SIZE,
    min_avg_turnover: float = DEFAULT_MIN_AVG_TURNOVER,
    min_trading_days: int = DEFAULT_MIN_TRADING_DAYS,
) -> list[str]:
    """Screen the investable universe at ref_date.

    Filters applied (in order):
    1. Minimum trading-day history  (proxy for listing-age requirement)
    2. Top ``market_cap_top_n`` stocks by estimated market cap
    3. 60-day average daily turnover ≥ ``min_avg_turnover``  (if turnover_wide given)

    ROE filter is **omitted** — quarterly earnings data is not in scope.
    This is acknowledged as an approximation gap that will cause our universe to be
    slightly broader than the official rule.

    Parameters
    ----------
    ref_date : Timestamp or str
        Prediction date.  Only data strictly before this date is used.
    prices_wide : pd.DataFrame
        Wide close-price DataFrame.
    turnover_wide : pd.DataFrame, optional
        Wide daily turnover (成交金額) DataFrame.
    shares_df : pd.DataFrame, optional
        Long-format shares-outstanding DataFrame.
    market_cap_top_n : int
        Size of market-cap universe (default 300).
    min_avg_turnover : float
        60-day avg turnover floor in TWD (default ~8e7).
    min_trading_days : int
        Minimum history in trading days (default 120 ≈ 6 months).

    Returns
    -------
    list[str]  — eligible stock_ids
    """
    ref_date = pd.Timestamp(ref_date)
    all_stocks = [c for c in prices_wide.columns if c != "TAIEX"]

    # ── Step 1: minimum history ──
    qualified = []
    for sid in all_stocks:
        if _count_trading_days_available(sid, ref_date, prices_wide) >= min_trading_days:
            qualified.append(sid)

    if not qualified:
        return []

    # ── Step 2: top N by market cap ──
    mc = {sid: _compute_market_cap(sid, ref_date, prices_wide, shares_df) for sid in qualified}
    mc_sorted = sorted(qualified, key=lambda s: mc.get(s, 0) or 0, reverse=True)
    universe = mc_sorted[:market_cap_top_n]

    # ── Step 3: liquidity filter ──
    if turnover_wide is not None:
        eligible = []
        for sid in universe:
            avg_vol = _compute_avg_turnover(sid, ref_date, turnover_wide)
            if pd.isna(avg_vol) or avg_vol >= min_avg_turnover:
                # NaN avg_vol → data missing, keep optimistically
                eligible.append(sid)
        return eligible

    return universe


def rank_by_index_rule(
    stock_ids: list[str],
    ref_date: pd.Timestamp | str,
    per_df: Optional[pd.DataFrame],
    review_period: str = "auto",
) -> pd.DataFrame:
    """Rank eligible stocks by 00919 index criteria.

    For **May reviews**: ranked by declared dividend yield (宣告股利率).
    For **December reviews**: estimated yield = trailing_12m_div / price ×
        (1 + YTD EPS growth).  Since EPS data is not in scope, we use
        trailing dividend yield as an **approximation** and flag this explicitly.

    Parameters
    ----------
    stock_ids : list[str]
        Eligible universe from build_eligible_universe().
    ref_date : Timestamp or str
        Prediction date.
    per_df : pd.DataFrame, optional
        Long-format PER data with columns [stock_id, dividend_yield, ...].
        DatetimeIndex.  If None, ranking falls back to market-cap order
        (poor proxy — logged as a warning).
    review_period : str
        "may", "december", or "auto" (infer from ref_date month).

    Returns
    -------
    pd.DataFrame
        Columns: stock_id, score (dividend_yield in %), rank (1-based).
        Sorted ascending by rank.
    """
    ref_date = pd.Timestamp(ref_date)

    period = review_period
    if period == "auto":
        period = "december" if ref_date.month >= 10 else "may"

    if period == "december":
        logger.info(
            "December review: estimated yield approximated as trailing yield "
            "(YTD EPS growth data not available)."
        )

    if per_df is None:
        logger.warning(
            "per_df not provided — falling back to arbitrary ordering. "
            "Ranking quality will be poor."
        )
        records = [{"stock_id": s, "score": np.nan, "rank": i + 1}
                   for i, s in enumerate(stock_ids)]
        return pd.DataFrame(records)

    scores = []
    for sid in stock_ids:
        yield_val = _last_value_before(per_df, sid, ref_date, "dividend_yield")
        scores.append({"stock_id": sid, "score": yield_val})

    ranked = (
        pd.DataFrame(scores)
        .sort_values("score", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def predict_full_constituents(
    ref_date: pd.Timestamp | str,
    prices_wide: pd.DataFrame,
    per_df: Optional[pd.DataFrame] = None,
    turnover_wide: Optional[pd.DataFrame] = None,
    shares_df: Optional[pd.DataFrame] = None,
    current_constituents: Optional[set] = None,
    top_k: int = DEFAULT_TOP_K,
    market_cap_top_n: int = MARKET_CAP_UNIVERSE_SIZE,
    min_avg_turnover: float = DEFAULT_MIN_AVG_TURNOVER,
    min_trading_days: int = DEFAULT_MIN_TRADING_DAYS,
) -> list[str]:
    """Predict the full next-period constituent list.

    Combines build_eligible_universe → rank_by_index_rule → _apply_buffer.
    All data is filtered to strictly before ref_date.

    Returns
    -------
    list[str]  — predicted full constituent list (up to top_k stocks)
    """
    ref_date = pd.Timestamp(ref_date)

    eligible = build_eligible_universe(
        ref_date=ref_date,
        prices_wide=prices_wide,
        turnover_wide=turnover_wide,
        shares_df=shares_df,
        market_cap_top_n=market_cap_top_n,
        min_avg_turnover=min_avg_turnover,
        min_trading_days=min_trading_days,
    )
    if not eligible:
        logger.warning("Empty eligible universe at %s", ref_date.date())
        return []

    ranked = rank_by_index_rule(eligible, ref_date, per_df)
    predicted = _apply_buffer(ranked, current_constituents, top_k)
    return predicted


def predict_additions(
    ref_date: pd.Timestamp | str,
    prices_wide: pd.DataFrame,
    per_df: Optional[pd.DataFrame] = None,
    turnover_wide: Optional[pd.DataFrame] = None,
    shares_df: Optional[pd.DataFrame] = None,
    current_constituents: Optional[set] = None,
    top_k: int = DEFAULT_TOP_K,
    **kwargs,
) -> list[str]:
    """Predict which stocks will be *newly added* in the next rebalancing.

    Returns the predicted constituent list minus current constituents,
    i.e. stocks we predict will enter the index.

    Parameters
    ----------
    ref_date : Timestamp or str
        Prediction date (should be announcement_date − offset trading days).
    prices_wide, per_df, turnover_wide, shares_df : see predict_full_constituents.
    current_constituents : set or None
        Stocks currently in the ETF.  Used both for the buffer mechanism and
        to derive the predicted additions.
    top_k : int
        Number of constituents in the index (default 40).

    Returns
    -------
    list[str]  — predicted newly-added stocks (not in current_constituents)
    """
    ref_date = pd.Timestamp(ref_date)
    curr = set(current_constituents) if current_constituents else set()

    predicted_full = predict_full_constituents(
        ref_date=ref_date,
        prices_wide=prices_wide,
        per_df=per_df,
        turnover_wide=turnover_wide,
        shares_df=shares_df,
        current_constituents=curr if curr else None,
        top_k=top_k,
        **kwargs,
    )
    return [s for s in predicted_full if s not in curr]


def evaluate_prediction(
    predicted_set: set | list,
    actual_added_set: set | list,
) -> dict:
    """Compute precision, recall, and F1 for a single-event prediction.

    precision = |predicted ∩ actual| / |predicted|   (how clean are our picks?)
    recall    = |predicted ∩ actual| / |actual|       (how complete are they?)
    F1        = harmonic mean of precision and recall

    Returns
    -------
    dict with keys: precision, recall, f1, n_predicted, n_actual, n_correct.
    """
    pred = set(predicted_set)
    actual = set(actual_added_set)
    correct = pred & actual

    n_pred = len(pred)
    n_actual = len(actual)
    n_correct = len(correct)

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_actual if n_actual else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predicted": n_pred,
        "n_actual": n_actual,
        "n_correct": n_correct,
        "correct": sorted(correct),
        "false_positives": sorted(pred - actual),
        "false_negatives": sorted(actual - pred),
    }


def backtest_predictions(
    events: pd.DataFrame,
    prices_wide: pd.DataFrame,
    per_df: Optional[pd.DataFrame] = None,
    turnover_wide: Optional[pd.DataFrame] = None,
    shares_df: Optional[pd.DataFrame] = None,
    prediction_date_offset: int = -14,
    top_k: int = DEFAULT_TOP_K,
    initial_constituents: Optional[set] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run the rule-based prediction model across all events.

    For each event (in chronological order), a prediction is made
    ``|prediction_date_offset|`` **trading days before** the announcement date.
    The constituent set is tracked across periods to support the buffer mechanism
    and to compute predicted additions correctly.

    Parameters
    ----------
    events : pd.DataFrame
        From load_events().  Must contain announcement_date, effective_date,
        added_stocks, removed_stocks, event_id.  Must be sorted chronologically.
    prices_wide : pd.DataFrame
    per_df, turnover_wide, shares_df : optional data.
    prediction_date_offset : int
        Negative integer.  E.g. -14 means predict 14 trading days before
        announcement.  Strictly no data from announcement date onwards is used.
    top_k : int
        Number of constituents to select.
    initial_constituents : set or None
        Known constituent list before the first event.  If None, starts empty.

    Returns
    -------
    pd.DataFrame
        Columns: event_id, announcement_date, prediction_date,
                 predicted (list), actual_added (list),
                 precision, recall, f1, n_predicted, n_actual, n_correct,
                 correct (list), false_positives (list), false_negatives (list).
    """
    assert prediction_date_offset < 0, "prediction_date_offset must be negative"

    trading_days = prices_wide.index
    events_sorted = events.sort_values("announcement_date").reset_index(drop=True)

    current_constituents = set(initial_constituents) if initial_constituents else set()
    records = []

    for _, event in events_sorted.iterrows():
        ann_date = pd.Timestamp(event["announcement_date"])
        eff_date = pd.Timestamp(event["effective_date"])

        # ── Compute prediction date (trading days before announcement) ──
        ann_future = trading_days[trading_days >= ann_date]
        if ann_future.empty:
            logger.warning("Event %s: announcement date out of trading day range — skip",
                           event.get("event_id"))
            continue

        ann_idx = trading_days.get_loc(ann_future[0])
        pred_idx = ann_idx + prediction_date_offset  # offset is negative

        if pred_idx < 0:
            logger.warning("Event %s: prediction date before data start — skip",
                           event.get("event_id"))
            continue

        pred_date = trading_days[pred_idx]

        # ── Safety check: pred_date must be strictly before ann_date ──
        assert pred_date < ann_date, (
            f"Look-ahead bias detected: pred_date={pred_date.date()} >= "
            f"ann_date={ann_date.date()}"
        )

        # ── Predict ──
        predicted_additions = predict_additions(
            ref_date=pred_date,
            prices_wide=prices_wide,
            per_df=per_df,
            turnover_wide=turnover_wide,
            shares_df=shares_df,
            current_constituents=current_constituents,
            top_k=top_k,
            **kwargs,
        )

        actual_added = list(event["added_stocks"])
        metrics = evaluate_prediction(predicted_additions, actual_added)

        records.append({
            "event_id":          event.get("event_id"),
            "announcement_date": ann_date,
            "prediction_date":   pred_date,
            "predicted":         predicted_additions,
            "actual_added":      actual_added,
            **{k: metrics[k] for k in
               ["precision", "recall", "f1", "n_predicted", "n_actual",
                "n_correct", "correct", "false_positives", "false_negatives"]},
        })

        # ── Update constituent set after effective date ──
        # (simulate the index rebalancing for the next event's buffer)
        current_constituents -= set(event["removed_stocks"])
        current_constituents |= set(event["added_stocks"])
        logger.info(
            "Event %s  pred=%s  ann=%s  precision=%.2f  recall=%.2f",
            event.get("event_id"), pred_date.date(), ann_date.date(),
            metrics["precision"], metrics["recall"],
        )

    return pd.DataFrame(records)
