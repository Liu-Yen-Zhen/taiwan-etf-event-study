"""
src/data_fetcher.py
-------------------
FinMind API wrappers for fetching Taiwan stock price data and ETF event lists.

FinMind free tier constraints:
  - ~600 requests/day, rate-limited; we sleep between calls to avoid 429s.
  - Authentication via token in .env (FINMIND_TOKEN). Anonymous calls also
    work but hit a tighter limit.

All returned DataFrames use a DatetimeIndex named "date" and column names
in snake_case. No data is modified beyond renaming and type-casting —
callers are responsible for any further cleaning.
"""

import os
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "")

# Minimum sleep between successive API calls (seconds).
# FinMind free tier allows roughly 1 req/sec sustained; 0.5 s is conservative.
INTER_CALL_SLEEP = 0.5

# Column rename map: FinMind raw name → our standard name
_PRICE_COL_MAP = {
    "date": "date",
    "stock_id": "stock_id",
    "Trading_Volume": "volume",
    "Trading_money": "turnover",
    "open": "open",
    "max": "high",
    "min": "low",
    "close": "close",
    "spread": "price_change",
    "Trading_turnover": "num_trades",
}

# ── Internal helpers ───────────────────────────────────────────────────────────


def _build_params(dataset: str, stock_id: str, start_date: str, end_date: str) -> dict:
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN
    return params


@retry(
    retry=retry_if_exception_type((requests.HTTPError, requests.ConnectionError, requests.Timeout)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _get(params: dict) -> dict:
    """Single GET to FinMind API with retry logic."""
    resp = requests.get(FINMIND_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 200:
        # FinMind returns HTTP 200 even for errors; check inner status
        raise RuntimeError(
            f"FinMind API error {payload.get('status')}: {payload.get('msg', 'unknown')}"
        )
    return payload


def _clean_corporate_action_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Convert FinMind 'corporate action' artifact rows to NaN for OHLC.

    FinMind's TaiwanStockPrice dataset encodes some corporate actions (cash
    dividends, capital reductions, mergers, split-adjustment placeholders) by
    emitting a row with **all OHLC = 0 and volume = 0**. These are not real
    prices and must be treated as missing data — leaving them as 0 produces
    spurious −100 % daily returns and corrupts downstream CAR calculations
    (verified: 5347 世界先進 on 2024-06-05 was a clean reproduction case).

    The cleanup is conservative: it only blanks OHLC + price_change on rows
    where ``close == 0``. ``volume`` and ``turnover`` are left as 0 (which
    is accurate — no trades did occur), and rows are kept in the frame so
    downstream date-arithmetic remains intact.
    """
    if "close" not in df.columns:
        return df
    artifact_mask = (df["close"] == 0)
    if not artifact_mask.any():
        return df
    for col in ["open", "high", "low", "close", "price_change"]:
        if col in df.columns:
            df.loc[artifact_mask, col] = np.nan
    return df


def _parse_price_response(payload: dict) -> pd.DataFrame:
    """Convert FinMind price payload into a cleaned DataFrame."""
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Keep only columns we have a mapping for; ignore extras silently
    cols_present = [c for c in _PRICE_COL_MAP if c in df.columns]
    df = df[cols_present].rename(columns=_PRICE_COL_MAP)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df.index.name = "date"

    # Cast numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "turnover",
                    "price_change", "num_trades"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert corporate-action artifact rows (close == 0) to NaN
    df = _clean_corporate_action_rows(df)

    return df


# ── Public API ─────────────────────────────────────────────────────────────────


def fetch_stock_daily(
    stock_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily OHLCV for a single Taiwan stock via FinMind.

    Parameters
    ----------
    stock_id : str
        Stock ticker, e.g. "2330", "00919".
    start_date : str
        ISO format, e.g. "2024-01-01".
    end_date : str
        ISO format, e.g. "2024-03-31".

    Returns
    -------
    pd.DataFrame
        DatetimeIndex("date"), columns: open, high, low, close, volume,
        turnover, price_change, num_trades, stock_id.
        Empty DataFrame if no data returned.
    """
    params = _build_params("TaiwanStockPrice", stock_id, start_date, end_date)
    payload = _get(params)
    df = _parse_price_response(payload)
    logger.info("fetch_stock_daily(%s) → %d rows", stock_id, len(df))
    return df


def fetch_market_index(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch TAIEX (Taiwan Weighted Index) daily prices via FinMind.

    Uses the same TaiwanStockPrice dataset with stock_id="TAIEX".

    Parameters
    ----------
    start_date : str
        ISO format, e.g. "2024-01-01".
    end_date : str
        ISO format, e.g. "2024-12-31".

    Returns
    -------
    pd.DataFrame
        DatetimeIndex("date"), columns: open, high, low, close, volume, …
        Empty DataFrame if no data returned.
    """
    params = _build_params("TaiwanStockPrice", "TAIEX", start_date, end_date)
    payload = _get(params)
    df = _parse_price_response(payload)
    logger.info("fetch_market_index() → %d rows", len(df))
    return df


def fetch_multiple_stocks(
    stock_ids: list[str],
    start_date: str,
    end_date: str,
    sleep_seconds: float = INTER_CALL_SLEEP,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for multiple stocks, respecting FinMind rate limits.

    Parameters
    ----------
    stock_ids : list[str]
        List of stock tickers.
    start_date : str
        ISO format.
    end_date : str
        ISO format.
    sleep_seconds : float
        Pause between API calls. Default 0.5 s (conservative for free tier).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of stock_id → DataFrame. Stocks that returned no data are
        included with an empty DataFrame so callers can detect gaps.
    """
    results: dict[str, pd.DataFrame] = {}
    total = len(stock_ids)
    for i, sid in enumerate(stock_ids, start=1):
        logger.info("Fetching %s (%d/%d) …", sid, i, total)
        try:
            results[sid] = fetch_stock_daily(sid, start_date, end_date)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", sid, exc)
            results[sid] = pd.DataFrame()
        if i < total:
            time.sleep(sleep_seconds)
    return results


# ── Event list loader ──────────────────────────────────────────────────────────

_EVENTS_PATH = Path(__file__).parent.parent / "data" / "raw" / "events.csv"


def load_events(path: str | Path = _EVENTS_PATH) -> pd.DataFrame:
    """Load and parse the manually curated ETF reconstitution event list.

    Expected CSV schema
    -------------------
    event_id, etf_code, announcement_date, effective_date,
    added_stocks, removed_stocks

    ``added_stocks`` and ``removed_stocks`` are "|"-delimited ticker strings,
    e.g. "2330|2317|2454". Empty cells become empty lists [].

    Returns
    -------
    pd.DataFrame
        Columns:
          event_id          str   e.g. "00919_20221216"
          etf_code          str   e.g. "00919"
          announcement_date datetime64[ns]
          effective_date    datetime64[ns]
          added_stocks      list[str]
          removed_stocks    list[str]

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist yet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Event list not found at {path}. "
            "Please create data/raw/events.csv before calling load_events()."
        )

    df = pd.read_csv(path, dtype=str)

    # event_id is a plain string (e.g. "00919_20221216"); no numeric cast.
    df["event_id"] = df["event_id"].str.strip()
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])
    df["effective_date"] = pd.to_datetime(df["effective_date"])

    def _parse_tickers(cell: str) -> list[str]:
        if pd.isna(cell) or str(cell).strip() == "":
            return []
        return [t.strip() for t in str(cell).split("|") if t.strip()]

    df["added_stocks"] = df["added_stocks"].apply(_parse_tickers)
    df["removed_stocks"] = df["removed_stocks"].apply(_parse_tickers)

    return df.reset_index(drop=True)


# ── Sanity check ───────────────────────────────────────────────────────────────


def verify_real_data(df: pd.DataFrame, stock_id: str = "") -> bool:
    """Sanity-check a price DataFrame for signs of synthetic / simulated data.

    This function does NOT raise — it emits ``logger.warning()`` calls and
    prints a human-readable summary. Returns True if the data looks real,
    False if one or more red-flags were found.

    Checks performed
    ----------------
    1. Required columns (date index + close) present.
    2. Stock ID format: "S001"–"S999" pattern → almost certainly synthetic.
    3. Starting close price is a perfectly round number (e.g. 100.0, 15.0).
    4. Return kurtosis near zero (real daily returns are fat-tailed; κ ≫ 0).
    5. Shapiro-Wilk normality test passes (real returns are non-normal).
    6. Date gaps: all consecutive gaps are exactly 1 calendar day (synthetic
       ``bdate_range`` leaves no weekends / holiday gaps).

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame as returned by ``fetch_stock_daily`` / ``fetch_market_index``.
        Expected to have a DatetimeIndex named "date" and a "close" column.
    stock_id : str
        Optional ticker label used in warning messages.

    Returns
    -------
    bool
        True  → no red-flags detected (data looks real).
        False → at least one red-flag detected.
    """
    import re
    import numpy as np
    from scipy import stats

    tag = f"[{stock_id}] " if stock_id else ""
    flags: list[str] = []

    # ── 1. Required columns ────────────────────────────────────────────────────
    if "close" not in df.columns:
        msg = f"{tag}缺少 'close' 欄位 — 無法進行 sanity check"
        logger.warning(msg)
        print(f"⚠  {msg}")
        return False

    if not isinstance(df.index, pd.DatetimeIndex):
        msg = f"{tag}index 不是 DatetimeIndex — 請先 set_index('date')"
        logger.warning(msg)
        print(f"⚠  {msg}")
        return False

    close = df["close"].dropna()
    if len(close) < 5:
        msg = f"{tag}收盤價資料不足 5 筆，無法判斷"
        logger.warning(msg)
        print(f"⚠  {msg}")
        return False

    # ── 2. Synthetic stock ID pattern ─────────────────────────────────────────
    if re.fullmatch(r"S\d{3}", stock_id or ""):
        msg = f"{tag}股票代號符合合成資料格式 'S###'，高度懷疑為模擬資料"
        logger.warning(msg)
        flags.append("合成代號 (S###)")

    # ── 3. Suspicious starting price ──────────────────────────────────────────
    # Taiwan stocks close in whole TWD, so integer prices are normal.
    # We only flag prices that match known synthetic "seed" values used by
    # NumPy simulation code (e.g. 100.0, 15.0, 18.0, 5.0, 50.0).
    _SYNTHETIC_SEED_PRICES = {5.0, 10.0, 15.0, 18.0, 20.0, 50.0, 100.0}
    first_price = float(close.iloc[0])
    if first_price in _SYNTHETIC_SEED_PRICES:
        msg = (
            f"{tag}起始收盤價為 {first_price:.1f} — "
            "此值為已知合成資料常見初始值（100, 50, 18, 15, 5 等），"
            "請確認資料來源"
        )
        logger.warning(msg)
        flags.append(f"可疑種子價格 ({first_price:.0f})")

    # ── 4. Return kurtosis near zero ──────────────────────────────────────────
    # Requires at least 60 observations: kurtosis estimates are too noisy
    # on short windows (e.g. one month of trading days ≈ 22 obs).
    returns = close.pct_change().dropna()
    _MIN_STAT_OBS = 60
    if len(returns) >= _MIN_STAT_OBS:
        kurt = float(returns.kurt())   # excess kurtosis; normal = 0
        if abs(kurt) < 0.5:
            msg = (
                f"{tag}報酬率超額峰度 = {kurt:.3f}（接近 0）— "
                "真實股票日報酬通常有厚尾（excess kurtosis > 1），"
                "接近 0 暗示正態分布模擬資料"
            )
            logger.warning(msg)
            flags.append(f"峰度過低 (κ={kurt:.2f})")

        # ── 5. Shapiro-Wilk normality test ────────────────────────────────────
        # Use at most 5 000 observations (Shapiro-Wilk limit).
        # Also requires _MIN_STAT_OBS for the same reason as kurtosis.
        sample = returns.values[:5_000]
        _, p_val = stats.shapiro(sample)
        if p_val > 0.05:
            msg = (
                f"{tag}Shapiro-Wilk 正態性檢定 p = {p_val:.4f} > 0.05 — "
                "無法拒絕正態假設，真實股票日報酬通常顯著偏離正態分布"
            )
            logger.warning(msg)
            flags.append(f"報酬近似正態分布 (p={p_val:.3f})")
    else:
        logger.debug(
            "%s樣本數 %d < %d，跳過峰度 / 正態性檢定（短窗口不可靠）",
            tag, len(returns), _MIN_STAT_OBS,
        )

    # ── 6. Date gap pattern ────────────────────────────────────────────────────
    dates = df.index.sort_values()
    if len(dates) >= 10:
        gaps = (dates[1:] - dates[:-1]).days  # numpy array of integer days
        unique_gaps = set(gaps)
        if unique_gaps == {1}:
            msg = (
                f"{tag}所有日期間距恰好為 1 日（共 {len(dates)} 筆）— "
                "真實交易日資料應有週末與假日空缺，完全連續暗示 bdate_range 合成"
            )
            logger.warning(msg)
            flags.append("日期完全連續（無假日缺口）")
        elif 1 in unique_gaps and len(unique_gaps) == 1:
            # Same as above but expressed differently — already covered
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  verify_real_data()  {tag}({len(close)} 筆收盤價)")
    print(f"{'─'*55}")
    if flags:
        print(f"  ⚠  發現 {len(flags)} 個可疑指標：")
        for f in flags:
            print(f"       • {f}")
        print("  → 請確認此資料來源為真實市場資料，而非合成資料。")
        print(f"{'─'*55}\n")
        return False
    else:
        print("  ✓  未發現合成資料特徵，資料看起來正常。")
        print(f"{'─'*55}\n")
        return True
