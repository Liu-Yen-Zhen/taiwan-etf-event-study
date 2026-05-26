"""
scripts/fetch_stock_prices.py
-----------------------------
Fetch real daily prices for the full analysis universe:

  • All distinct added / removed stocks listed in data/raw/events.csv
  • TAIEX (market benchmark)
  • 00919 (ETF itself, for premium/discount analysis)

Window: 2022-09-01 → 2026-01-31 (covers all 7 events ± buffer).

Outputs
-------
  data/processed/stock_prices.parquet
      Long-format: date, stock_id, open, high, low, close, volume, ...
  data/processed/fetch_failures.csv
      stock_id, error_message, timestamp  (only if any failure occurs)

Behaviour
---------
  • Uses src/data_fetcher.fetch_stock_daily — no rewrites.
  • Per-stock failures are recorded and the run continues (we expect a
    handful of delisted / renamed tickers across a 3-year horizon).
  • Each successful fetch is sanity-checked with verify_real_data();
    warnings are aggregated and reported, but do not abort the run.
  • Zero fallback to synthetic data.
"""

import io
import sys
import time
import contextlib
import random
import logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from data_fetcher import fetch_stock_daily, verify_real_data, load_events

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

START = "2022-09-01"
END   = "2026-01-31"
SLEEP = 0.6  # seconds between API calls (FinMind free tier ~1 req/s)

EVENTS_PATH  = ROOT / "data" / "raw" / "events.csv"
OUT_PARQUET  = ROOT / "data" / "processed" / "stock_prices.parquet"
OUT_FAILURES = ROOT / "data" / "processed" / "fetch_failures.csv"


def _silent_verify(df: pd.DataFrame, sid: str) -> tuple[bool, str]:
    """Run verify_real_data() and capture its stdout for later inspection."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ok = verify_real_data(df, stock_id=sid)
    return ok, buf.getvalue()


def main() -> None:
    events = load_events(EVENTS_PATH)
    print(f"Loaded {len(events)} events from {EVENTS_PATH.name}")

    # ── Build deduplicated stock universe ─────────────────────────────────
    added: set[str] = set()
    removed: set[str] = set()
    for _, row in events.iterrows():
        added.update(row["added_stocks"])
        removed.update(row["removed_stocks"])
    overlap = added & removed

    universe_stocks = sorted(added | removed)
    universe = universe_stocks + ["TAIEX", "00919"]

    print(f"  Added (unique)   : {len(added)}")
    print(f"  Removed (unique) : {len(removed)}")
    print(f"  Overlap          : {len(overlap)}")
    print(f"  Stocks (deduped) : {len(universe_stocks)}")
    print(f"  + TAIEX, 00919   → {len(universe)} total to fetch")
    print(f"  Window           : {START} → {END}")
    print(f"  Sleep / call     : {SLEEP}s   "
          f"(≈ {len(universe) * SLEEP / 60:.1f} min minimum)")
    print("=" * 70)

    all_frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    verify_failed: dict[str, str] = {}   # sid → captured verify output

    for i, sid in enumerate(universe, 1):
        print(f"  [{i:3d}/{len(universe)}] {sid:>6s}  …", end=" ", flush=True)

        try:
            df = fetch_stock_daily(sid, START, END)
        except Exception as exc:
            err = repr(exc)
            failures.append({
                "stock_id": sid,
                "error_message": err,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            print(f"FETCH FAILED  →  {err}")
            time.sleep(SLEEP)
            continue

        if df.empty:
            failures.append({
                "stock_id": sid,
                "error_message": "empty DataFrame returned (no data in window)",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            print("EMPTY (no rows in window)")
            time.sleep(SLEEP)
            continue

        # Ensure stock_id column is present (TaiwanStockPrice already includes it,
        # but be defensive).
        if "stock_id" not in df.columns:
            df["stock_id"] = sid

        ok, captured = _silent_verify(df, sid)
        if not ok:
            verify_failed[sid] = captured

        all_frames.append(df.reset_index())   # promote date back to column
        flag = "  ⚠ verify" if not ok else ""
        print(f"OK  ({len(df):4d} rows){flag}")

        time.sleep(SLEEP)

    # ── Persist outputs ────────────────────────────────────────────────────
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(OUT_PARQUET, index=False)
        print(f"\n✓ Wrote {len(combined):,} rows → {OUT_PARQUET.relative_to(ROOT)}")
    else:
        print("\n⚠ No data fetched — parquet NOT written.")

    if failures:
        pd.DataFrame(failures).to_csv(OUT_FAILURES, index=False)
        print(f"⚠ Wrote {len(failures)} failures → {OUT_FAILURES.relative_to(ROOT)}")
    elif OUT_FAILURES.exists():
        OUT_FAILURES.unlink()
        print(f"(removed stale {OUT_FAILURES.relative_to(ROOT)} — no failures this run)")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Attempted           : {len(universe)}")
    print(f"  Successful fetches  : {len(all_frames)}")
    print(f"  Failed fetches      : {len(failures)}")
    print(f"  verify_real_data ⚠  : {len(verify_failed)}")

    if failures:
        print("\n  Failure list:")
        for f in failures:
            print(f"    • {f['stock_id']:>6s}  {f['error_message']}")

    if verify_failed:
        print("\n  verify_real_data flagged (not fatal, just a heads-up):")
        for sid in verify_failed:
            print(f"    • {sid}")


if __name__ == "__main__":
    main()
