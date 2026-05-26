"""
scripts/fetch_universe_data.py
------------------------------
Step A2：對 static universe 中尚未抓過的股票，補抓 prices + shares + 全市場 delisting。

預期執行：
  • 137 股 - 87 (已有 prices) = 50 檔需要抓 prices
  • 137 股 × 1 calls = 137 calls 抓 shares (TaiwanStockShareholding 全期間)
  • 1 call 抓 TaiwanStockDelisting

總 API 約 188 calls × 0.6 s ≈ 2 分鐘。
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests
import pandas as pd
from data_fetcher import fetch_stock_daily, verify_real_data, _get, _build_params, _clean_corporate_action_rows

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

UNIVERSE_CSV  = ROOT / "data" / "raw"       / "static_universe.csv"
PRICES_PQ     = ROOT / "data" / "processed" / "stock_prices.parquet"
SHARES_OUT_PQ = ROOT / "data" / "processed" / "shareholding_v2.parquet"
DELIST_OUT    = ROOT / "data" / "processed" / "delisting.parquet"

DATA_START = "2022-09-01"
DATA_END   = "2026-01-31"
SLEEP      = 0.6


def fetch_shareholding(stock_id, start, end):
    """Fetch TaiwanStockShareholding for one stock over a date range."""
    params = _build_params("TaiwanStockShareholding", stock_id, start, end)
    payload = _get(params)
    data = payload.get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    keep = ["date", "stock_id", "NumberOfSharesIssued",
            "ForeignInvestmentSharesRatio"]
    return df[[c for c in keep if c in df.columns]]


def main():
    universe = pd.read_csv(UNIVERSE_CSV, dtype={"stock_id": str})
    print(f"Universe size: {len(universe)} stocks")

    # ── (a) 找出尚未有 prices 的股票 ────────────────────────────────
    existing = pd.read_parquet(PRICES_PQ)
    existing_ids = set(existing["stock_id"].unique())
    print(f"已有 prices: {len(existing_ids)} 檔（含 TAIEX、00919）")

    universe_ids = set(universe["stock_id"])
    need_prices = sorted(universe_ids - existing_ids)
    print(f"需要新抓 prices: {len(need_prices)} 檔")

    # ── (b) 抓新股票 prices ────────────────────────────────────────
    new_frames = []
    fail_prices = []
    for i, sid in enumerate(need_prices, 1):
        print(f"  [price {i:3d}/{len(need_prices)}] {sid} …", end=" ", flush=True)
        try:
            df = fetch_stock_daily(sid, DATA_START, DATA_END)
        except Exception as exc:
            fail_prices.append((sid, repr(exc)))
            print(f"FAIL: {exc}")
            time.sleep(SLEEP); continue
        if df.empty:
            fail_prices.append((sid, "empty"))
            print("EMPTY"); time.sleep(SLEEP); continue
        if "stock_id" not in df.columns:
            df["stock_id"] = sid
        new_frames.append(df.reset_index())
        print(f"OK ({len(df)} rows)")
        time.sleep(SLEEP)

    if new_frames:
        new_long = pd.concat(new_frames, ignore_index=True)
        # 合併進主 parquet
        combined = pd.concat([existing, new_long], ignore_index=True)
        combined.to_parquet(PRICES_PQ, index=False)
        print(f"\n✓ stock_prices.parquet 更新: {len(existing):,} → {len(combined):,} 列")

    # ── (c) 抓全市場 TaiwanStockDelisting (1 call) ─────────────────
    print(f"\n[delisting] 抓全市場下市清單 …")
    resp = requests.get("https://api.finmindtrade.com/api/v4/data",
        params={"dataset": "TaiwanStockDelisting"}, timeout=30)
    j = resp.json()
    if not j.get("data"):
        raise RuntimeError(f"TaiwanStockDelisting failed: {j.get('msg','?')}")
    delist = pd.DataFrame(j["data"])
    delist["date"] = pd.to_datetime(delist["date"])
    delist.to_parquet(DELIST_OUT, index=False)
    print(f"  ✓ {len(delist)} 筆下市記錄 → {DELIST_OUT.relative_to(ROOT)}")

    # ── (d) 抓 universe 所有股的 shareholding (137 calls) ──────────
    print(f"\n[shares] 抓 {len(universe_ids)} 檔股票的 NumberOfSharesIssued …")
    share_frames = []
    fail_shares = []
    for i, sid in enumerate(sorted(universe_ids), 1):
        print(f"  [share {i:3d}/{len(universe_ids)}] {sid} …", end=" ", flush=True)
        try:
            df = fetch_shareholding(sid, DATA_START, DATA_END)
        except Exception as exc:
            fail_shares.append((sid, repr(exc)))
            print(f"FAIL: {exc}"); time.sleep(SLEEP); continue
        if df.empty:
            fail_shares.append((sid, "empty"))
            print("EMPTY"); time.sleep(SLEEP); continue
        share_frames.append(df)
        print(f"OK ({len(df)} rows)")
        time.sleep(SLEEP)

    if share_frames:
        shares = pd.concat(share_frames, ignore_index=True)
        shares.to_parquet(SHARES_OUT_PQ, index=False)
        print(f"\n✓ shareholding_v2.parquet: {len(shares):,} 列 ({shares['stock_id'].nunique()} 檔股票)")

    # ── 總結 ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step A2 完成")
    print("=" * 60)
    print(f"  新抓 prices       : {len(new_frames)} 檔成功，{len(fail_prices)} 檔失敗")
    print(f"  delisting         : 1 檔（全市場 {len(delist)} 筆）")
    print(f"  shares 取得       : {len(share_frames)}/{len(universe_ids)} 檔")
    if fail_prices:
        print(f"\n  Price 失敗清單:")
        for sid, err in fail_prices:
            print(f"    {sid}: {err}")
    if fail_shares:
        print(f"\n  Shares 失敗清單:")
        for sid, err in fail_shares[:10]:
            print(f"    {sid}: {err}")


if __name__ == "__main__":
    main()
