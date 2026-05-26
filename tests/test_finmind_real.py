"""
tests/test_finmind_real.py
--------------------------
驗證 FinMind API 能否抓到真實資料，並對每筆資料執行 verify_real_data() 檢查。

執行：
    cd <project_root>
    python tests/test_finmind_real.py

任何抓取失敗都會直接 raise，印出原始錯誤，不 fallback。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_fetcher import fetch_stock_daily, fetch_market_index, verify_real_data, FINMIND_TOKEN

START = "2024-01-01"
END   = "2024-01-31"

# ── Token 狀態 ────────────────────────────────────────────────────────────────
print(f"FINMIND_TOKEN 設定狀態: {'有 token' if FINMIND_TOKEN else '無 token（匿名存取）'}")
print("=" * 60)


def _show(label: str, df):
    """印出 DataFrame 摘要，任何問題直接 raise。"""
    if df.empty:
        raise Exception(f"{label} 回傳空 DataFrame — API 失敗或該區間無資料")

    print(f"\n筆數        : {len(df)}")
    print(f"欄位        : {list(df.columns)}")
    print(f"\n前 3 列：\n{df.head(3).to_string()}")
    print(f"\n後 3 列：\n{df.tail(3).to_string()}")


# ── 1. 2330 台積電 ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[1] 2330 台積電  {START} ~ {END}")
print(f"{'='*60}")

df_2330 = fetch_stock_daily("2330", START, END)
_show("2330", df_2330)

print("\n--- verify_real_data() ---")
ok_2330 = verify_real_data(df_2330, stock_id="2330")
print(f"verify_real_data 回傳: {ok_2330}")


# ── 2. TAIEX 加權指數 ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[2] TAIEX 加權指數  {START} ~ {END}")
print(f"{'='*60}")

df_taiex = fetch_market_index(START, END)
_show("TAIEX", df_taiex)

print("\n--- verify_real_data() ---")
ok_taiex = verify_real_data(df_taiex, stock_id="TAIEX")
print(f"verify_real_data 回傳: {ok_taiex}")


# ── 3. 00919 ETF ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[3] 00919 ETF  {START} ~ {END}")
print(f"{'='*60}")

df_00919 = fetch_stock_daily("00919", START, END)
_show("00919", df_00919)

print("\n--- verify_real_data() ---")
ok_00919 = verify_real_data(df_00919, stock_id="00919")
print(f"verify_real_data 回傳: {ok_00919}")


# ── 總結 ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("【驗證結果總覽】")
print(f"  2330  : {'✓ 通過' if ok_2330  else '⚠ 有可疑指標'}")
print(f"  TAIEX : {'✓ 通過' if ok_taiex  else '⚠ 有可疑指標'}")
print(f"  00919 : {'✓ 通過' if ok_00919 else '⚠ 有可疑指標'}")
print(f"{'='*60}")
print("全部資料抓取成功，FinMind 真實資料可正常存取。")
