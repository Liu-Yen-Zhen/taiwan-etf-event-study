# 台灣高息 ETF 指數調整的反向效應

> 事件研究 × 預測模型 × 策略可行性檢驗  
> 0056 × 00713 × 00919｜38 個調整事件｜2017–2025

研究台灣三檔高股息 ETF（0056、00713、00919）指數成分股調整事件的價格效應。完整流程涵蓋：事件研究（CAR、t-test）、橫斷面迴歸、機器學習預測模型（LR / RF / XGBoost）、假設策略回測，以及成本敏感度分析。

📄 **作品集 PDF**：[`output/portfolio_etf_research.pdf`](output/portfolio_etf_research.pdf)（15 頁，含完整視覺化）

---

## 核心發現

| 視窗 | 整體（N=303） | 00919（N=63） | 00713（N=179） | 0056（N=61） |
|------|-------------|--------------|--------------|------------|
| [−30, 0] | **+2.19%\*\*\*** | **+5.05%\*\*\*** | +1.33%** | +1.75%（—） |
| [+1, +31] | **−4.20%\*\*\*** | **−8.30%\*\*\*** | **−3.39%\*\*\*** | −2.35%** |

**反向型態跨三個 ETF 成立**：公告前正向 run-up → 生效後反轉，方向與美股「指數納入溢價」相反。

**機制假說「規則透明度 ↔ 前移強度」獲跨 ETF 內部驗證：**
- 00919（TIP 規則透明）→ 強前移 → 即時反轉
- 00713（TIP 規則透明）→ 中等前移 → 遲滯反轉
- 0056（FTSE 規則含外資可買性等不可觀測條件）→ 無前移 → 公告後反而上漲（自然對照組）

**預測模型**：00919 單一 XGB Top-10 Precision = 50%（基準 9.53%）；多 ETF LR Top-10 = 22%（基準 6.09%）。殖利率相關特徵合計重要性 49.3%，跨 ETF 穩健。

**策略結論**：OOS 期間 break-even 31.6 bps < 實際成本 58.5 bps，目前不可行；誠實揭露 null result 並標記改善路徑。

---

## 專案結構

```
.
├── docs/                          # 研究文件（findings + 簡報 + 進度）
│   ├── findings_stage2_multi_etf.md   # 多 ETF CAR 事件研究
│   ├── findings_stage4_multi_etf.md   # 多 ETF 預測模型
│   ├── findings_stage2.md ~ stage6.md # 00919 單一 ETF 完整鏈
│   ├── presentation_outline.md        # 10 張簡報大綱
│   └── progress_overview.md           # 進度總覽
├── scripts/                       # 可執行腳本（資料抓取、建模、回測）
│   ├── fetch_universe_data.py
│   ├── build_prediction_panel_multi_etf.py
│   ├── train_models_multi_etf.py
│   ├── run_car_analysis_multi_etf.py
│   └── ...
├── src/                           # 模組化核心程式碼
│   ├── data_fetcher.py            # FinMind API wrapper
│   ├── event_study.py             # CAR / AR 計算
│   ├── cross_section.py           # 橫斷面 OLS
│   ├── prediction.py              # 預測模型 pipeline
│   ├── backtest.py                # 假設策略回測
│   └── sensitivity.py             # 成本敏感度
├── notebooks/                     # 探索性 Jupyter notebook（02–07）
├── output/
│   ├── figures/                   # 所有圖表 PNG
│   ├── tables/                    # CAR / 預測 / 回測結果 CSV
│   └── portfolio_etf_research.{docx,pdf}  # 作品集
└── tests/                         # 單元測試
```

---

## 快速重現

```bash
# 1. 安裝相依套件
pip install -r requirements.txt

# 2. 設定 FinMind API token（免費註冊 https://finmindtrade.com/）
cp .env.example .env
# 編輯 .env 填入 FINMIND_TOKEN=your_token

# 3. 抓取資料（會建立 data/raw/ 與 data/processed/）
python scripts/fetch_universe_data.py
python scripts/build_prediction_panel_multi_etf.py

# 4. 跑事件研究 & 模型
python scripts/run_car_analysis_multi_etf.py
python scripts/train_models_multi_etf.py
```

所有結果會輸出到 `output/figures/` 與 `output/tables/`，與 repo 中現有檔案應完全重現。

---

## 方法論

- **異常報酬模型**：市場調整模型 AR = r_stock − r_TAIEX（簡化假設 β=1；β 修正討論見 `findings_stage2.md` §5.5）
- **事件視窗**：[−30, 0], [−5, 0], [0, +5], [+1, +11], [+1, +31]（交易日）
- **預測模型 Train/Test 切分**：嚴格時序切分，Train ≤2023、Test ≥2024，避免 look-ahead
- **特徵集（Version A，多 ETF 主版本）**：7 個事前可觀察特徵（殖利率排名、log 成交額、波動率、beta、動能、is_previous、ETF dummies）
- **不平衡處理**：LR / RF `class_weight='balanced'`；XGB `scale_pos_weight = neg/pos`
- **評估指標**：Top-K Precision（K=5/8/10/15/20）、AUC-ROC、各 ETF 子集 Top-10

---

## 資料

- **價格**：FinMind `TaiwanStockPrice`（234 檔，2016–2026 日線）
- **指數**：加權股價指數（TAIEX）
- **殖利率與股本**：FinMind `TaiwanStockPER`、`TaiwanStockShareholding`
- **事件清單**：臺灣指數公司公告（TIP，00713 / 00919）+ FTSE Russell 公告（0056）

**資料原則**：100% 真實資料，無合成、無 fallback、無 look-ahead。FinMind 抓取失敗會 raise Exception，絕不使用任何形式的合成資料。

---

## 研究紀律

研究過程中主動發現並修正兩個方法論問題（記錄於 `findings_stage3.md`）：

| | v1（有問題） | v2（修正後） | 修正原因 |
|---|---|---|---|
| CAR_pre_30 R² | 0.345（虛高） | 0.072（真實） | 動能視窗重疊 24 天 → 機械式自相關 |
| CAR_post_30 R² | 0.214 | 0.312 | 5347 世界先進 close=0 artifact（Cook's D=0.445） |

---

## 限制

- OOS 樣本量小（00919 = 3 個事件），無法區分「alpha 消失」vs「隨機波動」
- β=1 假設可能低估 CAR 幅度約 0.6pp
- 候選池 234 檔 < 實際 600+ 有息股，正樣本率被高估
- 0056 規則不透明，FTSE Russell 「外資可買性」等條件機制為推斷
- 樣本跨越 2017–2025 不同市場環境，早期事件 ETF AUM 較小

---

## License

研究用途，無商業授權。原始資料屬 FinMind / 臺灣指數公司 / FTSE Russell。

---

*Author: 劉晏禎 ｜ 2026 年 5 月*
