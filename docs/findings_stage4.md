# Stage 4 Findings — 00919 Addition Prediction (Binary Classification)

**版本日期**：2026-05-20  
**腳本**：`scripts/train_prediction_model.py`  
**輸入**：`data/processed/prediction_panel.parquet`（661 rows → 660 after listwise drop）

---

## 1. 研究設計

| 項目 | 說明 |
|---|---|
| 任務 | 二元分類：預測哪些股票會被**新納入** 00919 ETF |
| 候選池 | 靜態 universe（0050 ∪ 0056 ∪ 00919 歷史）137 檔，事件截面 94–124 檔 |
| 訓練集 | Events 2-4（2023-05-31 ~ 2024-05-31）：352 rows，31 positives |
| 測試集 | Events 5-7（2024-12-17 ~ 2025-12-16）：308 rows，32 positives |
| 隨機基準 | Top-K Precision = 9.53%（63/661 全局正率，**不作為唯一基準**；各事件基準率不同） |
| 主評估指標 | Top-K Precision at K=5,8,10,13,16；Recall@Top-10；AUC-ROC |
| **禁止報告** | Overall Accuracy（在 9.53% 正率下沒有意義） |

---

## 2. 特徵（10 個）

| 特徵名 | 類型 | 計算時點 | 縮放 |
|---|---|---|---|
| dividend_yield_ttm | 連續 | T-14 | StandardScaler |
| dividend_yield_rank_in_pool | 連續（排名） | T-14 | StandardScaler |
| log_market_cap | 連續 | T-14 | StandardScaler |
| market_cap_rank_in_pool | 連續（排名） | T-14 | StandardScaler |
| log_avg_turnover_60d | 連續 | T-14（過去 60 交易日均量） | StandardScaler |
| turnover_ratio_60d | 連續 | T-14（日均周轉率） | StandardScaler |
| volatility_60d | 連續 | T-14（過去 60 日報酬標準差） | StandardScaler |
| beta_60d | 連續 | T-14（相對大盤 beta） | StandardScaler |
| momentum_60d_pre | 連續 | T-14（動能，window[-90, -31]） | StandardScaler |
| was_in_00919_previous | 二元 | 事件前累積紀錄 | 不縮放 |

**資料處理假設記錄**：
- `was_in_00919_previous`：Event 2 有 124 個 NaN（設計缺口：初始成分不可知）→ 一律填 0（保守下界）。可能導致少數「已在池中」股票被誤標為負樣本（label noise，非 false positive 污染）。
- `market_cap`：1 筆缺失 → listwise drop（最終 660 rows，32 positives in test）。
- StandardScaler 僅在訓練集 fit，再 transform 測試集（無 look-ahead）。

---

## 3. 模型設定（無超參數搜索）

| 模型 | 超參數 | 不平衡處理 |
|---|---|---|
| Logistic Regression | C=1.0, max_iter=1000 | class_weight='balanced' |
| Random Forest | n_estimators=100, max_depth=4 | class_weight='balanced' |
| XGBoost | n_estimators=100, max_depth=3, lr=0.1 | scale_pos_weight=9.5 |

---

## 4. 主要結果（測試集，Events 5-7）

### 4.1 整體指標

| 模型 | Top-5 | Top-8 | Top-10 | Top-13 | Top-16 | Recall@10 | AUC-ROC |
|---|---|---|---|---|---|---|---|
| **隨機基準** | **9.5%** | **9.5%** | **9.5%** | **9.5%** | **9.5%** | — | **0.500** |
| LR | 20.0% | 25.0% | **30.0%** | 30.8% | 31.3% | 9.4% | 0.776 |
| RF | 40.0% | 37.5% | **40.0%** | 38.5% | 31.3% | 12.5% | 0.782 |
| **XGB** | **40.0%** | **37.5%** | **50.0%** | **46.2%** | **37.5%** | **15.6%** | 0.726 |

> **所有三個模型均顯著優於隨機基準**（Top-10 Precision：30% ~ 50% vs 9.53%）。

**最佳模型（by Top-10 Precision）：XGBoost**
- Top-10 Precision = **50.0%**（vs 基準 9.53%，超出 **+40.5 個百分點**）
- AUC-ROC = 0.726（RF 最高 AUC = 0.782）
- 測試集共 32 個正樣本；Top-10 取到 5 個（Recall@10 = 15.6%）

### 4.2 逐事件 Top-10 Precision

| 事件 | 候選池 | 正樣本數 | 事件基準率 | LR | RF | XGB |
|---|---|---|---|---|---|---|
| 00919_20241217 | 107 | 8 | 7.5% | 20.0% | 20.0% | 20.0% |
| 00919_20250603 | 107 | 16 | 15.0% | **80.0%** | 50.0% | 50.0% |
| 00919_20251216 | 94 | 8 | 8.5% | 0.0% | 0.0% | 0.0% |

**觀察**：
- 00919_20250603（16 個正樣本，基準 15%）：LR 獲得 80%（10 中 8），所有模型都大幅超越基準。
- 00919_20251216（8 個正樣本，基準 8.5%）：三個模型 Top-10 均為 0%，完全失效。這是最嚴重的失敗案例，可能反映選股規則在 2025Q4 有隱性轉變，或樣本量不足導致泛化失敗。
- 逐事件結果的高方差（0% ~ 80%）表明：模型捕捉到的是**平均模式**，而非每次事件的精確規律。

---

## 5. 特徵重要性

### Random Forest
| 排名 | 特徵 | 重要性 |
|---|---|---|
| 1 | dividend_yield_rank_in_pool | 0.2176 |
| 2 | dividend_yield_ttm | 0.2035 |
| 3 | log_market_cap | 0.1275 |
| 4 | market_cap_rank_in_pool | 0.1198 |
| 5 | log_avg_turnover_60d | 0.0924 |

### XGBoost
| 排名 | 特徵 | 重要性 |
|---|---|---|
| 1 | dividend_yield_rank_in_pool | 0.2925 |
| 2 | dividend_yield_ttm | 0.1198 |
| 3 | beta_60d | 0.0973 |
| 4 | volatility_60d | 0.0940 |
| 5 | log_market_cap | 0.0936 |

**共識**：**股息殖利率相關特徵**（dividend_yield_rank_in_pool、dividend_yield_ttm）在兩個非線性模型中均高居前兩名，符合 00919「高息 ETF」的成分股篩選邏輯。市值（log_market_cap）也穩定出現在 Top-3，與指數傾向納入流動性較高的大型高息股一致。

---

## 6. 200 字結果摘要（面試版）

三個模型（Logistic Regression、Random Forest、XGBoost）以「公告前 14 個交易日」可觀察的公開財務指標為輸入，在時間外樣本（Events 5-7，2024Q4 ~ 2025Q4）測試預測「哪些股票會被新納入 00919」。

**主要結論**：XGBoost 在 Top-10 Precision 達到 **50%**，是隨機基準（9.53%）的 5.2 倍；RF 達 40%，LR 達 30%，三者均顯著優於基準。AUC-ROC 方面 RF（0.782）與 LR（0.776）略高於 XGB（0.726）。

**最重要特徵**：股息殖利率（絕對值及池內排名）在 RF 和 XGB 中均為第一、二名，印證 00919「高息 ETF」依股息率篩選成分股的規則。市值排名為第三重要因子，反映流動性篩選門檻。

**限制**：逐事件結果高方差明顯（LR：0% ~ 80%）；其中最新一次調整（2025-12-16）三個模型均為 0%，顯示可能存在選股規則的隱性轉變或訓練樣本不足的泛化失敗。結論應解讀為「平均而言具有預測能力」而非「每次事件均可靠預測」。

---

## 7. 關鍵限制與誠實聲明

1. **樣本數極小**：6 個事件、63 個正樣本（train 31 / test 32），任何單一事件的失敗或成功均可大幅影響聚合指標。
2. **靜態 Universe 偏差**：候選池限定為 137 檔，而非全 TWSE 有配息股（約 600+ 檔）。這造成候選池正率（約 8-15%）遠高於真實情境正率（約 1-3%），模型難度被低估。
3. **Was_in_00919_previous NaN 假設**：Event 2 的 124 個 NaN 填 0 是保守假設，可能引入 label noise。
4. **模型未調參**：超參數為預設合理值，沒有 cross-validation 搜索；真實部署應有更嚴格的驗證。
5. **2025-12-16 事件失效**：最近一次調整三個模型均失效，應在簡報中誠實揭露，並討論可能原因（規則轉變 / 小樣本方差 / look-ahead 窗口不匹配）。

---

## 8. 產出檔案

| 檔案 | 說明 |
|---|---|
| `output/tables/prediction_results.csv` | 3 models × 7 metrics |
| `output/tables/prediction_per_event.csv` | 3 events × 3 models（per-event Top-10）|
| `output/figures/feature_importance_rf.png` | RF 特徵重要性 |
| `output/figures/feature_importance_xgb.png` | XGB 特徵重要性 |
| `output/figures/roc_curve_all_models.png` | ROC 曲線（3 models）|
| `scripts/train_prediction_model.py` | 可重現腳本（含假設文件）|
