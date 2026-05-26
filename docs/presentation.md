---
marp: true
theme: default
paginate: true
backgroundColor: "#ffffff"
color: "#1a1a2e"
style: |
  section {
    font-family: "Helvetica Neue", "PingFang TC", "Microsoft JhengHei", sans-serif;
    font-size: 22px;
    padding: 40px 60px;
  }
  h1 {
    font-size: 36px;
    color: #1a1a2e;
    border-bottom: 3px solid #4C72B0;
    padding-bottom: 10px;
    margin-bottom: 20px;
  }
  h2 {
    font-size: 28px;
    color: #4C72B0;
    margin-bottom: 16px;
  }
  h3 {
    font-size: 22px;
    color: #555;
  }
  table {
    font-size: 18px;
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
  }
  th {
    background-color: #4C72B0;
    color: white;
    padding: 8px 12px;
    text-align: left;
  }
  td {
    padding: 7px 12px;
    border-bottom: 1px solid #ddd;
  }
  tr:nth-child(even) { background-color: #f8f9fa; }
  .highlight {
    color: #C44E52;
    font-weight: bold;
  }
  .positive { color: #2ca02c; font-weight: bold; }
  .negative { color: #C44E52; font-weight: bold; }
  .note {
    font-size: 16px;
    color: #888;
    margin-top: 10px;
    font-style: italic;
  }
  section.lead {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
  }
  section.lead h1 {
    border-bottom: none;
    font-size: 38px;
    line-height: 1.3;
  }
  .tag {
    display: inline-block;
    background: #4C72B0;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 16px;
    margin-right: 6px;
  }
  .warn {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 8px 14px;
    border-radius: 4px;
    font-size: 18px;
    margin: 10px 0;
  }
  .box {
    background: #f0f4ff;
    border-left: 4px solid #4C72B0;
    padding: 10px 16px;
    border-radius: 4px;
    margin: 10px 0;
  }
---

<!-- _class: lead -->

# 台灣高息 ETF（00919）<br>指數調整的反向效應

**事件研究、預測模型與策略可行性檢驗**

<br>

2026-05-20 ｜ 資料：FinMind（7 個真實調整事件，2022–2025）

---

# Slide 1｜研究動機

## 為什麼選 00919？

| | S&P 500 / 0050 | **00919（本研究）** |
|---|---|---|
| 選股規則 | 委員會主觀決定 | **公開確定（股息率排名 + 市值門檻）** |
| 研究飽和度 | 高（alpha 薄）| **低（2022 年成立）** |
| 對優式業務的對標 | 弱 | **強（ETF 套利核心問題）** |

<br>

<div class="box">

**核心問題**：當選股規則完全透明，「指數納入溢價」還存在嗎？

</div>

<div class="note">基金規模 ~800 億台幣；每次調整約 10–20 檔，影響規模顯著</div>

---

# Slide 2｜資料與方法

## 全流程可重現：真實資料，無合成

- **資料來源**：FinMind API（免費 tier）—— 日線價格、持股數、財務
- **Universe**：0050 ∪ 0056 ∪ 00919 歷史成分 = **137 檔**，候選池命中率 **100%**
- **事件**：7 個調整事件（2022-10-13 至 2025-12-16）
- **異常報酬**：AR = r_stock − r_TAIEX（市場調整模型）
- **訓練 / 測試分割**：Events 2–4（train）/ Events 5–7（test）— 嚴格時間序列切割

<br>

<div class="warn">⚠ 研究過程中主動發現並修正<b>兩次方法論問題</b>（詳見 Slide 7）</div>

---

# Slide 3｜主結果：CAR 反向效應

## 調入股：公告前漲、生效後跌

| 視窗 | Mean CAR | t-stat | p-value |
|---|---|---|---|
| 公告前 30 日 [−30, 0] | **+4.86%** | +5.27 | < 0.001 |
| 公告前 5 日 [−5, 0] | +2.38% | +4.51 | < 0.001 |
| 公告日→生效日 | −0.78% | −1.86 | 0.076 |
| **生效後 30 日 [+1, +31]** | **−7.42%** | **−5.54** | **< 0.001** |

<br>

<div class="box">

與 S&P 500 的「指數納入溢價（inclusion premium）」**方向完全相反**

</div>

<div class="note">N = 71 個調入股事件；AR = r_stock − r_TAIEX；v2 清理後數據</div>

---

# Slide 4｜為什麼跟美股相反

## 四個結構性差異共同解釋反向型態

**① 規則透明 → 預期提前消化**
市場在公告前已知道誰會被納入，買盤提前 → 公告本身資訊衝擊極小

**② 散戶主導 → 情緒動量買盤**
散戶追逐「高息 ETF 概念股」，公告後短線兌現 → 賣壓集中

**③ 高股息因子均值回歸**
「被貼標籤」的高殖利率股短期 dividend yield compression → 後段轉負

**④ 季配結構 → 每季集中追逐**
00919 每季配息，每季形成新一輪「高殖利率名單預測」→ 事件驅動規律穩定

<br>

<div class="note">→ 結果：[−5, 0] 正向、[+1, +31] 反轉，而非美股的納入後持續正報酬</div>

---

# Slide 5｜策略意涵：傳統策略失效

## 「公告日買入、生效日賣出」毛報酬即為負

| 策略 | 毛報酬 | 成本 | **淨報酬** |
|---|---|---|---|
| 傳統：公告日買→生效日賣 | −0.78% | 0.585% | **−1.37%** |
| 含滑點 | −0.78% | 0.985% | **−1.77%** |
| 放空調出股 | 統計不顯著 | — | 不可行 |

<br>

**可獲利的唯一視窗：[−5, 0]（+2.38%，扣費後 ~+1.8%）**

但這需要一個前提：

<div class="box">

**在公告前就知道誰會被納入 → 事前預測能力成為策略的決定性條件**

</div>

---

# Slide 6｜橫斷面分析：哪類調入股反向最強

## CAR_post_30 迴歸（N=62，R²=0.312）

| 特徵 | β | t-stat | p |
|---|---|---|---|
| **dividend_yield** | **−0.022** | **−2.26** | **0.028** |
| **log_avg_turnover** | **−0.189** | **−2.45** | **0.017** |
| **turnover_ratio_60d** | **+0.026** | **+2.15** | **0.036** |
| momentum_pre | +0.070 | +0.86 | 0.39 |

<br>

**解讀**：高殖利率 × 大成交額 × 高相對周轉率 = 散戶買盤集中 → 反向更深

<div class="warn">⚠ 觀察性補充（N=62，未做 OOS 驗證）；放空實務障礙高。<b>不建議作為交易訊號</b></div>

---

# Slide 7｜研究紀律：兩次方法論自我修正

## 主動發現錯誤、記錄過程、修正結論

|  | v1（有問題）| v2（修正後）| 修正原因 |
|---|---|---|---|
| CAR_pre_30 R² | **0.345** | **0.072** | 動能變數視窗重疊 24 天 → 機械式自相關 |
| CAR_post_30 R² | **0.214** | **0.312** | 5347 世界先進 close=0 artifact，Cook's D=0.445 |
| CAR_post_30 主結果 | −8.83% | −7.42% | artifact 清理後 outlier 效應移除 |

<br>

<div class="box">

**第一次**：讓假的 R²=0.345 消失，還原真實的 R²=0.072
**第二次**：artifact 被 outlier 遮蔽的真實訊號浮現，R²=0.214 → **0.312**

兩次方向相反，說明不同性質的錯誤各有不同的修正結果。

</div>

---

# Slide 8｜預測模型結果

## XGBoost Top-10 Precision = 50%（基準 9.53%）

| 模型 | Top-5 | **Top-10** | AUC-ROC |
|---|---|---|---|
| 隨機基準 | 9.5% | 9.5% | 0.500 |
| Logistic Regression | 20% | 30% | 0.776 |
| Random Forest | 40% | 40% | **0.782** |
| **XGBoost** | **40%** | **50%** | 0.726 |

**最重要特徵**（RF + XGB 一致）：`dividend_yield_rank_in_pool` 第一，`dividend_yield_ttm` 第二

<br>

**逐事件 Top-10（XGB）**：

| 事件 | 候選池 | 正樣本 | Top-10 Precision |
|---|---|---|---|
| 2024-12-17 | 107 | 8 | 20% |
| 2025-06-03 | 107 | 16 | 50% |
| 2025-12-16 | 94 | 8 | **0%** |

<div class="note">高方差：0%–50%。最新一次事件完全失效。</div>

---

# Slide 9｜OOS 驗證：alpha 未確認

<!-- _footer: "⚠ HYPOTHETICAL — 僅 3 個 OOS 事件，統計上不顯著" -->

## 誠實結論：現實成本下全部為負

| K | 無成本 | 手續費+稅（58.5 bps）| +滑點 20bps | **Break-even** |
|---|---|---|---|---|
| K=5 | +0.46% | −0.13% | −0.53% | **45.7 bps** |
| K=10 | +0.32% | −0.27% | −0.67% | **31.6 bps** |
| **完美預測** | **+0.15%** | −0.43% | — | **15.4 bps** |
| ← 實際最低成本 → | | **58.5 bps** | | |

**補充驗算（T-5 / T-7 建倉 + market-adjusted）**：OOS 三事件均為負（−1.08% ~ −1.48%）

<div class="warn">

⚠ **OOS 期間 pre-announcement alpha 本身未出現**，而非建倉時點問題。
可能原因：(a) N=3 統計無效 (b) 2024–2025 市場輪動（高息 → AI）削弱 run-up 動能

</div>

---

# Slide 10｜限制與未來方向

## 誠實邊界與三條改善路徑

**已知限制**

| 限制 | 嚴重性 |
|---|---|
| 僅 7 個事件（OOS 3 個），統計功效極低 | ⚠⚠⚠ |
| 靜態 Universe 137 檔（真實池 ~600 檔）| ⚠⚠ |
| 收盤價假設完全成交（忽略市場衝擊）| ⚠⚠ |
| 單一 ETF、單一市場 | ⚠⚠ |

**三條改善路徑**

1. **樣本擴展**：0056（2007–，30+ 事件）+ 00878（2020–，20+ 事件）→ N ≥ 60
2. **更短建倉**：T-5 附近建倉（需即時監控規則），理論上可捕捉 [−5,0] 的 +2.38%
3. **規則反推精確化**：直接解析指數公司公告，取代 ML 近似

<div class="note">「OOS alpha 未確認」≠「alpha 不存在」，而是「當前樣本不足以確認」</div>

---

<!-- _class: lead -->

# 謝謝

<br>

**資料**：`data/processed/prediction_panel.parquet`（完整可重現）

**程式碼**：`scripts/` 目錄下 6 支腳本（含完整 docstring）

**文件**：`docs/findings_stage2–6.md`（含兩次方法論修正紀錄）

<br>

*全部產出來自真實資料，無合成數字，無 look-ahead bias*
