"""
scripts/train_models_multi_etf.py
----------------------------------
Multi-ETF Prediction Model Training
  - Version A: 7 numeric features + etf_code one-hot, 6,765 rows
  - Version B: 10 numeric features + etf_code one-hot, ~3,437 rows (listwise drop on market cap)

Models: Logistic Regression, Random Forest, XGBoost
Preprocessing:
  1. was_in_etf_previous NaN → 0 (conservative: no prior holdings)
  2. price-based features: median imputation (fit on train only)
  3. etf_code: one-hot encode (fit on train only, test reindexed)
  4. LR continuous features: StandardScaler (fit on train only)
Evaluation: Top-K Precision (K=5,8,10,15,20), AUC-ROC, per-ETF Top-10
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "output" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PQ = ROOT / "data" / "processed" / "prediction_panel_multi_etf.parquet"

# ── Feature sets ───────────────────────────────────────────────────────────
PRICE_FEATS = [
    "log_avg_turnover_60d",
    "volatility_60d",
    "beta_60d",
    "momentum_60d_pre",
]

FEATS_A_BASE = [
    "dividend_yield_ttm",
    "dividend_yield_rank_in_pool",
    "log_avg_turnover_60d",
    "volatility_60d",
    "beta_60d",
    "momentum_60d_pre",
    "was_in_etf_previous",
]  # + etf_code dummies (added dynamically)

FEATS_B_EXTRA = [
    "log_market_cap",
    "market_cap_rank_in_pool",
    "turnover_ratio_60d",
]

FEATS_B_BASE = FEATS_A_BASE + FEATS_B_EXTRA

K_VALUES     = [5, 8, 10, 15, 20]
TRAIN_YEAR   = 2023   # ≤ TRAIN_YEAR → train; > TRAIN_YEAR → test
BASELINE_POS = 0.0609  # overall pos_rate from panel build


# ── Evaluation helpers ──────────────────────────────────────────────────────
def top_k_precision(df, prob_col, y_col, k):
    """Mean Top-K Precision across events (each event scored independently)."""
    precs = []
    for _, grp in df.groupby("event_id"):
        topk = grp.nlargest(k, prob_col)
        precs.append(topk[y_col].sum() / k)
    return float(np.mean(precs)) if precs else np.nan


# ── Data preparation ────────────────────────────────────────────────────────
def prepare(panel, feat_base, version_name):
    """
    Returns train/test DataFrames with preprocessed features.
    All imputation/encoding fitted on train only.
    """
    df = panel.copy()

    # 1. was_in_etf_previous: NaN → 0
    df["was_in_etf_previous"] = df["was_in_etf_previous"].fillna(0.0)

    # 2. Version B: listwise drop on market cap columns
    if any(c in feat_base for c in FEATS_B_EXTRA):
        before = len(df)
        df = df[df[FEATS_B_EXTRA].notna().all(axis=1)].copy()
        print(f"  [{version_name}] listwise drop: {before:,} → {len(df):,} rows")

    # Train / Test split
    ann_year = pd.to_datetime(df["ann_date"]).dt.year
    train = df[ann_year <= TRAIN_YEAR].copy()
    test  = df[ann_year  > TRAIN_YEAR].copy()

    # 3. Price-feature median imputation (fit on train)
    medians = {}
    for c in PRICE_FEATS:
        if c in feat_base:
            med = train[c].median()
            medians[c] = med
            train[c] = train[c].fillna(med)
            test[c]  = test[c].fillna(med)

    # 4. etf_code one-hot (fit on train)
    etf_train_dummies = pd.get_dummies(train["etf_code"], prefix="etf").astype(float)
    etf_test_dummies  = pd.get_dummies(test["etf_code"],  prefix="etf").astype(float)
    etf_cols = sorted(etf_train_dummies.columns.tolist())
    etf_test_dummies = etf_test_dummies.reindex(columns=etf_cols, fill_value=0.0)

    train = pd.concat([train.reset_index(drop=True),
                       etf_train_dummies.reset_index(drop=True)], axis=1)
    test  = pd.concat([test.reset_index(drop=True),
                       etf_test_dummies.reset_index(drop=True)], axis=1)

    all_feats = feat_base + etf_cols

    # Stats
    n_pos_tr = train["y"].sum()
    n_neg_tr = (train["y"] == 0).sum()
    spw      = n_neg_tr / n_pos_tr

    print(f"  [{version_name}] train: {len(train):,} rows  "
          f"pos={n_pos_tr}  neg={n_neg_tr}  spw={spw:.2f}")
    print(f"  [{version_name}] test:  {len(test):,} rows  "
          f"pos={test['y'].sum()}  neg={(test['y']==0).sum()}  "
          f"pos_rate={test['y'].mean()*100:.2f}%")
    print(f"  [{version_name}] features ({len(all_feats)}): "
          f"{[f for f in all_feats if not f.startswith('etf_')]} + {etf_cols}")

    return train, test, all_feats, spw, medians


# ── Model training & evaluation ─────────────────────────────────────────────
def run_version(panel, feat_base, version_name):
    train, test, all_feats, spw, medians = prepare(panel, feat_base, version_name)

    X_train = train[all_feats].values.astype(float)
    y_train = train["y"].values.astype(int)
    X_test  = test[all_feats].values.astype(float)
    y_test  = test["y"].values.astype(int)

    # Scaled inputs for Logistic Regression (fit on train only)
    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    model_specs = {
        "LR": (
            LogisticRegression(
                class_weight="balanced", max_iter=3000,
                C=0.1, solver="lbfgs", random_state=42
            ),
            X_train_sc, X_test_sc,
        ),
        "RF": (
            RandomForestClassifier(
                n_estimators=500, class_weight="balanced",
                min_samples_leaf=5, random_state=42, n_jobs=-1
            ),
            X_train, X_test,
        ),
        "XGB": (
            xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                scale_pos_weight=spw, subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric="logloss", verbosity=0,
            ),
            X_train, X_test,
        ),
    }

    rows        = []
    importances = {}

    for mname, (model, Xtr, Xte) in model_specs.items():
        print(f"    [{version_name}] Training {mname} …", end=" ", flush=True)
        model.fit(Xtr, y_train)
        prob = model.predict_proba(Xte)[:, 1]
        print("done")

        test_p        = test.copy()
        test_p["prob"] = prob

        row = {"version": version_name, "model": mname}
        row["train_pos"] = int(y_train.sum())
        row["train_neg"] = int((y_train == 0).sum())
        row["test_pos"]  = int(y_test.sum())
        row["AUC"]       = float(roc_auc_score(y_test, prob))

        for k in K_VALUES:
            row[f"Prec@{k}"] = top_k_precision(test_p, "prob", "y", k)

        # Per-ETF Top-10
        for etf in ["0056", "00713", "00919"]:
            sub = test_p[test_p["etf_code"] == etf]
            col = f"Top10_{etf}"
            row[col] = top_k_precision(sub, "prob", "y", 10) if (
                len(sub) > 0 and sub["y"].sum() > 0
            ) else np.nan

        rows.append(row)

        # Feature importance (tree models)
        if hasattr(model, "feature_importances_"):
            importances[mname] = pd.Series(
                model.feature_importances_, index=all_feats
            ).sort_values(ascending=False)

    return pd.DataFrame(rows), importances


# ── Pretty-print helpers ────────────────────────────────────────────────────
def print_comparison(ra, rb):
    metric_cols = (
        ["AUC"]
        + [f"Prec@{k}" for k in K_VALUES]
        + ["Top10_0056", "Top10_00713", "Top10_00919"]
    )

    print("\n" + "=" * 100)
    print("  Model Comparison: Version A vs Version B")
    print(f"  Baseline (random Top-K): ~{BASELINE_POS*100:.2f}%")
    print("=" * 100)

    hdr = f"  {'Model':<6} {'Ver':<3}"
    for c in metric_cols:
        hdr += f"  {c:>11}"
    print(hdr)
    print("  " + "-" * 97)

    for model in ["LR", "RF", "XGB"]:
        for ver, res in [("A", ra), ("B", rb)]:
            r = res[res["model"] == model].iloc[0]
            line = f"  {model:<6} {ver:<3}"
            for c in metric_cols:
                val = r.get(c, np.nan)
                if pd.isna(val):
                    line += f"  {'—':>11}"
                elif c == "AUC":
                    line += f"  {val:>11.4f}"
                else:
                    line += f"  {val*100:>10.1f}%"
            print(line)
        print()


def print_importance(imp_dict, version):
    print(f"\n── Feature Importance（Version {version}）" + "─" * 40)
    for mname, ser in imp_dict.items():
        print(f"\n  {mname}:")
        for feat, val in ser.items():
            bar = "█" * max(1, int(val * 300))
            print(f"    {feat:<35}  {val:.4f}  {bar}")


def print_summary(ra, rb):
    print("\n" + "=" * 65)
    print("  文字摘要")
    print("=" * 65)

    baseline = BASELINE_POS
    print(f"\n  基準（pos_rate ≈ random Top-10 Precision）: {baseline*100:.2f}%")

    # Best model by Top-10
    best_a = ra.loc[ra["Prec@10"].idxmax()]
    best_b = rb.loc[rb["Prec@10"].idxmax()]
    lift_a = (best_a["Prec@10"] - baseline) / baseline * 100
    lift_b = (best_b["Prec@10"] - baseline) / baseline * 100

    print(f"\n  【最佳 Top-10 Precision vs 基準】")
    print(f"    Version A：{best_a['model']}  "
          f"{best_a['Prec@10']*100:.1f}%  "
          f"（vs 基準 +{lift_a:.0f}%）")
    print(f"    Version B：{best_b['model']}  "
          f"{best_b['Prec@10']*100:.1f}%  "
          f"（vs 基準 +{lift_b:.0f}%）")

    print(f"\n  【Version A vs B 差異（Top-10 Precision）】")
    for m in ["LR", "RF", "XGB"]:
        va = ra[ra["model"] == m].iloc[0]["Prec@10"]
        vb = rb[rb["model"] == m].iloc[0]["Prec@10"]
        diff_pp  = (vb - va) * 100
        diff_rel = (vb - va) / va * 100 if va > 0 else np.nan
        sign = "+" if diff_pp >= 0 else ""
        print(f"    {m:<5}: A={va*100:.1f}%  B={vb*100:.1f}%  "
              f"Δ={sign}{diff_pp:.1f}pp ({sign}{diff_rel:.0f}%)")

    print(f"\n  【各 ETF Top-10 Precision（Version A 最佳模型：{best_a['model']}）】")
    r = ra[ra["model"] == best_a["model"]].iloc[0]
    for etf in ["0056", "00713", "00919"]:
        col  = f"Top10_{etf}"
        val  = r.get(col, np.nan)
        diff = (val - baseline) * 100 if not pd.isna(val) else np.nan
        sign = "+" if diff >= 0 else ""
        if not pd.isna(val):
            print(f"    {etf}: {val*100:.1f}%  "
                  f"（基準 {sign}{diff:.1f}pp）")
        else:
            print(f"    {etf}: N/A")

    print(f"\n  【模型難易度 by ETF（Version A，Top-10）】")
    for m in ["LR", "RF", "XGB"]:
        r = ra[ra["model"] == m].iloc[0]
        vals = {etf: r.get(f"Top10_{etf}", np.nan) for etf in ["0056", "00713", "00919"]}
        vals_str = "  ".join(
            f"{etf}={v*100:.1f}%" if not pd.isna(v) else f"{etf}=N/A"
            for etf, v in vals.items()
        )
        print(f"    {m}: {vals_str}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Multi-ETF Prediction Model Training")
    print("=" * 65)

    panel = pd.read_parquet(PANEL_PQ)
    print(f"Panel loaded: {len(panel):,} rows  "
          f"(pos={panel['y'].sum()}, neg={(panel['y']==0).sum()})")

    # ── Version A ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Version A  (7 numeric features, full 6,765 rows)")
    print("─" * 60)
    results_a, imp_a = run_version(panel, list(FEATS_A_BASE), "A")
    results_a.to_csv(OUT_DIR / "multi_etf_prediction_results_vA.csv", index=False)
    print(f"  ✓ Saved → output/tables/multi_etf_prediction_results_vA.csv")

    # ── Version B ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Version B  (10 numeric features, listwise drop ~3,437 rows)")
    print("─" * 60)
    results_b, imp_b = run_version(panel, list(FEATS_B_BASE), "B")
    results_b.to_csv(OUT_DIR / "multi_etf_prediction_results_vB.csv", index=False)
    print(f"  ✓ Saved → output/tables/multi_etf_prediction_results_vB.csv")

    # ── Combined ───────────────────────────────────────────────────────────
    combined = pd.concat([results_a, results_b], ignore_index=True)
    combined.to_csv(OUT_DIR / "multi_etf_prediction_results_combined.csv", index=False)
    print(f"  ✓ Saved → output/tables/multi_etf_prediction_results_combined.csv")

    # ── Feature importance (Version A → RF + XGB) ─────────────────────────
    if imp_a:
        imp_rows = []
        for mname, ser in imp_a.items():
            for feat, val in ser.items():
                imp_rows.append({"version": "A", "model": mname,
                                 "feature": feat, "importance": val})
        imp_df = pd.DataFrame(imp_rows)
        imp_df.to_csv(OUT_DIR / "feature_importance_vA.csv", index=False)
        print(f"  ✓ Saved → output/tables/feature_importance_vA.csv")

    # ── Output ────────────────────────────────────────────────────────────
    print_comparison(results_a, results_b)
    print_importance(imp_a, "A")
    print_summary(results_a, results_b)
    print("\n  完成。等待確認後再進行下一步。\n")


if __name__ == "__main__":
    main()
