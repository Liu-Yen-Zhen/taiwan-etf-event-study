"""
scripts/train_prediction_model.py
----------------------------------
Stage 4：二元分類預測模型
任務：預測哪些股票會被新納入 00919 ETF

資料：prediction_panel.parquet（661 rows, 63 positives = 9.53%）
特徵：10 個（9 個連續 + 1 個二元）
分割：Train = events 2-4（00919_20230531 ~ 00919_20240531）
      Test  = events 5-7（00919_20241217 ~ 00919_20251216）

模型（不做超參數搜索）：
  1. Logistic Regression  (C=1.0, class_weight='balanced')
  2. Random Forest         (n_estimators=100, max_depth=4, class_weight='balanced')
  3. XGBoost               (n_estimators=100, max_depth=3, scale_pos_weight=9.5, lr=0.1)

評估（測試集）：
  - Top-K Precision at K = 5, 8, 10, 13, 16
  - Recall at top-10
  - AUC-ROC
  - Per-event Top-K Precision（3 events × 3 models）
  - Feature Importance（RF + XGB）
  - ROC Curve（3 models）

產出：
  output/tables/prediction_results.csv
  output/tables/prediction_per_event.csv
  output/figures/feature_importance_rf.png
  output/figures/feature_importance_xgb.png
  output/figures/roc_curve_all_models.png
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────
PANEL_PQ      = ROOT / "data"   / "processed" / "prediction_panel.parquet"
OUT_TABLE_DIR = ROOT / "output" / "tables"
OUT_FIG_DIR   = ROOT / "output" / "figures"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
TRAIN_EVENTS = ["00919_20230531", "00919_20231218", "00919_20240531"]
TEST_EVENTS  = ["00919_20241217", "00919_20250603", "00919_20251216"]
RANDOM_BASELINE = 9.53 / 100   # 63/661

# 10 model features:
#   9 continuous (will be StandardScaled)
#   1 binary (was_in_00919_previous — no scaling)
CONT_FEATURES = [
    "dividend_yield_ttm",
    "dividend_yield_rank_in_pool",
    "log_market_cap",
    "market_cap_rank_in_pool",
    "log_avg_turnover_60d",
    "turnover_ratio_60d",
    "volatility_60d",
    "beta_60d",
    "momentum_60d_pre",
]
BIN_FEATURES  = ["was_in_00919_previous"]
ALL_FEATURES  = CONT_FEATURES + BIN_FEATURES


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Loading & Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare() -> pd.DataFrame:
    df = pd.read_parquet(PANEL_PQ)
    print(f"Raw panel: {len(df)} rows, {df['y'].sum()} positives")

    # ── Assumption A: impute was_in_00919_previous NaN → 0 ──────────────
    # Event 2 (00919_20230531) is the first event with real tracking;
    # we have no cumulative record of who was already in 00919 at that point.
    # Setting NaN → 0 (assumes "not seen before" = not previously included)
    # is a *conservative lower bound*: some initial 22 holdings may be mis-
    # labeled as 0, injecting label noise on negatives only (not positives).
    n_nan = df["was_in_00919_previous"].isna().sum()
    df["was_in_00919_previous"] = df["was_in_00919_previous"].fillna(0)
    print(f"Imputed was_in_00919_previous: {n_nan} NaN → 0  (event-2 design gap)")

    # ── Assumption B: listwise-drop 1 row with missing market_cap ────────
    n_before = len(df)
    df = df.dropna(subset=["market_cap"])
    print(f"Dropped {n_before - len(df)} row(s) with missing market_cap")

    return df


def split(df: pd.DataFrame):
    train = df[df["event_id"].isin(TRAIN_EVENTS)].copy()
    test  = df[df["event_id"].isin(TEST_EVENTS)].copy()

    assert len(train) + len(test) == len(df), "Split sanity check failed"

    print(f"\nTrain: {len(train)} rows, {train['y'].sum()} positives "
          f"(events: {TRAIN_EVENTS})")
    print(f"Test:  {len(test)} rows,  {test['y'].sum()} positives "
          f"(events: {TEST_EVENTS})")

    return train, test


def scale_features(train: pd.DataFrame, test: pd.DataFrame):
    """Fit StandardScaler on train continuous features, transform both."""
    scaler = StandardScaler()

    X_train_cont = scaler.fit_transform(train[CONT_FEATURES])
    X_test_cont  = scaler.transform(test[CONT_FEATURES])

    X_train = np.hstack([X_train_cont, train[BIN_FEATURES].values])
    X_test  = np.hstack([X_test_cont,  test[BIN_FEATURES].values])

    y_train = train["y"].values
    y_test  = test["y"].values

    return X_train, y_train, X_test, y_test, scaler


# ═══════════════════════════════════════════════════════════════════════════
# 2. Models
# ═══════════════════════════════════════════════════════════════════════════

def build_models():
    lr = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        scale_pos_weight=9.5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    return {"LR": lr, "RF": rf, "XGB": xgb}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════

def topk_precision(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Precision@K: rank by proba desc, check top-K labels."""
    idx = np.argsort(proba)[::-1][:k]
    return float(y_true[idx].sum()) / k


def recall_at_topk(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    """Recall@K: among all positives, how many are in top-K?"""
    idx = np.argsort(proba)[::-1][:k]
    total_pos = y_true.sum()
    if total_pos == 0:
        return 0.0
    return float(y_true[idx].sum()) / total_pos


def evaluate_model(name: str, model, X_test, y_test,
                   ks=(5, 8, 10, 13, 16)) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, proba)

    row = {"model": name, "auc_roc": round(auc, 4)}
    for k in ks:
        row[f"top{k}_precision"] = round(topk_precision(y_test, proba, k), 4)
    row["recall_at_top10"] = round(recall_at_topk(y_test, proba, 10), 4)

    return row, proba


def per_event_eval(name: str, model, test_df: pd.DataFrame,
                   X_test: np.ndarray, k: int = 10) -> list[dict]:
    """Top-K precision per event (using pooled probabilities)."""
    proba = model.predict_proba(X_test)[:, 1]
    test_df = test_df.copy()
    test_df["_proba"] = proba

    rows = []
    for ev in TEST_EVENTS:
        sub = test_df[test_df["event_id"] == ev]
        rows.append({
            "model":       name,
            "event_id":    ev,
            "pool_size":   len(sub),
            "n_positive":  int(sub["y"].sum()),
            "top_k":       k,
            "top_k_precision": round(
                topk_precision(sub["y"].values, sub["_proba"].values, k), 4
            ),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 4. Plotting
# ═══════════════════════════════════════════════════════════════════════════

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "axes.grid":        True,
    "grid.color":       "#e0e0e0",
    "grid.linestyle":   "--",
    "font.family":      "sans-serif",
}


def plot_feature_importance(model, model_name: str, feature_names: list[str],
                            out_path: Path) -> None:
    imp = model.feature_importances_
    order = np.argsort(imp)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4C72B0" if i < len(CONT_FEATURES) else "#DD8452"
                  for i in order]
        bars = ax.barh([feature_names[i] for i in order], imp[order],
                       color=colors, edgecolor="white", height=0.6)

        ax.set_xlabel("Feature Importance", fontsize=11)
        ax.set_title(f"Feature Importance — {model_name}", fontsize=13, pad=12)

        # legend
        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#4C72B0", label="Continuous (scaled)"),
            Patch(facecolor="#DD8452", label="Binary (was_in_00919_previous)"),
        ]
        ax.legend(handles=legend_els, fontsize=9, loc="lower right")

        # value labels
        for bar, v in zip(bars, imp[order]):
            ax.text(v + imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=8, color="#333333")

        ax.set_xlim(0, imp.max() * 1.18)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")


def plot_roc_curves(models_proba: dict, y_test: np.ndarray,
                    out_path: Path) -> None:
    colors = {"LR": "#4C72B0", "RF": "#55A868", "XGB": "#C44E52"}
    linestyles = {"LR": "-", "RF": "--", "XGB": "-."}

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))

        for name, proba in models_proba.items():
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            ax.plot(fpr, tpr,
                    color=colors[name],
                    linestyle=linestyles[name],
                    lw=2,
                    label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random (AUC = 0.500)")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curves — 00919 Addition Prediction\n(Test: Events 5-7)",
                     fontsize=12, pad=12)
        ax.legend(fontsize=10, loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {out_path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Stage 4: Binary Classification — 00919 Addition Prediction")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────────
    df = load_and_prepare()
    train_df, test_df = split(df)

    X_train, y_train, X_test, y_test, scaler = scale_features(train_df, test_df)
    print(f"\nFeature matrix: {X_train.shape[1]} features "
          f"({len(CONT_FEATURES)} continuous, {len(BIN_FEATURES)} binary)")
    print(f"  Continuous (scaled): {CONT_FEATURES}")
    print(f"  Binary (unscaled):   {BIN_FEATURES}")

    # ── Train ───────────────────────────────────────────────────────────
    models = build_models()
    print("\n[Training models]")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  {name}: done")

    # ── Overall Evaluation ──────────────────────────────────────────────
    print("\n[Overall Evaluation — Test Set]")
    results  = []
    all_proba = {}
    for name, model in models.items():
        row, proba = evaluate_model(name, model, X_test, y_test)
        results.append(row)
        all_proba[name] = proba

        print(f"\n  {name}:")
        print(f"    AUC-ROC          = {row['auc_roc']:.4f}")
        print(f"    Top-5  Precision = {row['top5_precision']:.4f}  "
              f"(baseline {RANDOM_BASELINE:.4f})")
        print(f"    Top-8  Precision = {row['top8_precision']:.4f}")
        print(f"    Top-10 Precision = {row['top10_precision']:.4f}")
        print(f"    Top-13 Precision = {row['top13_precision']:.4f}")
        print(f"    Top-16 Precision = {row['top16_precision']:.4f}")
        print(f"    Recall@Top-10    = {row['recall_at_top10']:.4f}")

    results_df = pd.DataFrame(results)
    out_path = OUT_TABLE_DIR / "prediction_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path.relative_to(ROOT)}")

    # ── Per-Event Evaluation ────────────────────────────────────────────
    print("\n[Per-Event Evaluation — Top-10 Precision]")
    per_event_rows = []
    for name, model in models.items():
        per_event_rows.extend(per_event_eval(name, model, test_df, X_test))

    per_event_df = pd.DataFrame(per_event_rows)
    pivot = per_event_df.pivot_table(
        index=["event_id", "pool_size", "n_positive"],
        columns="model",
        values="top_k_precision"
    )
    print(pivot.to_string())

    out_path = OUT_TABLE_DIR / "prediction_per_event.csv"
    per_event_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path.relative_to(ROOT)}")

    # ── Feature Importance ──────────────────────────────────────────────
    print("\n[Feature Importance Plots]")
    feature_labels = CONT_FEATURES + BIN_FEATURES

    plot_feature_importance(
        models["RF"], "Random Forest", feature_labels,
        OUT_FIG_DIR / "feature_importance_rf.png"
    )
    plot_feature_importance(
        models["XGB"], "XGBoost", feature_labels,
        OUT_FIG_DIR / "feature_importance_xgb.png"
    )

    # ── ROC Curves ─────────────────────────────────────────────────────
    print("\n[ROC Curve Plot]")
    plot_roc_curves(all_proba, y_test, OUT_FIG_DIR / "roc_curve_all_models.png")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULT SUMMARY")
    print("=" * 60)
    print(f"  Random baseline (Top-K Precision) = {RANDOM_BASELINE*100:.2f}%")
    print(f"  Test set: {len(test_df)} rows, {int(y_test.sum())} positives "
          f"across 3 events")
    print()

    best_row = results_df.sort_values("top10_precision", ascending=False).iloc[0]
    print(f"  Best model by Top-10 Precision: {best_row['model']}")
    print(f"    Top-10 Precision  = {best_row['top10_precision']*100:.1f}%  "
          f"(baseline {RANDOM_BASELINE*100:.1f}%)")
    print(f"    Top-10 vs random  = {(best_row['top10_precision']-RANDOM_BASELINE)*100:+.1f} pp")
    print(f"    AUC-ROC           = {best_row['auc_roc']:.3f}")

    print()
    print("  All models vs random baseline (Top-10 Precision):")
    for _, r in results_df.iterrows():
        beat = "✓ beats" if r["top10_precision"] > RANDOM_BASELINE else "✗ below"
        print(f"    {r['model']:4s}: {r['top10_precision']*100:.1f}%  "
              f"({beat} baseline {RANDOM_BASELINE*100:.1f}%)")

    print()
    print("  Feature Importance Top-3 (RF):")
    fi_rf = sorted(zip(feature_labels, models["RF"].feature_importances_),
                   key=lambda x: x[1], reverse=True)
    for fname, fimp in fi_rf[:3]:
        print(f"    {fname}: {fimp:.4f}")

    print()
    print("  Feature Importance Top-3 (XGB):")
    fi_xgb = sorted(zip(feature_labels, models["XGB"].feature_importances_),
                    key=lambda x: x[1], reverse=True)
    for fname, fimp in fi_xgb[:3]:
        print(f"    {fname}: {fimp:.4f}")

    print()
    print("  All outputs:")
    print("    output/tables/prediction_results.csv")
    print("    output/tables/prediction_per_event.csv")
    print("    output/figures/feature_importance_rf.png")
    print("    output/figures/feature_importance_xgb.png")
    print("    output/figures/roc_curve_all_models.png")

    print("\n" + "=" * 60)
    print("  Stage 4 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
