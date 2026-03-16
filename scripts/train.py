"""
Model Training & Validation Script
================================================
Implements the full training pipeline for the Home Credit Default Risk project.

Usage
-----
    python scripts/train.py
"""

from __future__ import annotations

import os
import warnings
import time
import gc

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import MaxAbsScaler
from utils import _print_header, _auc_str, _dense_f32, save_figure

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURE_ENG_DIR = os.path.join("results", "feature_engineering")
MODEL_DIR       = os.path.join("results", "model")

for d in (MODEL_DIR,):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS      = 5
TARGET_AUC   = 0.62

# Learning-curve diagnostic uses a capped subsample to stay memory-safe.
# The curve shape is reliable on 50 k rows; there is no need for the full set.
LC_MAX_ROWS   = 50_000
LC_N_SIZES    = 7          # number of training-size checkpoints
LC_MAX_ITER   = 150        # fixed (no early stopping) — fast diagnostic fit

HGBC_PARAMS = dict(
    learning_rate       = 0.05,
    max_iter            = 1_000,      # ceiling; early stopping usually fires at ~250
    max_leaf_nodes      = 63,
    max_depth           = None,
    min_samples_leaf    = 20,
    l2_regularization   = 0.1,
    max_features        = 0.8,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 50,
    scoring             = "roc_auc",
    class_weight        = "balanced",
    random_state        = RANDOM_STATE,
    verbose             = 0,
)


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_data():
    """
    Load preprocessed feature matrices and cast to float32 immediately.
    Halves the resident memory: 307 k × 417 × 4 B ≈ 500 MB vs 1 GB.
    """
    _print_header("Loading preprocessed data")

    X_train = _dense_f32(joblib.load(
        os.path.join(FEATURE_ENG_DIR, "X_train_processed.joblib")))
    X_test  = _dense_f32(joblib.load(
        os.path.join(FEATURE_ENG_DIR, "X_test_processed.joblib")))
    y_train  = np.asarray(
        joblib.load(os.path.join(FEATURE_ENG_DIR, "y_train.joblib")),
        dtype=np.int8)
    test_ids = joblib.load(os.path.join(FEATURE_ENG_DIR, "test_ids.joblib"))

    print(f"  X_train : {X_train.shape}  dtype={X_train.dtype}"
          f"  ({X_train.nbytes/1e9:.2f} GB)")
    print(f"  X_test  : {X_test.shape}  dtype={X_test.dtype}"
          f"  ({X_test.nbytes/1e9:.2f} GB)")
    print(f"  Positives: {y_train.mean():.2%}")
    return X_train, X_test, y_train, test_ids


# ===========================================================================
# STEP 1 — LOGISTIC REGRESSION BASELINE
# ===========================================================================

def train_baseline(X: np.ndarray, y: np.ndarray) -> list[float]:
    """
    5-fold stratified CV baseline with logistic regression.

    """
    _print_header("Baseline — Logistic Regression (5-fold CV)")

    # Scale once; LR is sensitive to feature magnitude
    scaler = MaxAbsScaler()
    X_sc   = scaler.fit_transform(X)
    del scaler; gc.collect()

    # Replace any surviving NaNs (HGBC tolerates them; LR does not)
    col_med = np.nanmedian(X_sc, axis=0)
    nan_mask = np.isnan(X_sc)
    X_sc[nan_mask] = np.take(col_med, np.where(nan_mask)[1])
    del col_med, nan_mask

    lr = LogisticRegression(
        penalty      = "l2",
        C            = 0.1,
        class_weight = "balanced",
        solver       = "saga",
        max_iter     = 300,
        random_state = RANDOM_STATE,
        n_jobs       = 1,          # single-threaded — avoids subprocess copies
    )

    skf  = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr, val) in enumerate(skf.split(X_sc, y), 1):
        lr.fit(X_sc[tr], y[tr])
        prob = lr.predict_proba(X_sc[val])[:, 1]
        auc  = roc_auc_score(y[val], prob)
        aucs.append(auc)
        print(f"  Fold {fold}: AUC = {auc:.4f}")

    del X_sc, lr; gc.collect()
    print(f"\n  Baseline CV AUC: {_auc_str(aucs)}")
    return aucs


# ===========================================================================
# STEP 2 — PRIMARY MODEL
# ===========================================================================

def train_primary_model(X: np.ndarray, y: np.ndarray):
    """
    5-fold stratified CV with HistGradientBoostingClassifier.

    Overfitting prevention
    ----------------------
    • L2 regularisation (l2_regularization=0.1)
    • Column sub-sampling (max_features=0.8)
    • Minimum leaf size (min_samples_leaf=20)
    • Early stopping: patience=50 rounds on a held-out 10 % validation split
    """
    _print_header("Primary model — HistGradientBoostingClassifier (5-fold CV)")

    skf       = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y), dtype=np.float32)
    fold_aucs = []
    models    = []
    iters_per_fold = []

    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        model = HistGradientBoostingClassifier(**HGBC_PARAMS)
        model.fit(X[tr], y[tr])

        prob = model.predict_proba(X[val])[:, 1].astype(np.float32)
        auc  = roc_auc_score(y[val], prob)

        oof_preds[val] = prob
        fold_aucs.append(auc)
        iters_per_fold.append(model.n_iter_)
        models.append(model)

        print(f"  Fold {fold}: AUC = {auc:.4f} | "
              f"n_iter = {model.n_iter_:4d} | {time.time()-t0:.1f}s")

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n  CV AUC (per-fold): {_auc_str(fold_aucs)}")
    print(f"  OOF AUC (global) : {oof_auc:.4f}")
    status = "✓" if oof_auc >= TARGET_AUC else "✗"
    print(f"  {status} OOF AUC {oof_auc:.4f} {'meets' if oof_auc >= TARGET_AUC else 'below'} target ≥ {TARGET_AUC}")

    return oof_preds, fold_aucs, models, iters_per_fold


# ===========================================================================
# STEP 3 — RETRAIN FINAL MODEL
# ===========================================================================

def retrain_final_model(X: np.ndarray, y: np.ndarray,
                        iters_per_fold: list[int]) -> HistGradientBoostingClassifier:
    """
    Retrain on the full training set using the mean early-stopping iteration
    count from CV — prevents overfitting without wasting labelled data.
    """
    _print_header("Retraining final model on full dataset")

    best_n_iter = int(np.mean(iters_per_fold))
    print(f"  Mean early-stopping n_iter across folds: {best_n_iter}")

    # Drop early-stopping keys — they are incompatible with fixed max_iter
    _drop = {"early_stopping", "n_iter_no_change", "validation_fraction", "scoring"}
    final_params = {k: v for k, v in HGBC_PARAMS.items() if k not in _drop}
    final_params.update({"max_iter": best_n_iter, "early_stopping": False})

    model = HistGradientBoostingClassifier(**final_params)
    model.fit(X, y)
    print(f"  Final model trained — n_iter = {model.n_iter_}")
    return model


# ===========================================================================
# STEP 4 — LEARNING CURVES  (memory-safe sequential manual implementation)
# ===========================================================================

def plot_learning_curves(X: np.ndarray, y: np.ndarray):
    """
    Plot training AUC vs validation AUC as training set size grows.

    Memory strategy
    ---------------
    sklearn.model_selection.learning_curve with n_jobs=-1 spawns
    N_SIZES × N_FOLDS processes in parallel, each holding a full copy
    of X_train in RAM (307 k × 417 × 8 B ≈ 1 GB each → up to 41 GB).

    This implementation avoids that in three ways:

    1. Stratified subsample  — the diagnostic needs at most LC_MAX_ROWS rows.
       The curve shape is reliable at this scale; using the full 307 k rows
       would give identical conclusions at 6× the memory cost.

    2. Sequential fitting    — one model is alive at a time.  After scoring,
       the model is deleted and gc.collect() is called before the next fit.

    3. Fixed iteration count — no early stopping in the diagnostic, so no
       internal validation split is held in memory alongside the model.
    """
    _print_header("Learning curves (training vs validation AUC)")

    # ── 1. Stratified subsample ───────────────────────────────────────────────
    n_total = len(y)
    if n_total > LC_MAX_ROWS:
        rng    = np.random.default_rng(RANDOM_STATE)
        # Sample separately from each class to preserve the ~8 % imbalance
        idx0   = np.where(y == 0)[0]
        idx1   = np.where(y == 1)[0]
        k1     = int(LC_MAX_ROWS * y.mean())
        k0     = LC_MAX_ROWS - k1
        chosen = np.concatenate([
            rng.choice(idx0, min(k0, len(idx0)), replace=False),
            rng.choice(idx1, min(k1, len(idx1)), replace=False),
        ])
        rng.shuffle(chosen)
        X_lc, y_lc = X[chosen], y[chosen]
        print(f"  Using stratified subsample: {len(y_lc):,} / {n_total:,} rows"
              f"  (positive rate: {y_lc.mean():.2%})")
    else:
        X_lc, y_lc = X, y
        print(f"  Using full dataset: {n_total:,} rows")

    # ── 2. Build params for diagnostic (fast fixed-iter, no early stopping) ───
    _drop = {"early_stopping", "n_iter_no_change", "validation_fraction", "scoring"}
    lc_params = {k: v for k, v in HGBC_PARAMS.items() if k not in _drop}
    lc_params.update({"max_iter": LC_MAX_ITER, "early_stopping": False})

    # ── 3. Sequential manual learning curve ──────────────────────────────────
    skf         = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    train_fracs = np.linspace(0.15, 1.0, LC_N_SIZES)
    n_lc        = len(X_lc)

    tr_aucs_all  = []
    val_aucs_all = []
    train_sizes  = []

    for frac in train_fracs:
        fold_tr, fold_val = [], []
        size_this = 0

        for tr_idx, val_idx in skf.split(X_lc, y_lc):
            # Apply the training-size fraction *within* the fold's train split
            n_use = max(2, int(len(tr_idx) * frac))
            tr_sub = tr_idx[:n_use]

            model = HistGradientBoostingClassifier(**lc_params)
            model.fit(X_lc[tr_sub], y_lc[tr_sub])

            fold_tr.append(roc_auc_score(y_lc[tr_sub],
                                         model.predict_proba(X_lc[tr_sub])[:, 1]))
            fold_val.append(roc_auc_score(y_lc[val_idx],
                                          model.predict_proba(X_lc[val_idx])[:, 1]))
            size_this = n_use

            # ── Free memory immediately ──────────────────────────────────────
            del model; gc.collect()

        tr_aucs_all.append(fold_tr)
        val_aucs_all.append(fold_val)
        train_sizes.append(size_this)
        print(f"  size={size_this:6,}  train={np.mean(fold_tr):.4f}"
              f"  val={np.mean(fold_val):.4f}")

    # ── Free subsample ────────────────────────────────────────────────────────
    if n_total > LC_MAX_ROWS:
        del X_lc, y_lc; gc.collect()

    # ── Plot ──────────────────────────────────────────────────────────────────
    tr_arr  = np.array(tr_aucs_all)   # shape (n_sizes, n_folds)
    val_arr = np.array(val_aucs_all)

    tr_mean,  tr_std  = tr_arr.mean(1),  tr_arr.std(1)
    val_mean, val_std = val_arr.mean(1), val_arr.std(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, tr_mean  - tr_std,  tr_mean  + tr_std,  alpha=0.15, color="C0")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="C1")
    ax.plot(train_sizes, tr_mean,  "o-", color="C0", label="Training AUC")
    ax.plot(train_sizes, val_mean, "s-", color="C1", label="Validation AUC")
    ax.axhline(TARGET_AUC, ls="--", color="grey", lw=1,
               label=f"Target AUC = {TARGET_AUC}")
    ax.set_xlabel("Training set size (subsample rows)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        "Learning Curves — HistGradientBoostingClassifier\n"
        f"(diagnostic on {min(n_total, LC_MAX_ROWS):,}-row stratified subsample)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "learning_curves.png")

    gap = tr_mean[-1] - val_mean[-1]
    print(f"\n  Largest training size — train AUC : {tr_mean[-1]:.4f} ± {tr_std[-1]:.4f}")
    print(f"  Largest training size — val AUC   : {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
    print(f"  Overfitting gap                   : {gap:.4f}"
          f"  {'(acceptable)' if gap < 0.05 else '(WARNING: gap > 5 pp)'}")


# ===========================================================================
# STEP 5 — EVALUATION PLOTS
# ===========================================================================

def evaluate_oof(y_true: np.ndarray, oof_preds: np.ndarray,
                 fold_aucs: list[float], baseline_aucs: list[float]) -> dict:
    _print_header("Evaluation on OOF predictions")

    oof_auc  = roc_auc_score(y_true, oof_preds)
    avg_prec = average_precision_score(y_true, oof_preds)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, thresh_roc = roc_curve(y_true, oof_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"OOF AUC = {oof_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Out-of-Fold)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    save_figure(fig, "roc_curve.png")

    # ── Precision-Recall curve ────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_true, oof_preds)
    base_rate = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"AP = {avg_prec:.4f}")
    ax.axhline(base_rate, ls="--", color="grey", lw=1,
               label=f"Baseline prevalence = {base_rate:.2%}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Out-of-Fold)")
    ax.legend(); ax.grid(alpha=0.3)
    save_figure(fig, "pr_curve.png")

    # ── Confusion matrix at Youden-optimal threshold ──────────────────────────
    J        = tpr - fpr
    best_idx = int(np.argmax(J))
    best_thr = float(thresh_roc[best_idx])
    y_pred   = (oof_preds >= best_thr).astype(int)
    cm       = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: No Default", "Pred: Default"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True: No Default", "True: Default"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13)
    ax.set_title(f"Confusion Matrix\n"
                 f"Threshold = {best_thr:.3f}  (Youden-optimal)\n"
                 f"TPR = {tp/(tp+fn):.2%}  |  FPR = {fp/(fp+tn):.2%}")
    plt.tight_layout()
    save_figure(fig, "confusion_matrix.png")

    print(f"  OOF AUC            : {oof_auc:.4f}")
    print(f"  Average Precision  : {avg_prec:.4f}")
    print(f"  Youden threshold   : {best_thr:.4f}")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"  Sensitivity (TPR)  : {tp/(tp+fn):.2%}")
    print(f"  Specificity (TNR)  : {tn/(tn+fp):.2%}")
    print(f"  Precision          : {tp/(tp+fp):.2%}")

    _print_header("Threshold choice justification")
    print(
        "  Youden's J (TPR − FPR) is maximised at the selected threshold.\n"
        "  It balances sensitivity and specificity without requiring a\n"
        "  business-specific cost matrix.  In production the bank would\n"
        "  tune the threshold against the relative cost of missed defaults\n"
        "  (credit loss) vs. false denials (lost revenue).\n"
    )

    return dict(oof_auc=oof_auc, avg_prec=avg_prec, threshold=best_thr,
                cm=cm, fold_aucs=fold_aucs, base_aucs=baseline_aucs)


# ===========================================================================
# STEP 6 — KAGGLE SUBMISSION
# ===========================================================================

def make_submission(final_model: HistGradientBoostingClassifier,
                    X_test: np.ndarray, test_ids: pd.Series):
    _print_header("Kaggle submission")
    probs = final_model.predict_proba(X_test)[:, 1]
    sub   = pd.DataFrame({"SK_ID_CURR": test_ids.values, "TARGET": probs})
    path  = os.path.join(MODEL_DIR, "submission.csv")
    sub.to_csv(path, index=False)
    print(f"  Submission saved → {path}")
    print(f"  Shape            : {sub.shape}")
    print(f"  Predicted default rate (test): {probs.mean():.2%}")


# ===========================================================================
# MAIN  —  explicit memory-lifecycle management
# ===========================================================================

def train_model():
    t_start = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, test_ids = load_data()

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_aucs = train_baseline(X_train, y_train)
    gc.collect()

    # ── Primary model CV ─────────────────────────────────────────────────────
    oof_preds, fold_aucs, fold_models, iters_per_fold = train_primary_model(
        X_train, y_train)
    gc.collect()

    # ── Save fold models then FREE them before the memory-heavy steps ─────────
    _print_header("Saving fold models (then freeing RAM)")
    fold_models_path = os.path.join(MODEL_DIR, "fold_models.pkl")
    joblib.dump(fold_models, fold_models_path)
    print(f"  Fold models → {fold_models_path}")
    del fold_models
    gc.collect()
    print("  fold_models deleted from RAM")

    # ── Learning curves  (uses its own subsample; sequential) ─────────────────
    plot_learning_curves(X_train, y_train)
    gc.collect()

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics = evaluate_oof(y_train, oof_preds, fold_aucs, baseline_aucs)

    # ── Retrain final model ───────────────────────────────────────────────────
    final_model = retrain_final_model(X_train, y_train, iters_per_fold)

    # ── Save model artefacts ──────────────────────────────────────────────────
    _print_header("Saving model artefacts")

    model_path = os.path.join(MODEL_DIR, "my_own_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"  Final model  → {model_path}")

    oof_path = os.path.join(MODEL_DIR, "oof_predictions.pkl")
    joblib.dump({"oof_preds": oof_preds, "y_true": y_train}, oof_path)
    print(f"  OOF preds    → {oof_path}")

    metrics_path = os.path.join(MODEL_DIR, "cv_metrics.pkl")
    joblib.dump(metrics, metrics_path)
    print(f"  CV metrics   → {metrics_path}")

    # ── Kaggle submission  (needs X_test) ────────────────────────────────────
    make_submission(final_model, X_test, test_ids)
    # Free X_test — no longer needed after this point
    del X_test, test_ids
    gc.collect()

    # ── Model report ──────────────────────────────────────────────────────────
    # write_model_report(metrics, iters_per_fold, baseline_aucs)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _print_header("Training complete")
    print(f"  Baseline LR CV AUC    : {_auc_str(baseline_aucs)}")
    print(f"  Primary model OOF AUC : {metrics['oof_auc']:.4f}")
    print(f"  Total wall time       : {elapsed/60:.1f} min")
    print(f"\nArtefacts in {MODEL_DIR}/")
    for fname in ("my_own_model.pkl", "submission.csv", "model_report.txt",
                  "roc_curve.png", "pr_curve.png",
                  "confusion_matrix.png", "learning_curves.png"):
        print(f"  {fname}")


if __name__ == "__main__":
    train_model()
