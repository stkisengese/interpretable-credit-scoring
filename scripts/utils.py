# =============================================================================
# HELPERS
# =============================================================================

import os
import textwrap
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from datetime import datetime

RANDOM_STATE    = 42
MODEL_DIR       = os.path.join("results", "model")
INTERP_DIR      = os.path.join("results", "interpretability")

# ── Risk bands (thresholds, colours, labels) ──────────────────────────────────
RISK_BANDS = [
    (0.00, 0.30, "#27ae60", "Low"),
    (0.30, 0.50, "#f39c12", "Medium"),
    (0.50, 0.70, "#e67e22", "High"),
    (0.70, 1.00, "#c0392b", "Very High"),
]

def _print_header(title: str):
    sep = "=" * 65
    print(f"\n{sep}\n{title}\n{sep}")


def _dense_float32(X) -> np.ndarray:
    """Convert sparse or double array to dense float32."""
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# =============================================================================
# TRAIN
# =============================================================================

def _save_fig(fig: plt.Figure, fname: str, directory: str = INTERP_DIR):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


def _stratified_sample(X, y, n):
    """Draw a stratified random subsample of size n."""
    rng  = np.random.default_rng(RANDOM_STATE)
    pos  = np.where(y == 1)[0]
    neg  = np.where(y == 0)[0]
    n_pos = min(int(n * y.mean()), len(pos))
    n_neg = min(n - n_pos, len(neg))
    idx   = np.concatenate([
        rng.choice(pos, n_pos, replace=False),
        rng.choice(neg, n_neg, replace=False),
    ])
    rng.shuffle(idx)
    return X[idx], y[idx]


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def _dense_f32(X) -> np.ndarray:
    """Return a float32 dense array — half the memory of float64."""
    arr = X.toarray() if sp.issparse(X) else np.asarray(X)
    return arr.astype(np.float32, copy=False)


def _auc_str(scores) -> str:
    a = np.asarray(scores)
    return f"{a.mean():.4f} ± {a.std():.4f}"


def save_figure(fig: plt.Figure, filename: str):
    path = os.path.join(MODEL_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# =============================================================================
# PREDICT
# =============================================================================

def _risk_label(p: float) -> str:
    for lo, hi, _, label in RISK_BANDS:
        if lo <= p < hi:
            return label
    return "Very High"


def _risk_color(p: float) -> str:
    for lo, hi, color, _ in RISK_BANDS:
        if lo <= p < hi:
            return color
    return "#c0392b"


def _wrap(text: str, width: int = 70) -> str:
    return "\n".join(textwrap.wrap(text, width))


def _get_gain_importance(model, n_features):
    """Extract normalised gain importance from HGBC internal trees."""
    gains = np.zeros(n_features)
    for iter_preds in model._predictors:
        for predictor in iter_preds:
            nodes = predictor.nodes
            for node in nodes[nodes["is_leaf"] == 0]:
                fi = int(node["feature_idx"])
                if 0 <= fi < n_features:
                    gains[fi] += max(0.0, float(node["gain"]))
    total = gains.sum()
    return gains / total if total > 0 else gains


def _make_title_page(
    client_id: int, prediction: float, y_true: int | None,
    oof_pred: float | None, dataset: str, note: str,
) -> plt.Figure:
    """Plain text title / summary page."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")

    lines = [
        ("CREDIT RISK REPORT", 18, "bold", _risk_color(prediction)),
        ("", 4, "normal", "black"),
        (f"Client ID : {client_id}", 14, "bold", "black"),
        (f"Dataset   : {dataset.upper()}", 11, "normal", "#555"),
        ("", 6, "normal", "black"),
        (f"Predicted Default Probability : {prediction:.4f}  ({prediction:.1%})",
         14, "bold", _risk_color(prediction)),
        (f"Risk Level : {_risk_label(prediction)}", 12, "normal",
         _risk_color(prediction)),
    ]

    if y_true is not None:
        label = "Default" if y_true else "No Default"
        correct = (prediction >= 0.5) == bool(y_true)
        lines += [
            ("", 4, "normal", "black"),
            (f"True Label  : {label}", 12, "normal", "black"),
            (f"OOF Prediction : {oof_pred:.4f}" if oof_pred is not None
             else "OOF Prediction : N/A", 11, "normal", "#555"),
            (f"Model Correct  : {'✓  Yes' if correct else '✗  No'}",
             12, "bold", "green" if correct else "red"),
        ]

    if note:
        lines += [
            ("", 8, "normal", "black"),
            ("── Analysis Note ──", 11, "bold", "#333"),
        ]
        for sub in textwrap.wrap(note, 80):
            lines.append((sub, 10, "normal", "#333"))

    lines += [
        ("", 12, "normal", "black"),
        ("Generated by: skisenge01edukisumu", 9, "normal", "#aaa"),
        (f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 9, "normal", "#aaa"),
        ("", 4, "normal", "black"),
        ("This report contains 5 sections:", 10, "bold", "#333"),
        ("  1. Risk Score Gauge", 9, "normal", "#555"),
        ("  2. SHAP Waterfall Plot (feature contributions)", 9, "normal", "#555"),
        ("  3. Client Feature Profile", 9, "normal", "#555"),
        ("  4. Client vs. Population Comparison", 9, "normal", "#555"),
        ("  5. Top Risk Factors", 9, "normal", "#555"),
    ]

    y_pos = 0.97
    for text, size, weight, color in lines:
        if text == "":
            y_pos -= size / 150
            continue
        ax.text(0.05, y_pos, text, transform=ax.transAxes,
                fontsize=size, fontweight=weight, color=color,
                verticalalignment="top")
        y_pos -= (size + 4) / 150

    return fig


# =============================================================================
# CLIENT SELECTION  (for the 3 required analyses at predict.py)
# =============================================================================

def select_correct_client(art: dict) -> int:
    """
    Find a training client the model predicts correctly with high confidence.
    Prefer a confirmed defaulter scored > 0.6 (correct positive), or a
    confirmed non-defaulter scored < 0.2 (correct negative).
    """
    if art["oof_preds"] is None:
        raise RuntimeError("OOF predictions not found. Run train.py first.")

    ids    = np.asarray(art["train_ids"])
    y      = art["y_train"]
    preds  = art["oof_preds"]

    # Confident correct defaults (TP)
    tp_mask = (y == 1) & (preds >= 0.55)
    if tp_mask.any():
        idx = int(np.where(tp_mask)[0][np.argmax(preds[tp_mask])])
        print(f"  Correct client (TP): {ids[idx]}  score={preds[idx]:.4f}  y={y[idx]}")
        return int(ids[idx])

    # Confident correct non-defaults (TN)
    tn_mask = (y == 0) & (preds <= 0.15)
    if tn_mask.any():
        idx = int(np.where(tn_mask)[0][np.argmin(preds[tn_mask])])
        print(f"  Correct client (TN): {ids[idx]}  score={preds[idx]:.4f}  y={y[idx]}")
        return int(ids[idx])

    raise RuntimeError("Could not find a high-confidence correct prediction.")


def select_wrong_client(art: dict) -> int:
    """
    Find a training client the model misclassifies.
    Prefer a defaulter scored < 0.3 (FN — dangerous miss) or a
    non-defaulter scored > 0.7 (FP — costly false alarm).
    """
    if art["oof_preds"] is None:
        raise RuntimeError("OOF predictions not found. Run train.py first.")

    ids   = np.asarray(art["train_ids"])
    y     = art["y_train"]
    preds = art["oof_preds"]

    # False negatives: defaulter the model missed
    fn_mask = (y == 1) & (preds <= 0.30)
    if fn_mask.any():
        idx = int(np.where(fn_mask)[0][np.argmin(preds[fn_mask])])
        print(f"  Wrong client (FN): {ids[idx]}  score={preds[idx]:.4f}  y={y[idx]}")
        return int(ids[idx])

    # False positives: non-defaulter the model flagged
    fp_mask = (y == 0) & (preds >= 0.70)
    if fp_mask.any():
        idx = int(np.where(fp_mask)[0][np.argmax(preds[fp_mask])])
        print(f"  Wrong client (FP): {ids[idx]}  score={preds[idx]:.4f}  y={y[idx]}")
        return int(ids[idx])

    raise RuntimeError("Could not find a clear misclassification.")


def select_test_client(art: dict) -> int:
    """Return the first test client (or a random one)."""
    ids = np.asarray(art["test_ids"])
    cid = int(ids[0])
    print(f"  Test client: {cid}")
    return cid
