"""
Global Model Interpretability
===========================================
Produces global feature-importance analysis for the trained
HistGradientBoostingClassifier using gain-based importance (extracted from
the internal tree nodes) and SHAP / permutation-importance-based explanations.

Outputs (saved to results/interpretability/)
--------------------------------------------
  feature_importance_gain.png     — gain-based bar chart (top 30)
  shap_beeswarm.png               — SHAP beeswarm (or approx. beeswarm)
  shap_bar.png                    — mean |SHAP| bar chart
  shap_dependence_1_<name>.png    — dependence plot for rank #1 feature
  ...
  shap_dependence_5_<name>.png    — dependence plot for rank #5 feature
  feature_narrative.txt           — plain-English narrative + regulatory check

Usage
-----
    python scripts/explain_global.py
"""

from __future__ import annotations

import gc
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.sparse as sp

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from custom_transformers import (
    DaysEmployedAnomalyFixer,
    OwnCarAgeImputer,
    TimeVariableTransformer,
    IncomeTransformer,
)
from utils import (
    _print_header, _save_fig, _dense_float32, _stratified_sample
)

warnings.filterwarnings("ignore")

# ── SHAP (optional) ──────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_ENG_DIR = os.path.join("results", "feature_engineering")
MODEL_DIR       = os.path.join("results", "model")
INTERP_DIR      = os.path.join("results", "interpretability")
CLIENTS_DIR     = os.path.join("results", "clients_outputs")

for d in (INTERP_DIR, CLIENTS_DIR):
    os.makedirs(d, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
N_SHAP_SAMPLES  = 2_000   # rows used for global SHAP computation
N_PERM_SAMPLES  = 3_000   # rows used for permutation importance (fallback)
N_PERM_REPEATS  = 5       # repeats for permutation importance


# =============================================================================
# STEP 1 — LOAD ARTEFACTS
# =============================================================================

def load_artifacts():
    _print_header("Loading artefacts")

    model    = joblib.load(os.path.join(MODEL_DIR, "my_own_model.pkl"))
    pipeline = joblib.load(os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))
    X_train  = _dense_float32(
        joblib.load(os.path.join(FEATURE_ENG_DIR, "X_train_processed.joblib"))
    )
    y_train  = np.asarray(
        joblib.load(os.path.join(FEATURE_ENG_DIR, "y_train.joblib")),
        dtype=np.int8,
    )

    mem_gb = X_train.nbytes / 1024**3
    print(f"  X_train : {X_train.shape}  ({mem_gb:.2f} GB float32)")
    print(f"  Positives: {y_train.mean():.2%}")
    print(f"  SHAP library: {'available ✓' if SHAP_AVAILABLE else 'NOT installed — using permutation importance'}")

    return model, pipeline, X_train, y_train


# =============================================================================
# STEP 2 — FEATURE NAMES
# =============================================================================

def get_feature_names(pipeline, n_features: int) -> np.ndarray:
    """
    Extract human-readable feature names from the fitted sklearn pipeline.

    The ColumnTransformer produces names like 'num__AMT_CREDIT' for numeric
    features and 'cat__NAME_CONTRACT_TYPE_Cash loans' for OHE categories.
    We strip the prefix so plots are readable.
    """
    try:
        ct    = pipeline.named_steps["preprocessor"]
        names = ct.get_feature_names_out()
        # Strip 'num__' / 'cat__' / 'remainder__' prefixes
        cleaned = []
        for n in names:
            cleaned.append(n.split("__", 1)[1] if "__" in n else n)
        print(f"  Feature names extracted: {len(cleaned)}")
        return np.array(cleaned, dtype=object)
    except Exception as exc:
        print(f"  WARNING: could not extract feature names ({exc})")
        return np.array([f"feat_{i}" for i in range(n_features)], dtype=object)


# =============================================================================
# STEP 3 — GAIN-BASED IMPORTANCE
# =============================================================================

def hgbc_gain_importance(model, n_features: int) -> np.ndarray:
    """
    Extract gain-based feature importance from HGBC internal tree nodes.

    HistGradientBoostingClassifier does not expose feature_importances_
    as a top-level attribute, but the split-gain for each internal node is
    stored in model._predictors[i][0].nodes['gain'].  We sum gains across
    all trees and all split nodes, then normalise to [0, 1].
    """
    gains = np.zeros(n_features, dtype=np.float64)
    for iter_predictors in model._predictors:
        for predictor in iter_predictors:
            nodes = predictor.nodes
            split_nodes = nodes[nodes["is_leaf"] == 0]
            for node in split_nodes:
                fi = int(node["feature_idx"])
                if 0 <= fi < n_features:
                    gains[fi] += max(0.0, float(node["gain"]))
    total = gains.sum()
    return gains / total if total > 0 else gains


def plot_builtin_importance(model, feature_names: np.ndarray, top_n: int = 30):
    """Bar chart of gain-based importance (top_n features)."""
    _print_header("Gain-based feature importance")

    n = min(model.n_iter_ and 1, 1)  # sanity
    importances = hgbc_gain_importance(model, len(feature_names))

    k     = min(top_n, len(importances))
    idx   = np.argsort(importances)[-k:]
    vals  = importances[idx]
    names = feature_names[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, k * 0.32)))
    palette = plt.cm.Blues(np.linspace(0.4, 0.9, k))
    ax.barh(range(k), vals, color=palette, edgecolor="white", lw=0.4)
    ax.set_yticks(range(k))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Relative gain importance (normalised)")
    ax.set_title(f"Top {k} Features — Gain-Based Importance\n"
                 "HistGradientBoostingClassifier")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, "feature_importance_gain.png")

    return importances

