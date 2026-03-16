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

