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

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _dense_f32(X) -> np.ndarray:
    """Return a float32 dense array — half the memory of float64."""
    arr = X.toarray() if sp.issparse(X) else np.asarray(X)
    return arr.astype(np.float32, copy=False)


def _print_header(title: str):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


def _auc_str(scores) -> str:
    a = np.asarray(scores)
    return f"{a.mean():.4f} ± {a.std():.4f}"


def save_figure(fig: plt.Figure, filename: str):
    path = os.path.join(MODEL_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


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

