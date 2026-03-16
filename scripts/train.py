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
