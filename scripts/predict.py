"""
Client Scoring & Local Interpretability
=====================================================
Takes a SK_ID_CURR, scores the client, and produces:
  • Predicted default probability
  • SHAP waterfall / force plot
  • 4-panel visualisation (profile, population comparison, gauge, top factors)
  • Full PDF report

Also auto-generates the three required client analyses:
  results/clients_outputs/client1_correct_train.pdf  — confidently correct on train
  results/clients_outputs/client2_wrong_train.pdf    — misclassified on train
  results/clients_outputs/client_test.pdf            — test-set client

Usage
-----
    python scripts/predict.py --client_id 100002
    python scripts/predict.py --run_all          # generates all 3 client PDFs
"""

from __future__ import annotations

import argparse
import gc
import os
import textwrap
import warnings
from datetime import datetime

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from custom_transformers import (
    DaysEmployedAnomalyFixer,
    OwnCarAgeImputer,
    TimeVariableTransformer,
    IncomeTransformer,
)
from utils import (
    _dense_float32, _risk_label, 
    _get_gain_importance, _make_title_page
)

from build_figures import (
    plot_waterfall, plot_client_profile, plot_population_comparison,
    plot_score_gauge, plot_top_factors,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
FEATURE_ENG_DIR = os.path.join("results", "feature_engineering")
MODEL_DIR       = os.path.join("results", "model")
CLIENTS_DIR     = os.path.join("results", "clients_outputs")
os.makedirs(CLIENTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
N_BG_SAMPLES   = 300    # background samples for local SHAP approximation
# WATERFALL_TOP  = 15     # features shown in waterfall

# Key interpretable features for Panels 1 & 2
KEY_FEATURES = [
    "EXT_SOURCE_MEAN",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "CREDIT_TERM",
    "BUREAU_MAX_OVERDUE_DAYS",
    "INST_PAYMENT_DELAY_MAX",
]


# =============================================================================
# STEP 1 — LOAD ARTEFACTS
# =============================================================================

def load_artifacts() -> dict:
    """
    Load all artefacts produced by preprocess.py and train.py.

    Returns a dict with keys:
      model, pipeline, feature_names,
      X_train, X_test, y_train, train_ids, test_ids,
      oof_preds, raw_train, raw_test
    """
    print("Loading artefacts …")

    model    = joblib.load(os.path.join(MODEL_DIR, "my_own_model.pkl"))
    pipeline = joblib.load(os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))

    X_train  = _dense_float32(
        joblib.load(os.path.join(FEATURE_ENG_DIR, "X_train_processed.joblib"))
    )
    X_test   = _dense_float32(
        joblib.load(os.path.join(FEATURE_ENG_DIR, "X_test_processed.joblib"))
    )
    y_train  = np.asarray(
        joblib.load(os.path.join(FEATURE_ENG_DIR, "y_train.joblib")), dtype=np.int8
    )
    train_ids = joblib.load(os.path.join(FEATURE_ENG_DIR, "train_ids.joblib"))
    test_ids  = joblib.load(os.path.join(FEATURE_ENG_DIR, "test_ids.joblib"))

    # OOF predictions (if train.py has been run)
    oof_preds = None
    oof_path  = os.path.join(MODEL_DIR, "oof_predictions.pkl")
    if os.path.exists(oof_path):
        d = joblib.load(oof_path)
        oof_preds = d.get("oof_preds")

    # Raw (pre-sklearn-pipeline) feature frames (if available)
    raw_train = None
    raw_test  = None
    raw_train_path = os.path.join(FEATURE_ENG_DIR, "train_features_raw.pkl")
    raw_test_path  = os.path.join(FEATURE_ENG_DIR, "test_features_raw.pkl")
    if os.path.exists(raw_train_path):
        raw_train = pd.read_pickle(raw_train_path)
        print(f"  Raw train features loaded: {raw_train.shape}")
    if os.path.exists(raw_test_path):
        raw_test = pd.read_pickle(raw_test_path)
        print(f"  Raw test  features loaded: {raw_test.shape}")

    # Feature names
    try:
        ct    = pipeline.named_steps["preprocessor"]
        names = ct.get_feature_names_out()
        names = np.array(
            [n.split("__", 1)[1] if "__" in n else n for n in names], dtype=object
        )
    except Exception:
        names = np.array([f"feat_{i}" for i in range(X_train.shape[1])], dtype=object)

    print(f"  X_train : {X_train.shape}  X_test : {X_test.shape}")
    print(f"  Features: {len(names)}")

    return dict(
        model=model, pipeline=pipeline, feature_names=names,
        X_train=X_train, X_test=X_test,
        y_train=y_train, train_ids=train_ids, test_ids=test_ids,
        oof_preds=oof_preds, raw_train=raw_train, raw_test=raw_test,
    )

