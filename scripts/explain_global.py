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

