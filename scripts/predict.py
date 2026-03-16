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
