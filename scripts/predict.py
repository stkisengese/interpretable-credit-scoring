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


# =============================================================================
# STEP 2 — SCORE A SINGLE CLIENT
# =============================================================================

def get_client_data(client_id: int, art: dict):
    """
    Retrieve processed features and raw metadata for a specific client.

    Returns
    -------
    X_client    : (1, n_features) float32  — processed features for prediction
    raw_row     : pd.Series or None        — raw interpretable feature values
    y_true      : int or None              — ground truth (None for test set)
    oof_pred    : float or None            — OOF prediction (None for test set)
    dataset     : 'train' or 'test'
    row_idx     : int                      — row position in X_train / X_test
    """
    train_ids = np.asarray(art["train_ids"])
    test_ids  = np.asarray(art["test_ids"])

    # Locate client
    in_train = client_id in train_ids
    in_test  = client_id in test_ids

    if in_train:
        row_idx  = int(np.where(train_ids == client_id)[0][0])
        X_client = art["X_train"][row_idx : row_idx + 1]
        y_true   = int(art["y_train"][row_idx])
        oof_pred = float(art["oof_preds"][row_idx]) if art["oof_preds"] is not None else None
        raw_row  = (
            art["raw_train"][art["raw_train"]["SK_ID_CURR"] == client_id].iloc[0]
            if art["raw_train"] is not None else None
        )
        return X_client, raw_row, y_true, oof_pred, "train", row_idx

    elif in_test:
        row_idx  = int(np.where(test_ids == client_id)[0][0])
        X_client = art["X_test"][row_idx : row_idx + 1]
        raw_row  = (
            art["raw_test"][art["raw_test"]["SK_ID_CURR"] == client_id].iloc[0]
            if art["raw_test"] is not None else None
        )
        return X_client, raw_row, None, None, "test", row_idx

    else:
        raise ValueError(
            f"Client {client_id} not found in train or test set.\n"
            f"Train has {len(train_ids):,} clients, Test has {len(test_ids):,} clients."
        )


# =============================================================================
# STEP 3 — LOCAL EXPLANATION  (SHAP or fast approximation)
# =============================================================================

def compute_local_explanation(
    model, X_client: np.ndarray, X_background: np.ndarray,
    feature_names: np.ndarray, gain_importance: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Compute per-feature contributions for one client.

    Returns
    -------
    contributions : (n_features,) array of signed contributions
    base_value    : expected prediction (background mean)
    prediction    : model's prediction for this client
    """
    prediction = float(model.predict_proba(X_client)[:, 1][0])

    # Background subsample
    n_bg  = min(N_BG_SAMPLES, len(X_background))
    rng   = np.random.default_rng(RANDOM_STATE)
    bg    = X_background[rng.choice(len(X_background), n_bg, replace=False)]
    base_value = float(model.predict_proba(bg)[:, 1].mean())

    # ── SHAP (TreeExplainer) ──────────────────────────────────────────────
    explainer = shap.TreeExplainer(model, bg)
    sv = explainer.shap_values(X_client)
    if isinstance(sv, list):
        sv = sv[1]
    bv = explainer.expected_value
    if isinstance(bv, (list, np.ndarray)):
        bv = float(bv[1])
    return sv[0].astype(np.float32), float(bv), prediction


# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

def generate_client_pdf(
    client_id: int, art: dict, save_path: str, analysis_note: str = ""
):
    """
    Full pipeline: score client → explain → 4 panels → PDF.

    Parameters
    ----------
    client_id     : SK_ID_CURR of the client
    art           : artefacts dict from load_artifacts()
    save_path     : path for the output PDF
    analysis_note : free-text note appended to the report (e.g., error analysis)
    """
    print(f"\n--- Generating report for client {client_id} ---")

    # ── Score ─────────────────────────────────────────────────────────────────
    X_client, raw_row, y_true, oof_pred, dataset, row_idx = get_client_data(
        client_id, art
    )
    prediction = float(art["model"].predict_proba(X_client)[:, 1][0])
    print(f"  Dataset  : {dataset}")
    print(f"  Prediction: {prediction:.4f}  ({_risk_label(prediction)} risk)")
    if y_true is not None:
        label_str = "Default" if y_true else "No Default"
        correct   = (prediction >= 0.5) == bool(y_true)
        print(f"  True label: {label_str}  |  Correct: {'✓' if correct else '✗'}")

    # ── Local explanation ─────────────────────────────────────────────────────
    gain_imp = _get_gain_importance(art["model"], X_client.shape[1])
    contribs, base_val, _ = compute_local_explanation(
        art["model"], X_client, art["X_train"],
        art["feature_names"], gain_imp,
    )

    # ── Resolve key features ──────────────────────────────────────────────────
    raw_feats = KEY_FEATURES
    if raw_row is not None:
        raw_feats = [f for f in KEY_FEATURES if f in raw_row.index]

    # ── Build figures ─────────────────────────────────────────────────────────
    fig_title   = _make_title_page(client_id, prediction, y_true, oof_pred,
                                   dataset, analysis_note)
    fig_wtfall  = plot_waterfall(contribs, art["feature_names"], base_val, prediction)
    fig_profile = plot_client_profile(raw_row, raw_feats, prediction,
                                      client_id, y_true)
    fig_popcomp = plot_population_comparison(raw_row, art["raw_train"],
                                             raw_feats, client_id, prediction)
    fig_gauge   = plot_score_gauge(prediction, client_id)
    fig_factors = plot_top_factors(contribs, art["feature_names"],
                                   client_id, prediction)

    # ── Write PDF ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with PdfPages(save_path) as pdf:
        for fig in (fig_title, fig_gauge, fig_wtfall,
                    fig_profile, fig_popcomp, fig_factors):
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = f"Credit Score Report — Client {client_id}"
        d["Author"]  = "skisenge01edukisumu"
        d["Subject"] = "Interpretable Credit Scoring"
        d["CreationDate"] = datetime.now()

    print(f"  PDF saved → {save_path}")
    gc.collect()

