# =============================================================================
# HELPERS
# =============================================================================

import os
import textwrap
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

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

