import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import (
    _risk_color, _wrap
)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
N_BG_SAMPLES   = 300    # background samples for local SHAP approximation
WATERFALL_TOP  = 15     # features shown in waterfall

# =============================================================================
# PANEL A — WATERFALL / FORCE PLOT
# =============================================================================

def plot_waterfall(
    contributions: np.ndarray, feature_names: np.ndarray,
    base_value: float, prediction: float,
    top_n: int = WATERFALL_TOP,
) -> plt.Figure:
    """
    Matplotlib waterfall chart showing feature contributions.

    Starting from base_value (mean prediction), each bar shows how much
    a feature pushes the score up (red) or down (blue).
    Final value equals the model's prediction for this client.
    """
    n = min(len(contributions), len(feature_names))
    c = contributions[:n]
    f = feature_names[:n]

    # Select top_n by absolute value; merge the rest into 'Other features'
    abs_c   = np.abs(c)
    top_idx = np.argsort(abs_c)[-top_n:][::-1]      # descending
    other   = c.sum() - c[top_idx].sum()

    # Build ordered display items  (most impactful first)
    top_idx_sorted = sorted(top_idx, key=lambda i: abs(c[i]), reverse=True)
    display_c    = list(c[top_idx_sorted]) + [other]
    display_f    = list(f[top_idx_sorted]) + ["Other features"]

    # Running cumulative starts
    starts = []
    cursor = base_value
    for val in display_c:
        starts.append(cursor)
        cursor += val

    colors = ["#e74c3c" if v >= 0 else "#3498db" for v in display_c]

    fig, ax = plt.subplots(figsize=(9, max(7, len(display_c) * 0.5)))

    bars = ax.barh(
        range(len(display_c)), display_c, left=starts,
        color=colors, edgecolor="white", lw=0.5, height=0.6
    )

    # Value labels inside / beside bars
    for i, (s, v) in enumerate(zip(starts, display_c)):
        lbl = f"+{v:+.4f}" if v >= 0 else f"{v:.4f}"
        mid = s + v / 2
        ax.text(mid, i, lbl, ha="center", va="center",
                fontsize=7, color="white" if abs(v) > 0.005 else "black")

    # Vertical reference lines
    ax.axvline(prediction, color="black", lw=2, ls="-",
               label=f"Prediction: {prediction:.4f}")
    ax.axvline(base_value, color="#7f8c8d", lw=1.5, ls="--",
               label=f"Base value (mean): {base_value:.4f}")

    ax.set_yticks(range(len(display_f)))
    ax.set_yticklabels(display_f, fontsize=9)
    ax.set_xlabel("Predicted Probability of Default")
    ax.set_title("SHAP Waterfall — Feature Contributions to Score\n"
                 "Red = raises default risk  |  Blue = lowers default risk")

    legend_els = [
        mpatches.Patch(color="#e74c3c", label="Raises risk"),
        mpatches.Patch(color="#3498db", label="Lowers risk"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# PANEL B — CLIENT PROFILE  (Panel 1 in spec)
# =============================================================================

def plot_client_profile(
    raw_row: pd.Series | None, key_feats: list[str],
    prediction: float, client_id: int, y_true: int | None,
) -> plt.Figure:
    """
    Horizontal bar chart of the client's key feature values (normalised).
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    if raw_row is None:
        ax.text(0.5, 0.5, "Raw features not available.\nRun preprocess.py first.",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title(f"Client {client_id} — Profile")
        return fig

    present = [f for f in key_feats if f in raw_row.index and pd.notna(raw_row[f])]
    if not present:
        ax.text(0.5, 0.5, "None of the key features found in raw row.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    vals  = np.array([float(raw_row[f]) for f in present])
    color = _risk_color(prediction)

    bars = ax.barh(present, vals, color=color, alpha=0.75, edgecolor="white", lw=0.5)
    ax.set_xlabel("Feature Value (raw)")

    title_parts = [f"Client {client_id} — Profile  |  Score: {prediction:.1%}"]
    if y_true is not None:
        title_parts.append(f"  True label: {'Default' if y_true else 'No Default'}")
    ax.set_title("".join(title_parts))

    # Annotate bars with value
    for bar, val in zip(bars, vals):
        ax.text(val, bar.get_y() + bar.get_height() / 2,
                f"  {val:.3g}", va="center", fontsize=8)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# PANEL C — CLIENT VS POPULATION  (Panel 2 in spec)
# =============================================================================

def plot_population_comparison(
    raw_row: pd.Series | None, raw_train: pd.DataFrame | None,
    key_feats: list[str], client_id: int, prediction: float,
) -> plt.Figure:
    """
    Violin plots showing where the client sits in the training population
    distribution for each key feature.  Client's value overlaid as a marker.
    """
    if raw_row is None or raw_train is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5,
                "Raw features unavailable.\nRun preprocess.py first.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"Client {client_id} — vs. Population")
        return fig

    present = [f for f in key_feats
               if f in raw_row.index and f in raw_train.columns
               and pd.notna(raw_row[f])]
    if not present:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No matching key features found.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    n_feat = len(present)
    fig, axes = plt.subplots(1, n_feat, figsize=(max(9, 2.5 * n_feat), 5))
    if n_feat == 1:
        axes = [axes]

    # Use a small subsample of the population to keep plot light
    pop_sample = raw_train.sample(min(5_000, len(raw_train)), random_state=RANDOM_STATE)

    for ax, feat in zip(axes, present):
        pop_vals = pop_sample[feat].dropna().values

        # Clip extreme outliers for readable plots
        lo, hi = np.nanpercentile(pop_vals, [1, 99])
        pop_clip = np.clip(pop_vals, lo, hi)

        ax.violinplot(pop_clip, positions=[0], showmedians=True, widths=0.7)

        client_val = float(raw_row[feat])
        ax.scatter([0], [np.clip(client_val, lo, hi)],
                   color=_risk_color(prediction), s=80, zorder=5,
                   label=f"Client: {client_val:.3g}")
        ax.set_xticks([])
        ax.set_title(_wrap(feat, 18), fontsize=8, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Client {client_id} vs. Population Distribution  "
        f"(Score: {prediction:.1%})",
        fontsize=10,
    )
    plt.tight_layout()
    return fig
