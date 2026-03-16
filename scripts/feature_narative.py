# =============================================================================
#  FEATURE NARRATIVE + REGULATORY CHECK
# =============================================================================

import os
import numpy as np
from utils import _print_header, INTERP_DIR

_CREDIT_DESCRIPTIONS = {
    # Application features
    "EXT_SOURCE_MEAN":          "Mean of three external credit bureau scores "
                                "(strong negative predictor: higher scores = safer applicants)",
    "EXT_SOURCE_MIN":           "Minimum external score — captures the worst single bureau rating",
    "EXT_SOURCE_PROD":          "Product of external scores — amplifies weakness when any score is low",
    "CREDIT_INCOME_RATIO":      "Credit amount / annual income — affordability ratio",
    "ANNUITY_INCOME_RATIO":     "Annuity payment / annual income — monthly debt burden",
    "CREDIT_TERM":              "Estimated loan term in months (credit / annuity)",
    "AGE_YEARS":                "Applicant age in years (older applicants tend to have lower default rates)",
    "YEARS_EMPLOYED":           "Years in current employment (job stability indicator)",
    "EMPLOYMENT_RATIO":         "Employment duration relative to age",
    "GOODS_CREDIT_RATIO":       "Goods price / credit amount — collateral coverage",
    # Bureau features
    "BUREAU_MAX_OVERDUE_DAYS":  "Worst overdue days recorded in credit bureau history",
    "BUREAU_DEBT_RATIO":        "Bureau debt-to-credit ratio",
    "BUREAU_ACTIVE_COUNT":      "Number of currently active bureau credits",
    # Installment features
    "INST_PAYMENT_DELAY_MAX":   "Worst single installment payment delay at previous loans",
    "INST_PAYMENT_DELAY_MEAN":  "Average payment delay across all previous installments",
    "INST_PAYMENT_RATIO_MIN":   "Minimum payment ratio (worst single underpayment)",
    # Credit card
    "CC_UTILISATION_MEAN":      "Average credit card utilisation rate",
    # Previous applications
    "PREV_APPROVAL_RATE":       "Fraction of previous Home Credit applications that were approved",
    "PREV_REFUSED_COUNT":       "Number of previous applications refused",
    "DOCUMENT_COUNT":           "Number of documents submitted with the application",
    "ADDRESS_MISMATCH_SCORE":   "Number of address inconsistencies across registration / work / home",
}

_PROTECTED_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "religion", "age",
    "disability", "national", "marital", "code_gender",
]

# Features that are legitimate business features despite containing 'age'
_AGE_WHITELIST = {"age_years", "years_employed"}


def write_feature_narrative(shap_vals, feature_names, top_n=10):
    """Plain-language top-10 narrative + regulatory check."""
    _print_header("Feature narrative + regulatory check")

    n_feat  = min(len(feature_names), shap_vals.shape[1])
    abs_imp = np.abs(shap_vals[:, :n_feat]).mean(axis=0)
    mean_sv = shap_vals[:, :n_feat].mean(axis=0)
    top_idx = np.argsort(abs_imp)[-top_n:][::-1]

    flagged = []
    lines   = [
        "HOME CREDIT DEFAULT RISK — GLOBAL FEATURE IMPORTANCE NARRATIVE",
        "=" * 65,
        "",
        "Analysis method: SHAP TreeExplainer",
        f"Top {top_n} features ranked by mean absolute impact on predicted",
        "default probability (higher = more influential).",
        "",
    ]

    for rank, fi in enumerate(top_idx, 1):
        if fi >= len(feature_names):
            continue
        fname     = feature_names[fi]
        impact    = float(abs_imp[fi])
        direction = "↑ increases" if mean_sv[fi] > 0 else "↓ decreases"
        desc      = _CREDIT_DESCRIPTIONS.get(fname, "(no description — engineered/OHE feature)")

        # Regulatory check
        fname_l = fname.lower()
        is_flag = (
            any(kw in fname_l for kw in _PROTECTED_KEYWORDS)
            and fname_l not in _AGE_WHITELIST
        )
        if is_flag:
            flagged.append(fname)

        lines += [
            f"{rank:2d}. {fname}",
            f"    Avg impact : {impact:.5f}",
            f"    Direction  : {direction} default risk",
            f"    Meaning    : {desc}",
        ]
        if is_flag:
            lines.append(
                f"    ⚠️  REGULATORY FLAG: feature name suggests possible "
                f"protected attribute."
            )
        lines.append("")

    # Regulatory summary
    lines += [
        "=" * 65,
        "REGULATORY / FAIRNESS SUMMARY",
        "=" * 65,
    ]
    if flagged:
        lines += [
            f"⚠️  {len(flagged)} potentially protected feature(s) in top {top_n}:",
        ] + [f"   • {f}" for f in flagged] + [
            "",
            "   Action required: conduct a full disparate-impact analysis",
            "   under applicable fair-lending regulations before deployment.",
        ]
    else:
        lines += [
            f"✓ No obviously discriminatory features detected in top {top_n}.",
            "",
            "  Important caveat: proxy discrimination is still possible.",
            "  Features such as CREDIT_INCOME_RATIO or EMPLOYMENT_RATIO may",
            "  correlate with protected classes even though they appear",
            "  facially neutral.  A full fairness / disparate-impact audit",
            "  (e.g., using AIF360 or Fairlearn) should be conducted before",
            "  production deployment.",
        ]

    narrative = "\n".join(lines)
    path      = os.path.join(INTERP_DIR, "feature_narrative.txt")
    with open(path, "w") as fh:
        fh.write(narrative)
    print(f"  Feature narrative saved → {path}")
    print()
    print(narrative)
