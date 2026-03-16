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

