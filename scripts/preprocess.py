"""
Data Preprocessing and Feature Engineering Script
==================================================
Preprocessing Pipeline and Feature Engineering
for the Home Credit Default Risk project.

Feature Engineering covers:
  - Application table ratio features
  - Bureau + Bureau Balance aggregations
  - Installment payment aggregations
  - Credit card balance aggregations
  - POS/Cash balance aggregations
  - Previous application aggregations

Usage:
    python scripts/preprocess.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "data"
RESULTS_DIR = "results"
MODEL_DIR = os.path.join(RESULTS_DIR, "model")
FEATURE_ENG_DIR = os.path.join(RESULTS_DIR, "feature_engineering")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_ENG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Memory utility
# ---------------------------------------------------------------------------
def reduce_mem_usage(df):
    """
    Iterate through all columns and downcast numeric dtypes to reduce memory.
    String columns are cast to 'category'.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(
        f"  Memory reduced: {start_mem:.1f} MB → {end_mem:.1f} MB"
        f" ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
    )
    return df


# ===========================================================================
# ISSUE #4 — FEATURE ENGINEERING FUNCTIONS
# ===========================================================================


def engineer_application_features(df):
    """
    Create ratio and composite features from the main application table.

    Features created
    ----------------
    CREDIT_INCOME_RATIO        : AMT_CREDIT / AMT_INCOME_TOTAL
    ANNUITY_INCOME_RATIO       : AMT_ANNUITY / AMT_INCOME_TOTAL  (affordability)
    CREDIT_TERM                : AMT_CREDIT / AMT_ANNUITY  (approx. loan term in months)
    GOODS_CREDIT_RATIO         : AMT_GOODS_PRICE / AMT_CREDIT  (collateral coverage)
    AGE_YEARS                  : abs(DAYS_BIRTH) / 365
    YEARS_EMPLOYED             : abs(DAYS_EMPLOYED) / 365  (after anomaly fix)
    EMPLOYMENT_RATIO           : DAYS_EMPLOYED / DAYS_BIRTH  (employment vs. age)
    EXT_SOURCE_MEAN            : mean of EXT_SOURCE_1/2/3
    EXT_SOURCE_MIN             : min of EXT_SOURCE_1/2/3  (worst external score)
    EXT_SOURCE_PROD            : product of non-null EXT_SOURCE values
    ADDRESS_MISMATCH_SCORE     : sum of 6 address-mismatch binary flags
    ENQUIRY_RECENT             : sum of credit bureau enquiries (hour → month)
    SOCIAL_CIRCLE_DEFAULT_RATE_30 : DEF_30 / OBS_30 in applicant's social circle
    DOCUMENT_COUNT             : total number of FLAG_DOCUMENT_* columns set to 1
    """
    df = df.copy()

    # ── Fix DAYS_EMPLOYED anomaly (365243 codes 'not employed / unknown') ──
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # ── Ratio features ──────────────────────────────────────────────────────
    eps = 1e-9  # avoid division by zero

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + eps)
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + eps)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + eps)

    # ── Time variables ───────────────────────────────────────────────────────
    df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"].astype(float)) / 365.0
    df["YEARS_EMPLOYED"] = np.abs(df["DAYS_EMPLOYED"].astype(float)) / 365.0  # NaN preserved
    # EMPLOYMENT_RATIO: ratio of employment span to life span (negative values normal)
    df["EMPLOYMENT_RATIO"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + eps)

    # ── External credit scores ───────────────────────────────────────────────
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available_ext = [c for c in ext_cols if c in df.columns]
    if available_ext:
        df["EXT_SOURCE_MEAN"] = df[available_ext].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[available_ext].min(axis=1)
        # Product only over non-null values; NaN if all are null
        df["EXT_SOURCE_PROD"] = df[available_ext].apply(
            lambda row: row.dropna().prod() if not row.isna().all() else np.nan,
            axis=1,
        )

    # ── Address mismatch score (6 binary indicator flags) ───────────────────
    address_flags = [
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ]
    present_flags = [c for c in address_flags if c in df.columns]
    df["ADDRESS_MISMATCH_SCORE"] = df[present_flags].sum(axis=1)

    # ── Recent credit bureau enquiries (HOUR + DAY + WEEK + MON) ────────────
    enquiry_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
    ]
    present_enquiry = [c for c in enquiry_cols if c in df.columns]
    df["ENQUIRY_RECENT"] = df[present_enquiry].sum(axis=1)

    # ── Social circle default rate (30 DPD) ──────────────────────────────────
    if (
        "DEF_30_CNT_SOCIAL_CIRCLE" in df.columns
        and "OBS_30_CNT_SOCIAL_CIRCLE" in df.columns
    ):
        df["SOCIAL_CIRCLE_DEFAULT_RATE_30"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / (
            df["OBS_30_CNT_SOCIAL_CIRCLE"] + eps
        )

    # ── Document count ────────────────────────────────────────────────────────
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)

    return df


def aggregate_bureau_features(bureau_df, bureau_balance_df=None):
    """
    Aggregate bureau and bureau_balance data per SK_ID_CURR.

    Bureau features
    ---------------
    BUREAU_LOAN_COUNT              : total number of past bureau credits
    BUREAU_ACTIVE_COUNT            : number of currently active credits
    BUREAU_CLOSED_COUNT            : number of closed credits
    BUREAU_MAX_OVERDUE_DAYS        : worst overdue days recorded at application
    BUREAU_AMT_CREDIT_MAX_OVERDUE_SUM/MAX : aggregates of max overdue amount
    BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM/MAX : aggregates of current overdue amount
    BUREAU_DEBT_RATIO              : total debt / total credit (bureau)
    BUREAU_CREDIT_TYPE_COUNT       : diversity of credit types in history
    BUREAU_DAYS_CREDIT_MAX         : days since most recent bureau credit opened

    Bureau Balance features
    -----------------------
    BB_DELINQUENT_MONTHS_TOTAL  : total months with any DPD (STATUS 1-5)
    BB_SEVERE_MONTHS_TOTAL      : total months with STATUS = 5 (120+ DPD)
    BB_MAX_DPD_EVER             : worst DPD bucket ever recorded
    BB_DELINQUENT_PROP_MEAN     : mean proportion of delinquent months per loan
    """
    print("  Aggregating bureau features...")

    # ── Bureau-level ─────────────────────────────────────────────────────────
    bur = bureau_df.copy()
    bur["IS_ACTIVE"] = (bur["CREDIT_ACTIVE"] == "Active").astype(int)
    bur["IS_CLOSED"] = (bur["CREDIT_ACTIVE"] == "Closed").astype(int)

    bureau_agg = (
        bur.groupby("SK_ID_CURR")
        .agg(
            BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
            BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
            BUREAU_CLOSED_COUNT=("IS_CLOSED", "sum"),
            BUREAU_MAX_OVERDUE_DAYS=("CREDIT_DAY_OVERDUE", "max"),
            BUREAU_AMT_CREDIT_MAX_OVERDUE_SUM=("AMT_CREDIT_MAX_OVERDUE", "sum"),
            BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
            BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
            BUREAU_AMT_CREDIT_SUM_OVERDUE_MAX=("AMT_CREDIT_SUM_OVERDUE", "max"),
            BUREAU_CREDIT_SUM_TOTAL=("AMT_CREDIT_SUM", "sum"),
            BUREAU_CREDIT_SUM_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT", "sum"),
            BUREAU_CREDIT_TYPE_COUNT=("CREDIT_TYPE", "nunique"),
            BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),
        )
        .reset_index()
    )

    eps = 1e-9
    bureau_agg["BUREAU_DEBT_RATIO"] = bureau_agg["BUREAU_CREDIT_SUM_DEBT_TOTAL"] / (
        bureau_agg["BUREAU_CREDIT_SUM_TOTAL"] + eps
    )

    # ── Bureau Balance ────────────────────────────────────────────────────────
    if bureau_balance_df is not None and len(bureau_balance_df) > 0:
        print("  Aggregating bureau balance features...")
        bb = bureau_balance_df.copy()

        # Map STATUS to numeric DPD bucket
        # '0'=current, '1'-'5'=months late, 'C'=closed, 'X'=unknown
        status_map = {
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
            "C": 0, "X": np.nan,
        }
        bb["STATUS_NUMERIC"] = bb["STATUS"].map(status_map)
        bb["IS_DELINQUENT"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)
        bb["IS_SEVERE"] = (bb["STATUS"] == "5").astype(int)

        # Aggregate per bureau loan
        bb_per_loan = (
            bb.groupby("SK_ID_BUREAU")
            .agg(
                BB_DELINQUENT_MONTHS=("IS_DELINQUENT", "sum"),
                BB_SEVERE_MONTHS=("IS_SEVERE", "sum"),
                BB_MAX_DPD=("STATUS_NUMERIC", "max"),
                BB_TOTAL_MONTHS=("MONTHS_BALANCE", "count"),
            )
            .reset_index()
        )
        bb_per_loan["BB_DELINQUENT_PROP"] = bb_per_loan["BB_DELINQUENT_MONTHS"] / (
            bb_per_loan["BB_TOTAL_MONTHS"] + eps
        )

        # Map SK_ID_BUREAU → SK_ID_CURR via bureau index
        id_map = bureau_df[["SK_ID_CURR", "SK_ID_BUREAU"]].copy()
        bb_with_curr = id_map.merge(bb_per_loan, on="SK_ID_BUREAU", how="left")

        bb_per_curr = (
            bb_with_curr.groupby("SK_ID_CURR")
            .agg(
                BB_DELINQUENT_MONTHS_TOTAL=("BB_DELINQUENT_MONTHS", "sum"),
                BB_SEVERE_MONTHS_TOTAL=("BB_SEVERE_MONTHS", "sum"),
                BB_MAX_DPD_EVER=("BB_MAX_DPD", "max"),
                BB_DELINQUENT_PROP_MEAN=("BB_DELINQUENT_PROP", "mean"),
            )
            .reset_index()
        )

        bureau_agg = bureau_agg.merge(bb_per_curr, on="SK_ID_CURR", how="left")

    return bureau_agg


def aggregate_installment_features(installments_df):
    """
    Aggregate installment payment data per SK_ID_CURR.

    Features
    --------
    INST_PAYMENT_DELAY_MEAN     : mean days between due and actual payment
                                  (positive = late, negative = early)
    INST_PAYMENT_DELAY_MAX      : worst single payment delay
    INST_PAYMENT_RATIO_MEAN     : mean(AMT_PAYMENT / AMT_INSTALMENT)
    INST_PAYMENT_RATIO_MIN      : min payment ratio (worst single installment)
    INST_LATE_PAYMENT_COUNT     : total number of late payments
    INST_MISSED_PAYMENT_COUNT   : installments with no DAYS_ENTRY_PAYMENT recorded
    INST_RECENT_DELAY_TREND     : mean delay (last 12 months) − mean delay (older)
                                  positive trend = getting worse
    """
    print("  Aggregating installment payment features...")
    inst = installments_df.copy()
    eps = 1e-9

    inst["PAYMENT_DELAY"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + eps)
    inst["IS_LATE"] = (inst["PAYMENT_DELAY"] > 0).astype(int)
    inst["IS_MISSED"] = inst["DAYS_ENTRY_PAYMENT"].isna().astype(int)

    # Trend: last 12 months vs historical
    # DAYS_INSTALMENT is negative (days before application date)
    recent_mask = inst["DAYS_INSTALMENT"] > -365
    recent_delay = (
        inst[recent_mask]
        .groupby("SK_ID_CURR")["PAYMENT_DELAY"]
        .mean()
        .rename("INST_RECENT_DELAY_MEAN")
    )
    hist_delay = (
        inst[~recent_mask]
        .groupby("SK_ID_CURR")["PAYMENT_DELAY"]
        .mean()
        .rename("INST_HIST_DELAY_MEAN")
    )

    inst_agg = (
        inst.groupby("SK_ID_CURR")
        .agg(
            INST_PAYMENT_DELAY_MEAN=("PAYMENT_DELAY", "mean"),
            INST_PAYMENT_DELAY_MAX=("PAYMENT_DELAY", "max"),
            INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
            INST_PAYMENT_RATIO_MIN=("PAYMENT_RATIO", "min"),
            INST_LATE_PAYMENT_COUNT=("IS_LATE", "sum"),
            INST_MISSED_PAYMENT_COUNT=("IS_MISSED", "sum"),
        )
        .reset_index()
    )

    # Merge trend components
    inst_agg = inst_agg.merge(recent_delay, on="SK_ID_CURR", how="left")
    inst_agg = inst_agg.merge(hist_delay, on="SK_ID_CURR", how="left")
    inst_agg["INST_RECENT_DELAY_TREND"] = (
        inst_agg["INST_RECENT_DELAY_MEAN"] - inst_agg["INST_HIST_DELAY_MEAN"]
    )
    # Drop intermediate columns
    inst_agg.drop(
        columns=["INST_RECENT_DELAY_MEAN", "INST_HIST_DELAY_MEAN"],
        inplace=True,
        errors="ignore",
    )

    return inst_agg


def aggregate_credit_card_features(cc_balance_df):
    """
    Aggregate credit card balance data per SK_ID_CURR.

    Features
    --------
    CC_UTILISATION_MEAN     : mean monthly utilisation (AMT_BALANCE / CREDIT_LIMIT)
    CC_UTILISATION_MAX      : peak monthly utilisation
    CC_ATM_DRAW_RATIO       : mean(ATM draws / total draws) — cash stress indicator
    CC_MIN_PAYMENT_RATIO    : mean(payment / minimum required) — underpayment signal
    CC_DPD_MAX              : worst SK_DPD across all months
    CC_DPD_MEAN             : average SK_DPD
    """
    print("  Aggregating credit card balance features...")
    cc = cc_balance_df.copy()
    eps = 1e-9

    cc["UTILISATION"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + eps)
    cc["ATM_DRAW_RATIO"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / (
        cc["AMT_DRAWINGS_CURRENT"] + eps
    )
    cc["MIN_PAYMENT_RATIO"] = cc["AMT_PAYMENT_CURRENT"] / (
        cc["AMT_INST_MIN_REGULARITY"] + eps
    )

    cc_agg = (
        cc.groupby("SK_ID_CURR")
        .agg(
            CC_UTILISATION_MEAN=("UTILISATION", "mean"),
            CC_UTILISATION_MAX=("UTILISATION", "max"),
            CC_ATM_DRAW_RATIO=("ATM_DRAW_RATIO", "mean"),
            CC_MIN_PAYMENT_RATIO=("MIN_PAYMENT_RATIO", "mean"),
            CC_DPD_MAX=("SK_DPD", "max"),
            CC_DPD_MEAN=("SK_DPD", "mean"),
        )
        .reset_index()
    )

    return cc_agg


def aggregate_pos_cash_features(pos_cash_df):
    """
    Aggregate POS/cash balance data per SK_ID_CURR.

    Features
    --------
    POS_DPD_COUNT                   : number of months with SK_DPD > 0
    POS_DPD_MAX                     : worst SK_DPD across all months
    POS_INSTALMENT_REMAINING_MEAN   : mean(CNT_INSTALMENT_FUTURE / CNT_INSTALMENT)
                                      — how far through the loan on average
    """
    print("  Aggregating POS/cash balance features...")
    pos = pos_cash_df.copy()
    eps = 1e-9

    pos["IS_DPD"] = (pos["SK_DPD"] > 0).astype(int)
    pos["INSTALMENT_REMAINING_PROP"] = pos["CNT_INSTALMENT_FUTURE"] / (
        pos["CNT_INSTALMENT"] + eps
    )

    pos_agg = (
        pos.groupby("SK_ID_CURR")
        .agg(
            POS_DPD_COUNT=("IS_DPD", "sum"),
            POS_DPD_MAX=("SK_DPD", "max"),
            POS_INSTALMENT_REMAINING_MEAN=("INSTALMENT_REMAINING_PROP", "mean"),
        )
        .reset_index()
    )

    return pos_agg


def aggregate_previous_application_features(prev_df):
    """
    Aggregate previous Home Credit application data per SK_ID_CURR.

    Features
    --------
    PREV_TOTAL_COUNT                     : total previous applications
    PREV_APPROVED_COUNT                  : number of approved applications
    PREV_REFUSED_COUNT                   : number of refused applications
    PREV_CANCELLED_COUNT                 : number of cancelled applications
    PREV_APPROVAL_RATE                   : approved / total
    PREV_AMT_CREDIT_APPLICATION_GAP_MEAN : mean (AMT_CREDIT − AMT_APPLICATION)
                                           — lender haircut signal
    PREV_PRODUCT_TYPE_COUNT              : distinct product types applied for
    PREV_DAYS_DECISION_MAX               : days since most recent previous application
    """
    print("  Aggregating previous application features...")
    prev = prev_df.copy()
    eps = 1e-9

    # Lender haircut: how much less was approved vs requested
    prev["AMT_CREDIT_APPLICATION_GAP"] = prev["AMT_CREDIT"] - prev["AMT_APPLICATION"]

    prev["IS_APPROVED"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["IS_REFUSED"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    prev["IS_CANCELLED"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(int)

    prev_agg = (
        prev.groupby("SK_ID_CURR")
        .agg(
            PREV_TOTAL_COUNT=("SK_ID_PREV", "count"),
            PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
            PREV_REFUSED_COUNT=("IS_REFUSED", "sum"),
            PREV_CANCELLED_COUNT=("IS_CANCELLED", "sum"),
            PREV_AMT_CREDIT_APPLICATION_GAP_MEAN=("AMT_CREDIT_APPLICATION_GAP", "mean"),
            PREV_PRODUCT_TYPE_COUNT=("NAME_CONTRACT_TYPE", "nunique"),
            PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),
        )
        .reset_index()
    )

    prev_agg["PREV_APPROVAL_RATE"] = prev_agg["PREV_APPROVED_COUNT"] / (
        prev_agg["PREV_TOTAL_COUNT"] + eps
    )

    return prev_agg


def build_feature_matrix(app_df, bureau_agg=None, inst_agg=None,
                          cc_agg=None, pos_agg=None, prev_agg=None):
    """
    Left-join all aggregated feature tables onto the application table.
    All applicants are preserved; missing joins produce NaN (handled by imputer).
    No duplicates are introduced because each aggregation is keyed on SK_ID_CURR.
    """
    print("  Merging feature tables into main matrix...")
    df = app_df.copy()
    n_start = df.shape[1]

    for agg_df, name in [
        (bureau_agg, "bureau + bureau_balance"),
        (inst_agg, "installments_payments"),
        (cc_agg, "credit_card_balance"),
        (pos_agg, "POS_CASH_balance"),
        (prev_agg, "previous_application"),
    ]:
        if agg_df is not None:
            df = df.merge(agg_df, on="SK_ID_CURR", how="left")
            print(
                f"    [{name}] → +{df.shape[1] - n_start} cols, "
                f"total shape: {df.shape}"
            )
            n_start = df.shape[1]

    return df


def save_feature_descriptions(df, exclude_cols=("SK_ID_CURR", "TARGET")):
    """Save a CSV documenting every feature name, dtype and missingness."""
    rows = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        rows.append(
            {
                "feature": col,
                "dtype": str(df[col].dtype),
                "pct_missing": round(df[col].isna().mean() * 100, 2),
                "example_values": str(df[col].dropna().iloc[:3].tolist())
                if len(df[col].dropna()) > 0
                else "all_nan",
            }
        )
    out = pd.DataFrame(rows)
    path = os.path.join(FEATURE_ENG_DIR, "feature_descriptions.csv")
    out.to_csv(path, index=False)
    print(f"  Feature descriptions saved → {path}  ({len(out)} features)")
    return out


# ===========================================================================
# SKLEARN PIPELINE CLASSES
# ===========================================================================


class DaysEmployedAnomalyFixer(BaseEstimator, TransformerMixin):
    """Replace the 365243 anomaly code in DAYS_EMPLOYED with NaN."""

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if 'DAYS_EMPLOYED' in X.columns:
            X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        return X


class OwnCarAgeImputer(BaseEstimator, TransformerMixin):
    """Impute OWN_CAR_AGE NaN with 0 — missing means no car."""

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if 'OWN_CAR_AGE' in X.columns:
            X['OWN_CAR_AGE'] = X['OWN_CAR_AGE'].fillna(0)
        return X


class TimeVariableTransformer(BaseEstimator, TransformerMixin):
    """Convert DAYS_BIRTH / DAYS_EMPLOYED to positive years (if not yet done)."""

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if 'DAYS_BIRTH' in X.columns and 'AGE_YEARS' not in X.columns:
            X['AGE_YEARS'] = np.abs(X['DAYS_BIRTH'].astype(float)) / 365.0
        if 'DAYS_EMPLOYED' in X.columns and 'YEARS_EMPLOYED' not in X.columns:
            X['YEARS_EMPLOYED'] = np.abs(X['DAYS_EMPLOYED'].astype(float)) / 365.0
        return X


class IncomeTransformer(BaseEstimator, TransformerMixin):
    """Log1p-transform AMT_INCOME_TOTAL to reduce right skew."""

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = X.copy()
        if 'AMT_INCOME_TOTAL' in X.columns:
            X['AMT_INCOME_TOTAL_LOG'] = np.log1p(
                X['AMT_INCOME_TOTAL'].astype(float)
            )
        return X


def build_sklearn_pipeline():
    """
    Build the full sklearn preprocessing pipeline:
      1. Cleanup  (anomaly fix, impute OWN_CAR_AGE, time vars, log income)
      2. ColumnTransformer
           numeric  → median imputation (with missing indicator)
           categoric → constant imputation + one-hot encoding
    """
    cleanup_pipeline = Pipeline([
        ('days_employed_fix', DaysEmployedAnomalyFixer()),
        ('own_car_age', OwnCarAgeImputer()),
        ('time_vars', TimeVariableTransformer()),
        ('income_log', IncomeTransformer())
    ])

    numeric_transformer = Pipeline(
        [('imputer', SimpleImputer(strategy='median', add_indicator=True))]
    )

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_transformer, make_column_selector(dtype_include=['category', 'object']))
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )

    return Pipeline([('cleanup', cleanup_pipeline), ('preprocessor', preprocessor)])


# ===========================================================================
# MAIN ORCHESTRATION
# ===========================================================================


def preprocess_data():
    """
    Full end-to-end preprocessing and feature engineering pipeline.

    Steps
    -----
    1. Load application train + test
    2. Engineer application-level ratio features  (Issue #4)
    3. Aggregate supplementary tables             (Issue #4)
    4. Merge into a unified feature matrix        (Issue #4)
    5. Fit sklearn pipeline on train, transform both (Issue #3)
    6. Save all artefacts
    """
    sep = "=" * 65

    # ── Step 1: Load application data ────────────────────────────────────────
    print(f"\n{sep}\nSTEP 1 — Loading application data\n{sep}")

    train_df = pd.read_csv(
        os.path.join(DATA_DIR, "application_train.csv"), low_memory=False
    )
    print("Train application loaded:")
    train_df = reduce_mem_usage(train_df)

    y_train = train_df["TARGET"].copy()
    train_ids = train_df["SK_ID_CURR"].copy()

    test_df = pd.read_csv(
        os.path.join(DATA_DIR, "application_test.csv"), low_memory=False
    )
    print("Test application loaded:")
    test_df = reduce_mem_usage(test_df)
    test_ids = test_df["SK_ID_CURR"].copy()

    # Combine train + test for consistent feature engineering (no target leakage)
    test_df["TARGET"] = np.nan
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    del train_df, test_df
    gc.collect()
    print(f"Combined shape: {combined_df.shape}")

    # ── Step 2: Application ratio features ───────────────────────────────────
    print(f"\n{sep}\nSTEP 2 — Engineering application-level features\n{sep}")
    combined_df = engineer_application_features(combined_df)
    print(f"After application feature engineering: {combined_df.shape}")

    # ── Step 3: Aggregate supplementary tables ────────────────────────────────
    print(f"\n{sep}\nSTEP 3 — Aggregating supplementary tables\n{sep}")

    bureau_agg = inst_agg = cc_agg = pos_agg = prev_agg = None

    # Bureau + Bureau Balance
    try:
        bureau_df = pd.read_csv(
            os.path.join(DATA_DIR, "bureau.csv"), low_memory=False
        )
        bureau_df = reduce_mem_usage(bureau_df)
        bb_df = pd.read_csv(
            os.path.join(DATA_DIR, "bureau_balance.csv"), low_memory=False
        )
        bb_df = reduce_mem_usage(bb_df)
        bureau_agg = aggregate_bureau_features(bureau_df, bb_df)
        del bureau_df, bb_df
        gc.collect()
        print(f"  Bureau agg shape: {bureau_agg.shape}")
    except FileNotFoundError as exc:
        print(f"  [SKIP] bureau: {exc}")

    # Installment Payments
    try:
        inst_df = pd.read_csv(
            os.path.join(DATA_DIR, "installments_payments.csv"), low_memory=False
        )
        inst_df = reduce_mem_usage(inst_df)
        inst_agg = aggregate_installment_features(inst_df)
        del inst_df
        gc.collect()
        print(f"  Installments agg shape: {inst_agg.shape}")
    except FileNotFoundError as exc:
        print(f"  [SKIP] installments: {exc}")

    # Credit Card Balance
    try:
        cc_df = pd.read_csv(
            os.path.join(DATA_DIR, "credit_card_balance.csv"), low_memory=False
        )
        cc_df = reduce_mem_usage(cc_df)
        cc_agg = aggregate_credit_card_features(cc_df)
        del cc_df
        gc.collect()
        print(f"  Credit card agg shape: {cc_agg.shape}")
    except FileNotFoundError as exc:
        print(f"  [SKIP] credit_card_balance: {exc}")

    # POS/Cash Balance
    try:
        pos_df = pd.read_csv(
            os.path.join(DATA_DIR, "POS_CASH_balance.csv"), low_memory=False
        )
        pos_df = reduce_mem_usage(pos_df)
        pos_agg = aggregate_pos_cash_features(pos_df)
        del pos_df
        gc.collect()
        print(f"  POS/cash agg shape: {pos_agg.shape}")
    except FileNotFoundError as exc:
        print(f"  [SKIP] POS_CASH_balance: {exc}")

    # Previous Applications
    try:
        prev_df = pd.read_csv(
            os.path.join(DATA_DIR, "previous_application.csv"), low_memory=False
        )
        prev_df = reduce_mem_usage(prev_df)
        prev_agg = aggregate_previous_application_features(prev_df)
        del prev_df
        gc.collect()
        print(f"  Previous application agg shape: {prev_agg.shape}")
    except FileNotFoundError as exc:
        print(f"  [SKIP] previous_application: {exc}")

    # ── Step 4: Build merged feature matrix ──────────────────────────────────
    print(f"\n{sep}\nSTEP 4 — Building merged feature matrix\n{sep}")
    combined_df = build_feature_matrix(
        combined_df,
        bureau_agg=bureau_agg,
        inst_agg=inst_agg,
        cc_agg=cc_agg,
        pos_agg=pos_agg,
        prev_agg=prev_agg,
    )
  
    gc.collect()

    # ── Step 5: Fit sklearn pipeline on train; transform both ─────────────────
    print(f"\n{sep}\nSTEP 5 — Applying sklearn pipeline (imputation + encoding)\n{sep}")
    pipeline = build_sklearn_pipeline()

    print("  Fitting + transforming training data …")
    X_train_processed = pipeline.fit_transform(train_full)

    print("  Transforming test data …")
    X_test_processed = pipeline.transform(test_full)

    del train_full, test_full
    gc.collect()

    # ── Step 6: Save artefacts ────────────────────────────────────────────────
    print(f"\n{sep}\nSTEP 6 — Saving artefacts\n{sep}")

    joblib.dump(X_train_processed, os.path.join(FEATURE_ENG_DIR, "X_train_processed.joblib"))
    joblib.dump(X_test_processed, os.path.join(FEATURE_ENG_DIR, "X_test_processed.joblib"))
    joblib.dump(y_train, os.path.join(FEATURE_ENG_DIR, "y_train.joblib"))
    joblib.dump(train_ids, os.path.join(FEATURE_ENG_DIR, "train_ids.joblib"))
    joblib.dump(test_ids, os.path.join(FEATURE_ENG_DIR, "test_ids.joblib"))
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))

    print(f"  X_train_processed : {X_train_processed.shape}")
    print(f"  X_test_processed  : {X_test_processed.shape}")
    print(f"\nAll artefacts saved to '{FEATURE_ENG_DIR}' and '{MODEL_DIR}'.")
    print("\n✓  Preprocessing + Feature Engineering complete.")


if __name__ == "__main__":
    preprocess_data()
