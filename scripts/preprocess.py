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

# --- Custom Transformers ---

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

            

class OwnCarAgeImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values in 'OWN_CAR_AGE' with 0.
    """
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'OWN_CAR_AGE' in X.columns:
            X['OWN_CAR_AGE'] = X['OWN_CAR_AGE'].fillna(0)
        return X

def preprocess_data():
    print("Loading training data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "application_train.csv"), low_memory=False)
    train_df = reduce_mem_usage(train_df)
    
    y_train = train_df['TARGET'].copy()
    train_ids = train_df['SK_ID_CURR'].copy()
    X_train = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
    del train_df
    gc.collect()

    print("Building and fitting full pipeline...")
    # 1. Cleanup components
    cleanup_pipeline = Pipeline([
        ('days_employed_fix', DaysEmployedAnomalyFixer()),
        ('own_car_age', OwnCarAgeImputer()),
        ('time_vars', TimeVariableTransformer()),
        ('income_log', IncomeTransformer())
    ])

    # 2. Imputation and Encoding components
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, min_frequency=0.01))
    ])
    
    # 3. Combine into full pipeline
    # NOTE: We use specific column types to ensure consistency
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_transformer, make_column_selector(dtype_include=['category', 'object']))
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )


    full_pipeline = Pipeline([
        ('cleanup', cleanup_pipeline),
        ('preprocessor', preprocessor)
    ])

    print("Fitting and transforming training data...")
    X_train_processed = full_pipeline.fit_transform(X_train)
    
    # Save processed training data and labels
    joblib.dump(X_train_processed, os.path.join(FEATURE_ENG_DIR, "X_train_processed.joblib"))
    joblib.dump(y_train, os.path.join(FEATURE_ENG_DIR, "y_train.joblib"))
    joblib.dump(train_ids, os.path.join(FEATURE_ENG_DIR, "train_ids.joblib"))
    
    # Save the fitted pipeline
    joblib.dump(full_pipeline, os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))
    
    del X_train, X_train_processed, y_train, train_ids
    gc.collect()

    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "application_test.csv"), low_memory=False)
    test_df = reduce_mem_usage(test_df)
    test_ids = test_df['SK_ID_CURR'].copy()
    X_test = test_df.drop(columns=['SK_ID_CURR'])
    del test_df
    gc.collect()

    print("Transforming test data...")
    X_test_processed = full_pipeline.transform(X_test)
    
    joblib.dump(X_test_processed, os.path.join(FEATURE_ENG_DIR, "X_test_processed.joblib"))
    joblib.dump(test_ids, os.path.join(FEATURE_ENG_DIR, "test_ids.joblib"))
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()
