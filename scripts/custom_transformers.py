# ===========================================================================
# SKLEARN PIPELINE CLASSES
# ===========================================================================

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
