# Data Preprocessing Script for Credit Scoring Project
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

# --- Constants ---
DATA_DIR = "data"
RESULTS_DIR = "results"
MODEL_DIR = os.path.join(RESULTS_DIR, "model")
FEATURE_ENG_DIR = os.path.join(RESULTS_DIR, "feature_engineering")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_ENG_DIR, exist_ok=True)

# --- Memory Optimization ---
def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df

# --- Custom Transformers ---

class DaysEmployedAnomalyFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'DAYS_EMPLOYED' in X.columns:
            # Handle both float and int types for the value
            X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
        return X

class TimeVariableTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'DAYS_BIRTH' in X.columns:
            X['AGE_YEARS'] = np.abs(X['DAYS_BIRTH'].astype(float)) / 365.0
        if 'DAYS_EMPLOYED' in X.columns:
            X['YEARS_EMPLOYED'] = np.abs(X['DAYS_EMPLOYED'].astype(float)) / 365.0
        return X

class IncomeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if 'AMT_INCOME_TOTAL' in X.columns:
            X['AMT_INCOME_TOTAL_LOG'] = np.log1p(X['AMT_INCOME_TOTAL'].astype(float))
        return X

class OwnCarAgeImputer(BaseEstimator, TransformerMixin):
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

    print("Initial cleanup on training data...")
    cleanup_pipeline = Pipeline([
        ('days_employed_fix', DaysEmployedAnomalyFixer()),
        ('own_car_age', OwnCarAgeImputer()),
        ('time_vars', TimeVariableTransformer()),
        ('income_log', IncomeTransformer())
    ])
    X_train_clean = cleanup_pipeline.fit_transform(X_train)
    del X_train
    gc.collect()


if __name__ == "__main__":
    preprocess_data()
