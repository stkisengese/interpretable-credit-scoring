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

    # Identify numeric and categorical features
    numeric_features = X_train_clean.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_train_clean.select_dtypes(include=['category', 'object']).columns.tolist()

    print("Fitting preprocessor on training data...")
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median', add_indicator=True))])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, min_frequency=0.01))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )
    
    X_train_processed = preprocessor.fit_transform(X_train_clean)
    del X_train_clean
    gc.collect()

    print("Saving processed training data and pipeline...")
    joblib.dump(X_train_processed, os.path.join(FEATURE_ENG_DIR, "X_train_processed.joblib"))
    joblib.dump(y_train, os.path.join(FEATURE_ENG_DIR, "y_train.joblib"))
    joblib.dump(train_ids, os.path.join(FEATURE_ENG_DIR, "train_ids.joblib"))
    del X_train_processed, y_train, train_ids
    gc.collect()

    # Define full pipeline
    full_pipeline = Pipeline([
        ('cleanup', cleanup_pipeline),
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
                ('cat', categorical_transformer, make_column_selector(dtype_include=['category', 'object']))
            ],
            verbose_feature_names_out=False,
            remainder='passthrough'
        ))
    ])
    joblib.dump(full_pipeline, os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl"))

    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "application_test.csv"), low_memory=False)
    test_df = reduce_mem_usage(test_df)
    test_ids = test_df['SK_ID_CURR'].copy()
    X_test = test_df.drop(columns=['SK_ID_CURR'])
    del test_df
    gc.collect()

    print("Applying transformations to test data...")
    X_test_clean = cleanup_pipeline.transform(X_test)
    del X_test
    gc.collect()

    X_test_processed = preprocessor.transform(X_test_clean)
    del X_test_clean
    gc.collect()

    print("Saving processed test data...")
    joblib.dump(X_test_processed, os.path.join(FEATURE_ENG_DIR, "X_test_processed.joblib"))
    joblib.dump(test_ids, os.path.join(FEATURE_ENG_DIR, "test_ids.joblib"))
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()
