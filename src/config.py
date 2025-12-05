"""
Configuration settings for Hospital Readmission Prediction System
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DATA_FILE = DATA_DIR / "hospital_readmissions.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_data.pkl"

XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_model.pkl"
LIGHTGBM_MODEL_FILE = MODELS_DIR / "lightgbm_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.pkl"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.pkl"

XGBOOST_PARAMS = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

LIGHTGBM_PARAMS = {
    'num_leaves': [31, 50, 70],
    'max_depth': [5, 7, 9, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_samples': [20, 30, 50],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING_METRIC = 'roc_auc'

API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.1

CATEGORICAL_FEATURES = [
    'age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3',
    'glucose_test', 'A1Ctest', 'change', 'diabetes_med'
]

NUMERICAL_FEATURES = [
    'time_in_hospital', 'n_lab_procedures', 'n_procedures',
    'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency'
]

TARGET_COLUMN = 'readmitted'

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
