"""
Data preprocessing pipeline for Hospital Readmission Prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import *
from utils import logger, save_model, convert_to_binary, check_data_quality

class DataPreprocessor:
    """Handle all data preprocessing tasks"""
    
    def __init__(self):
        self.label_encoders = {}
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self, filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
        """Load raw data from CSV"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Convert target variable to binary (yes=1, no=0)
        df_clean[TARGET_COLUMN] = convert_to_binary(df_clean[TARGET_COLUMN], 'yes')
        
        # Handle missing values in categorical features
        for col in CATEGORICAL_FEATURES:
            if col in df_clean.columns:
                # Replace 'Missing' or NaN with 'Unknown'
                df_clean[col] = df_clean[col].fillna('Unknown')
                df_clean[col] = df_clean[col].replace('Missing', 'Unknown')
        
        # Handle missing values in numerical features (fill with median)
        for col in NUMERICAL_FEATURES:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering"""
        logger.info("Creating engineered features...")
        df_features = df.copy()
        
        df_features['healthcare_utilization'] = (
            df_features['n_outpatient'] + 
            df_features['n_inpatient'] + 
            df_features['n_emergency']
        )
        
        df_features['medication_complexity'] = (
            df_features['n_medications'] / 
            (df_features['time_in_hospital'] + 1)  # +1 to avoid division by zero
        )
        
        df_features['lab_intensity'] = (
            df_features['n_lab_procedures'] / 
            (df_features['time_in_hospital'] + 1)
        )
        
        df_features['procedure_intensity'] = (
            df_features['n_procedures'] / 
            (df_features['time_in_hospital'] + 1)
        )
        
        if 'diabetes_med' in df_features.columns:
            df_features['has_diabetes_med'] = convert_to_binary(df_features['diabetes_med'], 'yes')
        
        if 'change' in df_features.columns:
            df_features['medication_changed'] = convert_to_binary(df_features['change'], 'yes')
        
        if 'glucose_test' in df_features.columns:
            df_features['glucose_tested'] = (df_features['glucose_test'] != 'no').astype(int)
            df_features['glucose_high'] = (df_features['glucose_test'] == 'high').astype(int)
        
        if 'A1Ctest' in df_features.columns:
            df_features['a1c_tested'] = (df_features['A1Ctest'] != 'no').astype(int)
            df_features['a1c_high'] = (df_features['A1Ctest'] == 'high').astype(int)
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} new features")
        return df_features
    
    def prepare_for_training(self, df: pd.DataFrame):
        """Prepare data for model training"""
        logger.info("Preparing data for training...")
        
        # Separate features and target
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        # Get all categorical and numerical columns (including engineered features)
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
        
        logger.info(f"Categorical columns: {len(categorical_cols)}")
        logger.info(f"Numerical columns: {len(numerical_cols)}")
        
        # Create preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        num_features = numerical_cols
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        self.feature_names = list(num_features) + list(cat_features)
        
        logger.info(f"Transformed array shape: {X_processed.shape}")
        logger.info(f"Feature names count: {len(self.feature_names)}")
        
        # Convert to DataFrame - use actual shape from transformed data
        X_processed = pd.DataFrame(
            X_processed,
            columns=[f'feature_{i}' for i in range(X_processed.shape[1])]
        )
        
        logger.info(f"Final feature count: {X_processed.shape[1]}")
        
        return X_processed, y
    
    def split_data(self, X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        """Split data into train and test sets"""
        logger.info(f"Splitting data with test_size={test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Positive class ratio - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self):
        """Save preprocessor and feature names"""
        save_model(self.preprocessor, PREPROCESSOR_FILE)
        save_model(self.feature_names, FEATURE_NAMES_FILE)
        logger.info("Preprocessor saved successfully")

def main():
    """Main preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    df = preprocessor.load_data()
    
    quality_report = check_data_quality(df)
    logger.info(f"Data Quality Report:\n{quality_report}")
    
    df_clean = preprocessor.clean_data(df)
    
    df_features = preprocessor.create_features(df_clean)
    
    X, y = preprocessor.prepare_for_training(df_features)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    preprocessor.save_preprocessor()
    
    logger.info("Saving processed data...")
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    save_model(processed_data, PROCESSED_DATA_FILE)
    
    logger.info("Preprocessing pipeline completed successfully!")
    return processed_data

if __name__ == "__main__":
    main()
