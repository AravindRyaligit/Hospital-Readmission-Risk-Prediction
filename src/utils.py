"""
Utility functions for Hospital Readmission Prediction System
"""
import logging
import joblib
import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np

def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format
        )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Model serialization
def save_model(model: Any, filepath: Path):
    """Save model to disk using joblib"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(filepath: Path) -> Any:
    """Load model from disk"""
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model

# Data validation
def validate_input_data(data: pd.DataFrame, required_columns: list) -> bool:
    """Validate that input data has required columns"""
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    return True

def convert_to_binary(series: pd.Series, positive_value: str = 'yes') -> pd.Series:
    """Convert yes/no to 1/0"""
    return (series.str.lower() == positive_value.lower()).astype(int)

# JSON utilities
def save_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"JSON saved to {filepath}")

def load_json(filepath: Path) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"JSON loaded from {filepath}")
    return data

# Metrics formatting
def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary for display"""
    return "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

def check_data_quality(df: pd.DataFrame) -> Dict:
    """Generate data quality report"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    return report
