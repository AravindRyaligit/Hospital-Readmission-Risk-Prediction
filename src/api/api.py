"""
FastAPI REST API for Hospital Readmission Prediction
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import *
from utils import logger, load_model

app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="Predict 30-day hospital readmission risk using XGBoost and LightGBM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    xgboost_model = load_model(XGBOOST_MODEL_FILE)
    lightgbm_model = load_model(LIGHTGBM_MODEL_FILE)
    preprocessor = load_model(PREPROCESSOR_FILE)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    xgboost_model = None
    lightgbm_model = None
    preprocessor = None

class PatientData(BaseModel):
    age: str = Field(..., description="Age bracket (e.g., '[70-80)')")
    time_in_hospital: int = Field(..., ge=1, le=14, description="Days in hospital (1-14)")
    n_lab_procedures: int = Field(..., ge=0, description="Number of lab procedures")
    n_procedures: int = Field(..., ge=0, description="Number of procedures")
    n_medications: int = Field(..., ge=0, description="Number of medications")
    n_outpatient: int = Field(..., ge=0, description="Outpatient visits in past year")
    n_inpatient: int = Field(..., ge=0, description="Inpatient visits in past year")
    n_emergency: int = Field(..., ge=0, description="Emergency visits in past year")
    medical_specialty: str = Field(..., description="Medical specialty")
    diag_1: str = Field(..., description="Primary diagnosis")
    diag_2: str = Field(..., description="Secondary diagnosis")
    diag_3: str = Field(..., description="Additional diagnosis")
    glucose_test: str = Field(..., description="Glucose test result (no/normal/high)")
    A1Ctest: str = Field(..., description="A1C test result (no/normal/high)")
    change: str = Field(..., description="Medication change (yes/no)")
    diabetes_med: str = Field(..., description="Diabetes medication prescribed (yes/no)")

class PredictionResponse(BaseModel):
    model_config = {'protected_namespaces': ()}  # Allow 'model_' prefix
    
    readmission_risk: float
    risk_level: str
    model_used: str
    confidence: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_patients: int
    high_risk_count: int

def preprocess_patient_data(patient: PatientData) -> pd.DataFrame:
    """Convert patient data to DataFrame and apply preprocessing"""
    # Convert to dictionary
    data_dict = patient.dict()
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Feature engineering (same as in preprocessing.py)
    df['healthcare_utilization'] = df['n_outpatient'] + df['n_inpatient'] + df['n_emergency']
    df['medication_complexity'] = df['n_medications'] / (df['time_in_hospital'] + 1)
    df['lab_intensity'] = df['n_lab_procedures'] / (df['time_in_hospital'] + 1)
    df['procedure_intensity'] = df['n_procedures'] / (df['time_in_hospital'] + 1)
    df['has_diabetes_med'] = (df['diabetes_med'].str.lower() == 'yes').astype(int)
    df['medication_changed'] = (df['change'].str.lower() == 'yes').astype(int)
    df['glucose_tested'] = (df['glucose_test'] != 'no').astype(int)
    df['glucose_high'] = (df['glucose_test'] == 'high').astype(int)
    df['a1c_tested'] = (df['A1Ctest'] != 'no').astype(int)
    df['a1c_high'] = (df['A1Ctest'] == 'high').astype(int)
    
    # Apply preprocessor
    X_processed = preprocessor.transform(df)
    
    return X_processed

def get_risk_level(probability: float) -> str:
    """Categorize risk level based on probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hospital Readmission Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Single patient prediction",
            "/batch_predict": "Batch prediction from CSV",
            "/health": "Health check",
            "/api/models/performance": "Model performance metrics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = all([xgboost_model, lightgbm_model, preprocessor])
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData, model: str = "lightgbm"):
    """
    Predict readmission risk for a single patient
    
    Args:
        patient: Patient data
        model: Model to use ('xgboost' or 'lightgbm')
    """
    try:
        X = preprocess_patient_data(patient)
        
        if model.lower() == "xgboost":
            selected_model = xgboost_model
        else:
            selected_model = lightgbm_model
        
        if selected_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        probability = selected_model.predict_proba(X)[0, 1]
        
        risk_level = get_risk_level(probability)
        
        confidence = abs(probability - 0.5) * 2
        
        return PredictionResponse(
            readmission_risk=float(probability),
            risk_level=risk_level,
            model_used=model.lower(),
            confidence=float(confidence)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...), model: str = "lightgbm"):
    """
    Batch prediction from CSV file
    
    Args:
        file: CSV file with patient data
        model: Model to use ('xgboost' or 'lightgbm')
    """
    try:
        df = pd.read_csv(file.file)
        
        predictions = []
        high_risk_count = 0
        
        for _, row in df.iterrows():
            patient = PatientData(**row.to_dict())
            result = await predict(patient, model)
            predictions.append(result)
            
            if result.risk_level == "High":
                high_risk_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_patients=len(predictions),
            high_risk_count=high_risk_count
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        # Load metrics
        xgboost_metrics_path = REPORTS_DIR / 'xgboost_metrics.json'
        lightgbm_metrics_path = REPORTS_DIR / 'lightgbm_metrics.json'
        
        with open(xgboost_metrics_path, 'r') as f:
            xgboost_metrics = json.load(f)
        
        with open(lightgbm_metrics_path, 'r') as f:
            lightgbm_metrics = json.load(f)
        
        return {
            "xgboost": {
                "auc_roc": xgboost_metrics['auc_roc'],
                "auc_pr": xgboost_metrics['auc_pr']
            },
            "lightgbm": {
                "auc_roc": lightgbm_metrics['auc_roc'],
                "auc_pr": lightgbm_metrics['auc_pr']
            }
        }
    
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)
