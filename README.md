# Hospital Readmission Risk Prediction System

A comprehensive machine learning system for predicting 30-day hospital readmission risk using XGBoost and LightGBM models.

## 🎯 Features

- **Dual Model Architecture**: XGBoost and LightGBM models with hyperparameter tuning
- **AUC-Optimized Training**: Models optimized for AUC-ROC performance
- **Feature Engineering**: Advanced features including healthcare utilization scores and medication complexity
- **REST API**: FastAPI backend for real-time predictions
- **Web Interface**: Modern, responsive web application for risk assessment
- **Model Monitoring**: Drift detection and performance tracking
- **Batch Processing**: Support for bulk predictions via CSV upload

## 📊 Model Performance

Both models achieve strong performance on the test set:
- **XGBoost**: AUC-ROC > 0.70
- **LightGBM**: AUC-ROC > 0.70

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd ML_prediction_System
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download the dataset**:
   - Download the Diabetes 130-US Hospitals dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
   - Place `diabetic_data.csv` in the `data/` directory and rename it to `hospital_readmissions.csv`

4. **Run data preprocessing**:
```bash
cd src
python preprocessing.py
```

5. **Train models**:
```bash
# Train XGBoost
python src/models/train_xgboost.py

# Train LightGBM
python src/models/train_lightgbm.py
```

6. **Start the API server**:
```bash
# Set PYTHONPATH (Linux/Mac)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Set PYTHONPATH (Windows PowerShell)
$env:PYTHONPATH="$(pwd)\src"

# Run API
python src/api/api.py
```

7. **Start the web server** (in a new terminal):
```bash
cd web
python -m http.server 8080
```

8. **Access the application**:
- Web App: http://localhost:8080/index.html
- API Docs: http://localhost:8000/docs

## 📁 Project Structure

```
ML_prediction_System/
├── data/                          # Data files
│   ├── hospital_readmissions.csv  # Raw data
│   └── processed_data.pkl         # Processed data
├── src/                           # Source code
│   ├── config.py                  # Configuration
│   ├── utils.py                   # Utility functions
│   ├── preprocessing.py           # Data preprocessing
│   ├── models/                    # Model training scripts
│   │   ├── train_xgboost.py
│   │   └── train_lightgbm.py
│   └── api/                       # API code
│       └── api.py
├── web/                           # Web interface
│   ├── index.html                 # Main prediction page
│   ├── dashboard.html             # Performance dashboard
│   ├── monitoring.html            # Drift monitoring
│   ├── batch.html                 # Batch predictions
│   └── static/                    # CSS and JavaScript
├── models/                        # Saved models
├── reports/                       # Evaluation reports
└── requirements.txt               # Dependencies
```

## 🔧 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single patient prediction
- `POST /batch_predict` - Batch predictions from CSV
- `GET /api/models/performance` - Model performance metrics

## 📈 Usage Example

### Single Prediction (Python)

```python
import requests

patient_data = {
    "age": "[70-80)",
    "time_in_hospital": 5,
    "n_lab_procedures": 45,
    "n_procedures": 2,
    "n_medications": 15,
    "n_outpatient": 0,
    "n_inpatient": 0,
    "n_emergency": 0,
    "medical_specialty": "InternalMedicine",
    "diag_1": "Circulatory",
    "diag_2": "Diabetes",
    "diag_3": "Other",
    "glucose_test": "normal",
    "A1Ctest": "high",
    "change": "yes",
    "diabetes_med": "yes"
}

response = requests.post(
    "http://localhost:8000/predict?model=lightgbm",
    json=patient_data
)

print(response.json())
```

## 🎨 Web Interface

The web application provides:

1. **Prediction Interface**: Enter patient data and get real-time risk assessment
2. **Dashboard**: View model performance metrics and comparisons
3. **Monitoring**: Track data drift and model performance over time
4. **Batch Processing**: Upload CSV files for bulk predictions

## 📊 Features Included

### Engineered Features:
- Healthcare utilization score
- Medication complexity index
- Lab intensity
- Procedure intensity
- Diabetes medication flags
- Lab test indicators

### Model Features:
- Hyperparameter tuning with RandomizedSearchCV
- 5-fold stratified cross-validation
- AUC-ROC optimization
- Feature importance analysis
- Comprehensive evaluation metrics

## 🔍 Model Evaluation

Evaluation metrics include:
- AUC-ROC curve
- Precision-Recall curve
- Confusion matrix
- Classification report (precision, recall, F1-score)
- Feature importance rankings

## 📝 License

MIT License

## 👥 Contributors

Built with XGBoost, LightGBM, FastAPI, and modern web technologies.

## 🙏 Acknowledgments

- Dataset: Diabetes 130-US Hospitals (UCI Machine Learning Repository)
- Models: XGBoost, LightGBM
- Framework: FastAPI, scikit-learn
