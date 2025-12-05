# Models Directory

This directory stores the trained machine learning models and preprocessing artifacts.

## Generated Files

After running the training scripts, the following files will be created:

- `xgboost_model.pkl` - Trained XGBoost model
- `lightgbm_model.pkl` - Trained LightGBM model
- `preprocessor.pkl` - Fitted preprocessing pipeline
- `feature_names.pkl` - Feature names after transformation
- `xgboost_params.json` - Best hyperparameters for XGBoost
- `lightgbm_params.json` - Best hyperparameters for LightGBM

## Note

Model files are excluded from Git due to their large size. You need to train the models locally by running:

```bash
cd src/models
python train_xgboost.py
python train_lightgbm.py
```

The models will be automatically saved to this directory.
