# Reports Directory

This directory contains model evaluation reports and visualizations.

## Generated Files

After training the models, the following visualization files will be created:

### XGBoost
- `xgboost_roc_curve.png` - ROC curve
- `xgboost_feature_importance.png` - Feature importance plot
- `xgboost_confusion_matrix.png` - Confusion matrix
- `xgboost_metrics.json` - Performance metrics
- `xgboost_feature_importance.csv` - Feature importance data

### LightGBM
- `lightgbm_roc_curve.png` - ROC curve
- `lightgbm_feature_importance.png` - Feature importance plot
- `lightgbm_confusion_matrix.png` - Confusion matrix
- `lightgbm_metrics.json` - Performance metrics
- `lightgbm_feature_importance.csv` - Feature importance data

## Note

Report files are excluded from Git. They will be automatically generated when you train the models.
