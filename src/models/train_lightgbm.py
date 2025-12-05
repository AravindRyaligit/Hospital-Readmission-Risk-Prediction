"""
LightGBM model training with hyperparameter tuning
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import *
from utils import logger, load_model, save_model, save_json

class LightGBMTrainer:
    """Train and evaluate LightGBM model"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def train(self, X_train, y_train):
        """Train LightGBM with hyperparameter tuning"""
        logger.info("Starting LightGBM training with hyperparameter tuning...")
        
        base_model = LGBMClassifier(
            objective='binary',
            metric='auc',
            random_state=RANDOM_STATE,
            verbose=-1
        )
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=LIGHTGBM_PARAMS,
            n_iter=20,
            scoring=SCORING_METRIC,
            cv=CV_FOLDS,
            verbose=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating LightGBM model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        
        metrics = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"LightGBM AUC-ROC: {auc_roc:.4f}")
        logger.info(f"LightGBM AUC-PR: {auc_pr:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        return metrics, y_pred, y_pred_proba
    
    def plot_feature_importance(self, top_n=20):
        """Plot top N feature importances"""
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='plasma')
        plt.title('LightGBM - Top Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        plot_path = REPORTS_DIR / 'lightgbm_feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {plot_path}")
        plt.close()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'LightGBM (AUC = {auc_score:.4f})', linewidth=2, color='purple')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('LightGBM - ROC Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plot_path = REPORTS_DIR / 'lightgbm_roc_curve.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {plot_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=True)
        plt.title('LightGBM - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        plot_path = REPORTS_DIR / 'lightgbm_confusion_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {plot_path}")
        plt.close()
    
    def save_model(self):
        """Save trained model"""
        save_model(self.model, LIGHTGBM_MODEL_FILE)
        
        # Save best parameters
        params_path = MODELS_DIR / 'lightgbm_params.json'
        save_json(self.best_params, params_path)
        
        # Save feature importance
        importance_path = REPORTS_DIR / 'lightgbm_feature_importance.csv'
        self.feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")

def main():
    """Main training pipeline for LightGBM"""
    logger.info("Loading processed data...")
    processed_data = load_model(PROCESSED_DATA_FILE)
    
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    trainer = LightGBMTrainer()
    
    trainer.train(X_train, y_train)
    
    metrics, y_pred, y_pred_proba = trainer.evaluate(X_test, y_test)
    
    trainer.plot_feature_importance()
    trainer.plot_roc_curve(y_test, y_pred_proba)
    trainer.plot_confusion_matrix(y_test, y_pred)
    
    trainer.save_model()
    
    metrics_path = REPORTS_DIR / 'lightgbm_metrics.json'
    save_json(metrics, metrics_path)
    
    logger.info("LightGBM training completed successfully!")
    return trainer.model, metrics

if __name__ == "__main__":
    main()
