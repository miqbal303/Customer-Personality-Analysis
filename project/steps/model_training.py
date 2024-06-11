import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import numpy as np
from zenml.steps import step, BaseParameters
from project.logger import logging
from project.exception import CustomException

class ModelTrainingParams(BaseParameters):
    test_size: float = 0.3
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = None

@step
def model_training(transformed_data_path: str, cluster_file_path: str, params: ModelTrainingParams) -> str:
    try:
        logging.info("Starting model training process.")
        data = pd.read_csv(transformed_data_path)
        clusters = pd.read_csv(cluster_file_path)

        X = data.values
        y = clusters['Cluster'].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.test_size, random_state=params.random_state)

        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=params.n_estimators, max_depth=params.max_depth, random_state=params.random_state),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=params.n_estimators, max_depth=params.max_depth, random_state=params.random_state)
        }

        accuracy_scores = {}
        auc_scores = {}
        classification_reports = {}
        model_paths = {}

        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores[model_name] = accuracy
            report = classification_report(y_test, y_pred)
            classification_reports[model_name] = report

            auc = None
            if y_pred_proba is not None:
                if len(np.unique(y_train)) > 2:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                else:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                auc_scores[model_name] = auc

            # Log metrics to MLflow
            mlflow.log_metric(f"{model_name}_accuracy", accuracy)
            if auc is not None:
                mlflow.log_metric(f"{model_name}_auc", auc)
            mlflow.log_text(report, f"{model_name}_classification_report.txt")

            model_path = os.path.join("artifacts", f"{model_name.lower().replace(' ', '_')}_model.pkl")
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            mlflow.log_artifact(model_path)

            model_paths[model_name] = model_path

        # Find the best model
        best_model_name = max(accuracy_scores, key=accuracy_scores.get)
        best_model_accuracy = accuracy_scores[best_model_name]
        best_model_auc = auc_scores.get(best_model_name, "N/A")
        best_classification_report = classification_reports[best_model_name]

        if best_model_accuracy >= 0.60:
            logging.info(f'Best Model: {best_model_name}')
            logging.info(f'Accuracy: {best_model_accuracy}')
            logging.info(f'AUC_ROC Score: {best_model_auc}')
            logging.info(f'Classification Report: {best_classification_report}')
            mlflow.autolog()

            return model_paths[best_model_name]
        else:
            logging.info("There is no best model with accuracy above 60%")
            return "There is no best model with accuracy above 60%"

    except Exception as e:
        raise CustomException(e, sys)
