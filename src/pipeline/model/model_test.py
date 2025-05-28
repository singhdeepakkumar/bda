import json
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import mlflow
from mlflow import set_experiment, start_run, log_metric, log_artifact
from src.common.util import logging_setup

logger = logging_setup('model_testing')
mlflow.autolog()

def load_test_data(path):
    return pd.read_csv(path)

def load_model(path="models/decision_tree_model.pkl"):
    return joblib.load(path)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    logger.info("Classification Report:\n", classification_report(y_test, y_pred))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

def save_metrics(metrics: dict, file_path: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    logger.debug('Metrics saved to %s', file_path)
    
def main():
    set_experiment("decision_tree_testing")
    with start_run():
        test_df = load_test_data("data/processed/telecom_customer_churn_test.csv")
        X_test = test_df.drop(columns=["Customer Status"])
        y_test = test_df["Customer Status"]

        model = load_model("models/decision_tree_model.pkl")
        metrics = evaluate(model, X_test, y_test)

        for key, value in metrics.items():
            log_metric(key, value)

        metrics_path = 'reports/metrics.json'
        save_metrics(metrics, metrics_path)
        log_artifact(metrics_path)

if __name__ == "__main__":
    main()
