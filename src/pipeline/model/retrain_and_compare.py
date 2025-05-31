import os
import mlflow
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from src.common.util import init_mlflow, load_params, logging_setup

logger = logging_setup("train_and_compare")
init_mlflow("model_retraining")

MODEL_NAME = "DecisionTreeTelecomCustomerChurnModel"

def load_data():
    df_train = pd.read_csv("data/processed/telecom_customer_churn_train.csv")
    df_test = pd.read_csv("data/processed/telecom_customer_churn_test.csv")
    X_train = df_train.drop(columns=["Customer Status"])
    y_train = df_train["Customer Status"]
    X_test = df_test.drop(columns=["Customer Status"])
    y_test = df_test["Customer Status"]
    return X_train, y_train, X_test, y_test

def main():
    params = load_params("parameters.yml")
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_leaf_nodes": range(2, 10)
    }

    X_train, y_train, X_test, y_test = load_data()

    with mlflow.start_run() as run:
        random_state = params.get("random_state",42)
        clf = GridSearchCV(DecisionTreeClassifier(random_state=random_state),
                           param_grid, cv=5, scoring="accuracy")
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)

        model_uri = mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        logger.info(f"Logged new model with accuracy: {acc:.4f}, f1: {f1:.4f}")

if __name__ == "__main__":
    main()
