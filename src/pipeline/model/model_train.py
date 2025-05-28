import os
import joblib
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from mlflow import log_metric, log_param, log_artifact, set_experiment, start_run, mlflow
from src.common.util import load_params, logging_setup

logger = logging_setup('model_building')
mlflow.autolog()

def load_data(input_path):    
    print(f"Loading data from: {input_path}")
    logger.info(f"Loading data from: {input_path}")    
    df = pd.read_csv(input_path)
    logger.debug('DataFrame shape %s', df.shape)
    return df

def save_model(model, path="models/decision_tree_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path

def separate_features_and_target(df):
    X = df.drop(columns=["Customer Status"])
    y = df["Customer Status"]
    return X, y

def train_model(train_df):
    X_train, y_train = separate_features_and_target(train_df)

    param_grid = {
           'criterion': ['gini', 'entropy'],
           'max_leaf_nodes': range(1, 20)
    }
    params = load_params('./parameters.yml')['model_building']
    random_state = params["random_state"]
    mlflow.log_metric("training_sample_count", len(X_train))
    mlflow.log_param("random_state", params["random_state"])
    mlflow.log_param("random_grid", param_grid)
    mlflow.log_param("cv", 2)
    base_model = DecisionTreeClassifier(random_state=random_state )
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logger.info(f"Best Params: {grid_search.best_params_}")
    mlflow.log_param("best_params", grid_search.best_params_)
    mlflow.log_metric("best_score", grid_search.best_score_)

    return best_model
    

def main():
    mlflow.set_experiment("model_building_decision_tree_training_experiment")
    with mlflow.start_run() as run:
        mlflow.log_param("test_param", 123)
        train_df = load_data("data/processed/telecom_customer_churn_train.csv")
        best_model = train_model(train_df)
        model_path = save_model(best_model)
        log_artifact(model_path)

if __name__ == '__main__':
    main()
