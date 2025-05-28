import os
import pandas as pd
from mlflow import start_run, set_experiment, log_metric, log_artifact, mlflow
from src.common.util import logging_setup, init_mlflow

logger = logging_setup('data_loading')
init_mlflow("data_loading")

def load_data(input_path):    
    print(f"Loading data from: {input_path}")
    return pd.read_csv(input_path)

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data saved to : {output_path}")
    logger.debug('data saved to %s', output_path)
    
def main():
    set_experiment("data_loading")
    with start_run():        
        input_path = os.getenv("RAW_DATA_PATH", "data/raw/telecom_customer_churn.csv")
        df = load_data(input_path)
        mlflow.log_metric("rows", df.shape[0])
        mlflow.log_metric("columns", df.shape[1])
        output_path = "data/raw/telecom_customer_churn_data.csv"
        save_data(df, output_path)
        mlflow.log_artifact(output_path)

if __name__ == "__main__":
    main()
