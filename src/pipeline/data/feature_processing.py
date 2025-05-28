from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from mlflow import start_run, set_experiment, log_param, log_metric, log_artifact, mlflow
import pandas as pd

from src.common.util import load_params, logging_setup

logger = logging_setup('feature_engineering')
mlflow.autolog()

def load_data(input_path):    
    print(f"Loading data from: {input_path}")
    logger.debug('Loading data from %s', input_path)
    return pd.read_csv(input_path)



def process(df):
    log_metric("rows", df.shape[0])
    log_metric("columns", df.shape[1])
    df = df.loc[df['Customer Status'] != 'Joined']
    logger.debug('Removed data where customer_status != joined')
    # These features will hinder the model's ability 
    df = df.drop(columns=['Customer ID', 'City', 'Zip Code', 'Latitude', 'Longitude'])
    logger.debug('Removed columns Customer ID City Zip Code Latitude &Longitude')
    

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data saved to : {output_path}")
    logger.debug('data saved to %s', output_path)
    

def main():
    set_experiment("feature_engineering")
    with start_run():           
        df = load_data("./data/raw/telecom_customer_churn_cleaned.csv")
        df_new = process(df)
        output_path = "data/processed/telecom_customer_churn.csv"
        save_data(df_new,output_path)
        log_artifact(output_path)
    

if __name__ == "__main__":
    main()
