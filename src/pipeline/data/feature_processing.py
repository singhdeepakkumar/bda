from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from mlflow import start_run, set_experiment, log_param, log_metric, log_artifact, mlflow
import pandas as pd

from src.common.util import load_params, logging_setup, init_mlflow

logger = logging_setup('feature_engineering')
init_mlflow("feature_engineering")

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
from sklearn.model_selection import train_test_split

def save_train_test_data(df):
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Save to separate CSVs
    train_output_path = "data/processed/telecom_customer_churn_train.csv"
    test_output_path = "data/processed/telecom_customer_churn_test.csv"
    save_data(train_df, train_output_path)
    save_data(test_df, test_output_path)

    # Log artifacts to MLflow
    mlflow.log_artifact(train_output_path)
    mlflow.log_artifact(test_output_path)

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data saved to : {output_path}")
    logger.debug('data saved to %s', output_path)
    

def main():
    set_experiment("feature_engineering")
    with start_run():           
        df = load_data("data/raw/telecom_customer_churn_cleaned.csv")
        df_new = process(df)        
        save_train_test_data(df_new)

if __name__ == "__main__":
    main()
