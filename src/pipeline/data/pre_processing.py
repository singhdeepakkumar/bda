from pathlib import Path
from mlflow import start_run, set_experiment, log_param, log_metric, log_artifact, mlflow
import os
import pandas as pd

from src.common.util import init_mlflow, load_params, logging_setup

logger = logging_setup('data_preprocessing')
init_mlflow("fdata_preprocessing")

def load_data(input_path):    
    print(f"Loading data from: {input_path}")
    logger.info(f"Loading data from: {input_path}")
    return pd.read_csv(input_path)



def preprocess(df):
    initial_rows = df.shape[0]
    # Remove duplicate rows
    df = df.drop_duplicates(keep='first')
    logger.debug('Duplicates removed')
    final_rows = df.shape[0]
    logger.debug(f"Removed {initial_rows - final_rows} duplicate rows")
    mlflow.log_metric("rows_after_dedup", final_rows)
    # filling 0.0 GB for customers with no internet service
    df.loc[df['Internet Service'] == 'No', 'Avg Monthly GB Download'] = df.loc[df['Internet Service'] == 'No', 'Avg Monthly GB Download'].fillna(0.0)
    
    # filling 0.0 Charges for customers with no phone services
    df.loc[df['Phone Service'] == 'No', 'Avg Monthly Long Distance Charges'] = df.loc[df['Phone Service'] == 'No', 'Avg Monthly Long Distance Charges'].fillna(0.0)

    # customers without internet service cannot have these services
    net_dependent_features = ['Internet Type', 'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data']
    df.loc[df['Internet Service'] == 'No', net_dependent_features] = df.loc[df['Internet Service'] == 'No', net_dependent_features].fillna('No internet service')

    # customers with no phone service cannot opt for multiple lines
    df.loc[df['Phone Service'] == 'No', 'Multiple Lines'] = df.loc[df['Phone Service'] == 'No', 'Multiple Lines'].fillna('No phone service')

    # assuming that some customers did not avail any offer
    df['Offer'] = df['Offer'].fillna('No Offer')

    df = df.drop(columns=['Churn Category', 'Churn Reason'])
    logger.debug('Churn Category &  Churn Reason column drooped from the dataframe')
    
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")
    

def main():
    set_experiment("data_preprocessing")
    with start_run():
        df = load_data("data/raw/telecom_customer_churn_data.csv")
        df_cleaned = preprocess(df)
        output_path = "data/raw/telecom_customer_churn_cleaned.csv"
        save_data(df_cleaned, output_path)
        mlflow.log_artifact(output_path)

if __name__ == "__main__":
    main()
