import os
from pathlib import Path
import pandas as pd

def load_data(input_path):
    input_path = os.getenv("RAW_DATA_PATH", "data/raw/churn.csv")
    print(f"Loading data from: {input_path}")
    return pd.read_csv(input_path)

def preprocess(df):
    print(f"Preprocessing")
    df.dropna(inplace=True)
    df = df.drop(columns=['customerID'], errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Data saved to : {output_path}")

def main():
    input_path = os.getenv("RAW_DATA_PATH", "data/raw/churn.csv")
    output_path = os.getenv("OUTPUT_PATH", "data/processed/telecom_customer_churn.csv")
    df = load_data(input_path)
    df_pre_processed = preprocess(df)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    save_data(df_pre_processed,output_path)


if __name__ == "__main__":
    main()
