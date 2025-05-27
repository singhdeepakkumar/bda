import os
import pandas as pd

def load_data(input_path):
    input_path = os.getenv("RAW_DATA_PATH", "data/raw/churn.csv")
    print(f"Loading data from: {input_path}")
    return pd.read_csv(input_path)

def preprocess(df):
    df.dropna(inplace=True)
    df = df.drop(columns=['customerID'], errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
