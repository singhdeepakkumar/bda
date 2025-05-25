import os
import pandas as pd
from zipfile import ZipFile

def download_from_kaggle(dataset="archive/telco-customer-churn", output_path="data/unprocessed"):
    os.makedirs(output_path, exist_ok=True)
    print(f"Downloading dataset: {dataset}")
    os.system(f"kaggle datasets download -d {dataset} -p {output_path}")

    for file in os.listdir(output_path):
        if file.endswith(".zip"):
            with ZipFile(os.path.join(output_path, file), 'r') as zip_ref:
                zip_ref.extractall(output_path)
            os.remove(os.path.join(output_path, file))

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.dropna()
    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    unprocessed_dir = "data/unprocessed"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    download_from_kaggle()
    df = load_data(f"{unprocessed_dir}/telecom_customer_churn.csv")
    df_clean = preprocess(df)
    save_data(df_clean, f"{processed_dir}/churn_clean.csv")
