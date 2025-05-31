import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()
MODEL_NAME = "DecisionTreeTelecomCustomerChurnModel"

def get_metrics(run_id):
    metrics = client.get_run(run_id).data.metrics
    return {
        "accuracy": metrics.get("accuracy", 0.0),
        "f1_score": metrics.get("f1_score", 0.0),
        "recall": metrics.get("recall", 0.0),
        "precision": metrics.get("precision", 0.0)
    }

def main():
    latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not latest_versions:
        print("No model versions found.")
        return

    latest_model = sorted(latest_versions, key=lambda x: int(x.version))[-1]
    new_metrics = get_metrics(latest_model.run_id)

    try:
        prod_model = client.get_model_version_by_alias(MODEL_NAME, "production")
        prod_metrics = get_metrics(prod_model.run_id)
    except Exception:
        print("No production alias set. Skipping production comparison.")
        prod_metrics = {}

    print(f"New model metrics: {new_metrics}")
    print(f"Production model metrics: {prod_metrics}")

    if all(new_metrics.get(m, 0) > prod_metrics.get(m, 0) for m in ["accuracy", "f1_score"]):
        print("New model is better. Promoting to Staging (candidate)...")
        client.set_registered_model_alias(MODEL_NAME, "candidate", latest_model.version)        
    else:
        print("New model is not better than production. Keeping as-is.")

if __name__ == "__main__":
    main()
