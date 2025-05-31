# MLOps: Telecom Customer Churn Prediction

An end-to-end MLOps pipeline to automate the training, versioning, evaluation, and deployment of a telecom churn prediction model. Built on GitHub Actions, MLflow, DVC, and Kubernetes.

---

## Features

- Scheduled & manual model retraining workflows
- CI/CD workflows with GitHub Actions
- Model tracking and registry via MLflow
- Data versioning via DVC with Dagshub
- Kubernetes deployment on GCP using self-hosted runners
- Promotion of best models to Production using model aliases

---

--

## Pipelines

### CI/CD Pipeline

Triggered on push to `master`, `release/**`, or `feature/**`

Steps:

- Run data loading, preprocessing, feature engineering
- Train and test model
- Track artifacts in MLflow and DVC

### Retraining Workflow

Triggered nightly (cron) or manually

Steps:

- Re-run data pipeline
- Train new model
- Compare to `production` model via MLflow alias
- Register as candidate if better

### Promotion Workflow

Triggered on `release/**` or manual version input

Steps:

- Promote model version to alias `production`
- Trigger app deployment in app repo

### App Deployment

Triggered by model promotion

Steps:

- Download model
- Build Docker image
- Push to Docker Hub
- Deploy via `kubectl`

---

## Tools & Integrations

- **GitHub Actions**: CI/CD engine
- **MLflow**: Model tracking, registry, aliasing
- **DVC + Dagshub**: Data & model versioning
- **Docker + Kubernetes**: Deployment
- **GCP VM**: Hosts Minikube cluster and GitHub self-hosted runner

---

## Branching Strategy

- `master`: stable, production-ready
- `feature/**`: isolated dev work
- `development/**`: integration
- `release/**`: integration & promotion triggers

---

## Setup & Usage

### Prerequisites

- Python 3.9, Docker, Git, DVC, Minikube
- GitHub Secrets: `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `GH_TOKEN`, etc.

### Run Locally

```bash
PYTHONPATH=. python src/pipeline/data/data_loading.py
PYTHONPATH=. python src/pipeline/data/pre_processing.py
PYTHONPATH=. python src/pipeline/data/feature_processing.py
PYTHONPATH=. python src/pipeline/model/model_train.py
PYTHONPATH=. python src/pipeline/model/model_test.py
```
