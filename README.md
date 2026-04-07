# mlflow-k8s-ml-pipeline

End-to-end MLOps platform for training, registering, and serving machine learning models with **MLflow**, **MinIO**, **FastAPI**, **Docker**, and **Kubernetes**.

## Overview

This project demonstrates a production-oriented machine learning workflow deployed on Kubernetes.

It focuses on:

- reproducible training
- experiment tracking
- model registry
- remote artifact storage
- containerized inference serving
- Kubernetes-native deployment

The initial use case is **customer churn prediction** using the Telco Customer Churn dataset.

## Architecture

![Architecture](docs/architecture.png)

Additional details:
- [Architecture Notes](docs/architecture.md)
- [Operations Guide](docs/operations.md)

## Key Features

- Train and evaluate tabular ML models with scikit-learn
- Track runs, metrics, and artifacts in MLflow
- Register the best model in MLflow Model Registry
- Use a stable model alias (`champion`) for serving
- Store model artifacts in MinIO (S3-compatible object storage)
- Serve predictions through a FastAPI inference API
- Deploy the inference service on Kubernetes with Ingress
- Separate runtime configuration from code using ConfigMaps and Secrets

## Tech Stack

- Python
- pandas
- scikit-learn
- MLflow
- MinIO
- FastAPI
- Docker
- Kubernetes
- Ingress NGINX
- PostgreSQL

## Project Structure

```text
mlflow-k8s-ml-pipeline/
├── docs/         # architecture and operational documentation
├── docker/       # container build definitions
├── examples/     # sample requests and examples
├── infra/        # Helm values and ingress manifests for platform services
├── k8s/          # Kubernetes manifests for serving deployment
├── serving/      # FastAPI inference service
├── training/     # training pipeline and model registration
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

## Workflow

### 1. Training

The training pipeline:

- loads and cleans the dataset
- applies preprocessing with `ColumnTransformer`
- trains multiple baseline models
- evaluates metrics
- logs runs to MLflow
- registers the best model in Model Registry
- assigns the alias `champion`

### 2. Model Management

The serving layer does not depend on a specific run ID.

Instead, it loads the active model using:

`models:/telco-churn-model@champion`

This allows model promotion without changing application code.

### 3. Serving

The FastAPI service:

- loads the model from remote MLflow
- retrieves artifacts from MinIO
- exposes `/predict`
- runs locally, in Docker, and on Kubernetes

## Quick Start

### Training

```bash
export MLFLOW_TRACKING_URI=http://mlflow.192.168.1.121.nip.io
export MLFLOW_S3_ENDPOINT_URL=http://minio-api.192.168.1.121.nip.io
export AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY>
export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_KEY>
export AWS_S3_FORCE_PATH_STYLE=true

python training/train.py
```

### Local Serving

```bash
export MODEL_URI=models:/telco-churn-model@champion
export MLFLOW_TRACKING_URI=http://mlflow.192.168.1.121.nip.io
export MLFLOW_S3_ENDPOINT_URL=http://minio-api.192.168.1.121.nip.io
export AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY>
export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_KEY>
export AWS_S3_FORCE_PATH_STYLE=true

uvicorn serving.app:app --reload
```

### Docker Serving

```bash
docker build -f docker/Dockerfile.serving -t rcabe005/mlflow-serving:latest .

docker run -p 8000:8000 \
  -e MODEL_URI=models:/telco-churn-model@champion \
  -e MLFLOW_TRACKING_URI=http://mlflow.192.168.1.121.nip.io \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio-api.192.168.1.121.nip.io \
  -e AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY> \
  -e AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_KEY> \
  -e AWS_S3_FORCE_PATH_STYLE=true \
  rcabe005/mlflow-serving:latest
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/api-ingress.yaml
```

## Documentation

Architecture
Operations Guide

## Project Status

Current stage:

- ✅ End-to-end training pipeline
- ✅ Remote MLflow + MinIO integration
- ✅ Model Registry with `champion` alias
- ✅ FastAPI serving on Kubernetes
- ✅ Ingress-based access to MLflow and serving
- 🔜 Data validation with Great Expectations
- 🔜 Drift monitoring with Evidently
- 🔜 CI/CD and image versioning
- 🔜 Feature store integration with Feast

## Design Decisions

- **MLflow Model Registry** is used to decouple serving from individual training runs
- **MinIO** is used as an S3-compatible artifact store
- **FastAPI** provides a lightweight and production-oriented inference API
- **Kubernetes + Ingress** are used instead of port-forwarding to make the platform reusable
- **ConfigMaps and Secrets** separate runtime configuration from code