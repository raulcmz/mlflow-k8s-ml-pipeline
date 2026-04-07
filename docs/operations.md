# Operations Guide

## Overview

This document describes the operational workflow for the `mlflow-k8s-ml-pipeline` project.

The platform is composed of:

- MLflow as tracking server and model registry
- PostgreSQL as backend store
- MinIO as S3-compatible artifact store
- FastAPI as inference service
- Kubernetes for deployment
- Ingress for external access

---

## Platform Endpoints

### MLflow
- Tracking/UI: `http://mlflow.192.168.1.121.nip.io`

### Serving API
- Inference docs: `http://mlflow-serving.192.168.1.121.nip.io/docs`

### MinIO
- API: `http://minio-api.192.168.1.121.nip.io`
- Console: `http://minio-console.192.168.1.121.nip.io`

---

## Training Workflow

Before running training, export the required environment variables:

```bash
export MLFLOW_TRACKING_URI=http://mlflow.192.168.1.121.nip.io
export MLFLOW_S3_ENDPOINT_URL=http://minio-api.192.168.1.121.nip.io
export AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY>
export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_KEY>
export AWS_S3_FORCE_PATH_STYLE=true
```

## Run training:

`python training/train.py`

This will:

1. Load and clean the Telco Churn dataset
2. Train baseline models
3. Compare metrics
4. Register the best model in MLflow Model Registry
5. Assign the alias `champion` to the best version

## Model Promotion Workflow

The serving layer loads the model using this URI pattern:

`models:/telco-churn-model@champion`

This means the API does not depend on a specific run ID.

If a new version should become active, update the alias in MLflow Model Registry.

The alias champion should always point to the currently deployed production model.

## Serving Workflow
### Local

Export the environment variables:

```bash
export MODEL_URI=models:/telco-churn-model@champion
export MLFLOW_TRACKING_URI=http://mlflow.192.168.1.121.nip.io
export MLFLOW_S3_ENDPOINT_URL=http://minio-api.192.168.1.121.nip.io
export AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY>
export AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_KEY>
export AWS_S3_FORCE_PATH_STYLE=true
```


Run:

```bash
uvicorn serving.app:app --reload
```

### Docker

Build:

```bash
docker build -f docker/Dockerfile.serving -t rcabe005/mlflow-serving:latest .
```

Run:

```bash
docker run -p 8000:8000 \
  -e MODEL_URI=models:/telco-churn-model@champion \
  -e MLFLOW_TRACKING_URI=http://mlflow.192.168.1.121.nip.io \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio-api.192.168.1.121.nip.io \
  -e AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY> \
  -e AWS_SECRET_ACCESS_KEY=<YOUR_SECRET_KEY> \
  -e AWS_S3_FORCE_PATH_STYLE=true \
  rcabe005/mlflow-serving:latest
```

## Kubernetes Deployment Workflow

Apply resources:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/api-ingress.yaml
```

## Restart deployment if config changes:

```bash
kubectl rollout restart deploy/mlflow-serving -n ml-serving
kubectl rollout status deploy/mlflow-serving -n ml-serving
```

## Health Checks
### Local

`http://127.0.0.1:8000/health`

### Kubernetes

`http://mlflow-serving.192.168.1.121.nip.io/health`

## Troubleshooting
### Model not loading

Check:

- MODEL_URI
- MLFLOW_TRACKING_URI
- MLFLOW_S3_ENDPOINT_URL
- S3 credentials
- alias existence in MLflow Model Registry

## Serving pod not starting

```bash
kubectl get pods -n ml-serving
kubectl describe pod -n ml-serving <pod-name>
kubectl logs -n ml-serving <pod-name>
```

## Training fails when logging artifacts

Check that:

- boto3 is installed
- MinIO endpoint is reachable

## MLflow UI blocked by host validation

Ensure MLflow is configured with:

MLFLOW_SERVER_ALLOWED_HOSTS
MLFLOW_SERVER_CORS_ALLOWED_ORIGINS
