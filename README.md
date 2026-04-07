# mlflow-k8s-ml-pipeline

End-to-end machine learning pipeline for training, tracking, and serving a tabular classification model using MLflow, FastAPI, Docker, and Kubernetes.

## Overview

This project demonstrates a production-oriented ML workflow deployed on Kubernetes. It focuses on reproducibility, experiment tracking, model registration, and containerized inference serving rather than model complexity.

The initial use case is customer churn prediction using the Telco Customer Churn dataset.

## Goals

- Train and evaluate a tabular ML model
- Track experiments and metrics with MLflow
- Register trained models in MLflow Model Registry
- Serve predictions through a FastAPI inference service
- Deploy all components on a local Kubernetes cluster
- Keep the architecture simple, modular, and reproducible

## Architecture

See [docs/architecture.md](docs/architecture.md)

## Tech Stack

- Python
- scikit-learn
- pandas
- MLflow
- FastAPI
- Docker
- Kubernetes

## Project Structure

```text
mlflow-k8s-ml-pipeline/
├── docs/
├── training/
├── serving/
├── docker/
├── mlflow/
├── k8s/
└── examples/

## Architecture

- [Architecture](docs/architecture.svg)
- [Operations Guide](docs/operations.md)