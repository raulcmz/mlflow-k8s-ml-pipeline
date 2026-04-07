.PHONY: help train serve-local docker-build-serving docker-run-serving deploy-serving restart-serving logs-serving

IMAGE_NAME=rcabe005/mlflow-serving:latest
MODEL_URI=models:/telco-churn-model@champion
MLFLOW_TRACKING_URI=http://mlflow.192.168.1.121.nip.io
MLFLOW_S3_ENDPOINT_URL=http://minio-api.192.168.1.121.nip.io

help:
	@echo "Available targets:"
	@echo "  make train                 - Run the training pipeline"
	@echo "  make serve-local           - Run FastAPI locally"
	@echo "  make docker-build-serving  - Build the serving Docker image"
	@echo "  make docker-run-serving    - Run the serving Docker container"
	@echo "  make deploy-serving        - Deploy serving resources to Kubernetes"
	@echo "  make restart-serving       - Restart serving deployment in Kubernetes"
	@echo "  make logs-serving          - Show logs from serving deployment"

train:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	MLFLOW_S3_ENDPOINT_URL=$(MLFLOW_S3_ENDPOINT_URL) \
	AWS_ACCESS_KEY_ID=$$AWS_ACCESS_KEY_ID \
	AWS_SECRET_ACCESS_KEY=$$AWS_SECRET_ACCESS_KEY \
	AWS_S3_FORCE_PATH_STYLE=true \
	python -m training.train

serve-local:
	MODEL_URI=$(MODEL_URI) \
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	MLFLOW_S3_ENDPOINT_URL=$(MLFLOW_S3_ENDPOINT_URL) \
	AWS_ACCESS_KEY_ID=$$AWS_ACCESS_KEY_ID \
	AWS_SECRET_ACCESS_KEY=$$AWS_SECRET_ACCESS_KEY \
	AWS_S3_FORCE_PATH_STYLE=true \
	uvicorn serving.app:app --reload

docker-build-serving:
	docker build -f docker/Dockerfile.serving -t $(IMAGE_NAME) .

docker-run-serving:
	docker run -p 8000:8000 \
		-e MODEL_URI=$(MODEL_URI) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		-e MLFLOW_S3_ENDPOINT_URL=$(MLFLOW_S3_ENDPOINT_URL) \
		-e AWS_ACCESS_KEY_ID=$$AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY=$$AWS_SECRET_ACCESS_KEY \
		-e AWS_S3_FORCE_PATH_STYLE=true \
		$(IMAGE_NAME)

deploy-serving:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secret.yaml
	kubectl apply -f k8s/api-deployment.yaml
	kubectl apply -f k8s/api-service.yaml
	kubectl apply -f k8s/api-ingress.yaml

restart-serving:
	kubectl rollout restart deploy/mlflow-serving -n ml-serving
	kubectl rollout status deploy/mlflow-serving -n ml-serving

logs-serving:
	kubectl logs -n ml-serving deploy/mlflow-serving --tail=100 -f