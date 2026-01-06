# MLOps Heart Disease Prediction Project

## Introduction

This project implements an end-to-end MLOps pipeline for heart disease prediction ml classifier. The system is designed to predict the presence of heart disease in patients based on various clinical parameters such as age, sex, chest pain type, blood pressure, cholesterol levels, and other medical indicators.

The project demonstrates comprehensive MLOps practices including:
- **Data Processing & Feature Engineering**: Automated data cleaning, preprocessing, and feature transformation
- **Model Training & Experimentation**: Training multiple ML models (Logistic Regression, Random Forest) with MLflow tracking
- **Model Deployment**: Containerized Flask REST API for real-time predictions
- **Container Orchestration**: Kubernetes deployment with service discovery and load balancing
- **Monitoring & Observability**: Integrated Prometheus and Grafana for metrics visualization and system monitoring
- **CI/CD Ready**: Docker-based deployment pipeline with health checks and logging

### Team Contributions

## Group: Group 55

| Name | BITS ID | Contribution |
|------|---------|--------------|
| KIRANJEET KAUR ISHAR | 2024aa05769 | 100% |
| SENTHILKUMAR K | 2024aa05227 | 100% |
| SENTHIL KUMAR K | 2024aa05233 | 100% |
| SHIV PRASAD VERMA | 2024aa05874 | 100% |
| GURUPRASAD MISHRA | 2024aa05858 | 100% |

---

## Model Training

### Prerequisites
- Python 3.12
- Required packages: scikit-learn, pandas, mlflow, joblib

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd MLOps-Heart-Disease-Project
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Training the Model

#### Using GitHub Actions (CI/CD)

1. **Ensure GitHub Actions workflow exists:**
   - Check `.github/workflows/main.yml`

2. **Trigger the workflow:**

   **Manually trigger:**
   - Go to GitHub repository → Actions tab
   - Select "main" workflow
   - Click "Run workflow"

   **Automatic trigger:**
   - Push changes to the repository
   - Workflow runs automatically on push/pull request to main branch

3. **View workflow results:**
   - Navigate to Actions tab in GitHub
   - Click on the workflow run
   - View training logs and model metrics

4. **Download the trained model:**
   - After workflow completes successfully
   - Model artifact will be available for downloads
   - Pull the latest model and store it in /models folder

### Verify Model Output

After training, verify the model file exists:
```bash
ls -lh models/heartDisease_Classifier.pkl
```

### View MLflow Tracking UI

To explore training metrics and experiments:
```bash
mlflow ui
```

Then open: http://localhost:5000

### Using the Trained Model

The trained model (`models/heartDisease_Classifier.pkl`) is automatically used by:
- Flask API (`app.py`)
- Docker container
- Kubernetes deployment

After training a new model, rebuild and redeploy using the instructions provided in the next section.

---

## Kubernetes Deployment Guide - Docker Desktop

## Prerequisites

**Enable Kubernetes in Docker Desktop:**
1. Open Docker Desktop
2. Go to Settings (⚙️) → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"
5. Wait for Kubernetes to start (green indicator in bottom-left)

## Deployment Steps

### 1. Build the Docker Image
```bash
docker build -t heart-disease-app:latest .
```

### 2. Deploy Application to Kubernetes
```bash
# Deploy the ML application
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 3. Deploy Monitoring Stack

#### Deploy Prometheus
```bash
kubectl apply -f k8s/prometheus-deployment.yaml
```

#### Deploy Grafana
```bash
kubectl apply -f k8s/grafana-deployment.yaml
```

### 4. Verify All Deployments

Check all resources:
```bash
kubectl get all
```

Expected output should show:
- `deployment.apps/heart-disease-app` (READY: 2/2)
- `deployment.apps/prometheus` (READY: 1/1)
- `deployment.apps/grafana` (READY: 1/1)
- Services for all three deployments
- Pods in Running state

Check pod status:
```bash
kubectl get pods
kubectl get deployments
kubectl get services
```

### 5. Access the Services

#### Access the ML Application
Get the service details:
```bash
kubectl get svc heart-disease-app-service
```

The LoadBalancer service will be available at `localhost:80`

**Test health endpoint:**
```bash
curl http://localhost/health
```

**Test prediction endpoint:**
```bash
curl -X POST http://localhost/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```
**Test prediction endpoint: Run Multiple Requests**
```bash
for i in {1..100}; do echo -n "Request $i: "; curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST http://localhost/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'; done
```

#### Access Prometheus
```bash
kubectl get svc prometheus
```

Access Prometheus UI:
- URL: http://localhost:9090
- Check targets: http://localhost:9090/targets
- Verify the app metrics are being scraped

**Useful Prometheus Queries:**
- `flask_http_request_total` - Total HTTP requests
- `rate(flask_http_request_total[5m])` - Request rate over 5 minutes
- `flask_http_request_exceptions_total` - Total exceptions

#### Access Grafana
```bash
kubectl get svc grafana
```

Access Grafana UI:
- URL: http://localhost:3000
- Default credentials: admin/admin (change on first login)

**Configure Grafana Dashboard:**
1. Login to Grafana
2. Go to Configuration → Data Sources
3. Verify Prometheus data source is configured (http://prometheus:9090). Click "Save & Test" - it should show "Data source is working".
4. Create Dashboard with above Prometheus Queries

## Monitoring Guide

### View Application Logs
```bash
# View logs for the ML application
kubectl logs -l app=heart-disease-app

# Follow logs in real-time
kubectl logs -f deployment/heart-disease-app

# View logs for specific pod
kubectl logs <pod-name>
```

### View Prometheus Metrics
```bash
# Check Prometheus metrics endpoint
curl http://localhost/metrics
```

### Monitor Resource Usage
```bash
# Watch pod status in real-time
kubectl get pods -w

# View resource usage
kubectl top pods
kubectl top nodes
```

### View Events
```bash
# View recent events
kubectl get events --sort-by=.metadata.creationTimestamp | tail -20

# Watch events in real-time
kubectl get events -w
```

### Describe Resources
```bash
# Get detailed information about deployment
kubectl describe deployment heart-disease-app

# Get detailed information about pod
kubectl describe pod <pod-name>

# Get detailed information about service
kubectl describe service heart-disease-app-service
```

## Scaling

Scale up the application:
```bash
kubectl scale deployment heart-disease-app --replicas=3
kubectl get pods
```

Scale down:
```bash
kubectl scale deployment heart-disease-app --replicas=1
```

## Troubleshooting

### Check Pod Logs
```bash
kubectl logs deployment/heart-disease-app
kubectl logs deployment/prometheus
kubectl logs deployment/grafana
```

### Check Pod Status
```bash
kubectl get pods
kubectl describe pod <pod-name>
```

### Restart Deployment
```bash
kubectl rollout restart deployment heart-disease-app
kubectl rollout restart deployment prometheus
kubectl rollout restart deployment grafana
```

### Check Service Endpoints
```bash
kubectl get endpoints heart-disease-app-service
kubectl get endpoints prometheus-service
kubectl get endpoints grafana-service
```

## Cleanup

Delete all resources:
```bash
kubectl delete -f k8s/
```

Or delete individually:
```bash
kubectl delete deployment heart-disease-app
kubectl delete service heart-disease-app-service
kubectl delete deployment prometheus
kubectl delete service prometheus-service
kubectl delete deployment grafana
kubectl delete service grafana-service
```