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

### 2. Deploy to Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### 3. Verify Deployment

Check all resources:
```bash
kubectl get all
```

Check pod status:
```bash
kubectl get pods
kubectl get deployments
```

View logs:
```bash
kubectl logs -l app=heart-disease-app
```

### 4. Access the Application

Get the service details:
```bash
kubectl get svc heart-disease-app-service
```

The LoadBalancer service will be available at `localhost:80`

Test health endpoint:
```bash
curl http://localhost/health
```

Test prediction endpoint:
```bash
curl -X POST http://localhost/predict -H "Content-Type: application/json" -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

## Scaling

Scale up:
```bash
kubectl scale deployment heart-disease-app --replicas=3
kubectl get pods
```

Scale down:
```bash
kubectl scale deployment heart-disease-app --replicas=1
```

## Monitoring

Watch pod status in real-time:
```bash
kubectl get pods -w
```

Describe deployment details:
```bash
kubectl describe deployment heart-disease-app
```

View service endpoints:
```bash
kubectl get endpoints heart-disease-app-service
```

View recent events:
```bash
kubectl get events --sort-by=.metadata.creationTimestamp | tail -20
```

## Verification Commands

Run these commands for documentation:
```bash
# 1. Show all resources
kubectl get all

# 2. Show deployment details
kubectl describe deployment heart-disease-app

# 3. Show pod details
kubectl get pods -o wide

# 4. Show service details
kubectl get svc heart-disease-app-service

# 5. Test endpoints
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
```
```
curl http://localhost/health
curl -X POST http://localhost/predict -H "Content-Type: application/json" -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```
