# MLOps Platform - Human Mind & Aptitude Evaluation Model & AKS Deployment

An end-to-end Machine Learning Operations (MLOps) project that trains a model (`model.pkl`) to evaluate human mind readiness / career aptitude based on educational history (10th, 12th, higher education) and current job role, packages the model into a FastAPI microservice, containerizes it with Docker, and automates deployment to Azure Kubernetes Service (AKS).

---

## 📁 Repository Structure

```
mlops/
├── app/                        # FastAPI microservice for model inference
│   ├── __init__.py
│   ├── main.py                 # FastAPI endpoints (/health, /predict, /info)
│   ├── model_loader.py         # Lazy loader for model artifact
│   └── schemas.py              # Pydantic input/output validation schemas
├── src/                        # Machine Learning pipeline source code
│   └── train.py                # Data generation, pipeline preprocessing & RandomForest model training
├── models/
│   └── model.pkl               # Serialized trained scikit-learn model artifact
├── k8s/                        # Kubernetes deployment manifests for AKS
│   ├── namespace.yaml          # K8s namespace ('mlops-system')
│   ├── deployment.yaml         # K8s Deployment with health probes & resource limits
│   ├── service.yaml            # K8s Service (LoadBalancer)
│   └── hpa.yaml                # HorizontalPodAutoscaler (CPU/Memory scaling)
├── .github/workflows/
│   └── deploy.yml              # GitHub Actions CI/CD pipeline for AKS
├── tests/                      # Automated API & model integration tests
│   └── test_api.py
├── Dockerfile                  # Production container definition
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🧠 Machine Learning Model & Features

The model evaluates a candidate's profile based on the following input parameters:

| Parameter | Type | Range / Options | Description |
|---|---|---|---|
| `score_10th` | `float` | 0.0 - 100.0 | 10th grade percentage / score |
| `score_12th` | `float` | 0.0 - 100.0 | 12th grade percentage / score |
| `education_level` | `str` | High School, Bachelor, Master, PhD | Highest education degree |
| `current_job` | `str` | Software Engineer, Data Scientist, Manager, etc. | Current professional domain/role |
| `years_of_experience` | `float` | 0.0 - 50.0 | Years of relevant experience |

### Output Prediction:
- **`predicted_readiness_score`**: 0.0 - 100.0 (Continuous readiness index)
- **`readiness_tier`**: High Alignment (>=80), Moderate Alignment (60-79), or Emerging Phase (<60).

---

## 🚀 Local Quickstart Guide

### 1. Set up Virtual Environment & Dependencies
```bash
py -3 -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

### 2. Train the Model & Generate `model.pkl`
```bash
python src/train.py
```
*Outputs: `models/model.pkl`*

### 3. Run API Microservice Locally
```bash
uvicorn app.main:app --reload --port 8000
```
- Interactive API Swagger Documentation: `http://localhost:8000/docs`
- Healthcheck endpoint: `http://localhost:8000/health`

### 4. Sample Inference Request
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "score_10th": 88.5,
    "score_12th": 92.0,
    "education_level": "Master",
    "current_job": "Data Scientist",
    "years_of_experience": 4.5
  }'
```

---

## 🐳 Docker Containerization

To build and run the container locally:
```bash
docker build -t mind-eval-model:latest .
docker run -d -p 8000:8000 --name mind-eval-service mind-eval-model:latest
```

---

## ☸️ Manual Kubernetes / AKS Deployment

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

The `.github/workflows/deploy.yml` pipeline automates:
1. Model training & sanity test execution.
2. Building Docker container image.
3. Pushing image to Azure Container Registry (ACR).
4. Authenticating with AKS and applying Kubernetes manifests.

### Required GitHub Repository Secrets:
- `AZURE_CREDENTIALS`: Service Principal JSON output from `az ad sp create-for-rbac`.
- `AZURE_ACR_NAME`: Azure Container Registry name.
- `AZURE_RESOURCE_GROUP`: Azure Resource Group containing AKS.
- `AZURE_AKS_CLUSTER`: Name of the Azure Kubernetes Service cluster.

---

## 🌐 Deploying to Render (render.com)

You can deploy this application directly to Render as a Web Service using the included `render.yaml` blueprint.

### Option 1: Automatic Blueprint Deploy (Recommended)
1. Push this repository to GitHub or GitLab.
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** -> **Blueprints**.
4. Connect your repository. Render will automatically detect `render.yaml` and set up the Web Service.
5. Click **Apply**.

### Option 2: Manual Web Service Setup on Render
1. Click **New +** -> **Web Service**.
2. Connect your Git repository.
3. Configure the following settings:
   - **Environment**: `Python 3` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt && python src/train.py`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
4. Click **Create Web Service**.

