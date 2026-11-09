# VisionGuard: AI vs Real Image Classification & MLOps Infrastructure

[![CI](https://github.com/shayanO-O/23L-0801_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/shayanO-O/23L-0801_A2_Ops/actions/workflows/ci.yml)

VisionGuard is a production-grade, end-to-end MLOps system that detects and classifies whether an uploaded image is AI-generated or authentic real-world photography. The platform integrates reproducible data and model versioning via DVC and MLflow hosted on DagsHub, a containerized multi-service Flask web dashboard backed by PostgreSQL for audit trail tracking, automated GitHub Actions CI testing, and an independently deployed cloud inference microservice hosted on AWS EC2.

---

## Project Links

| Resource | URL |
|---|---|
| **GitHub Repository** | [https://github.com/shayanO-O/23L-0801_A2_Ops](https://github.com/shayanO-O/23L-0801_A2_Ops) |
| **DagsHub Project** | [https://dagshub.com/shayanO-O/23L-0801_A2_Ops](https://dagshub.com/shayanO-O/23L-0801_A2_Ops) |
| **Live Inference API** | [http://3.24.214.144](http://3.24.214.144) |
| **API Health Check** | [http://3.24.214.144/health](http://3.24.214.144/health) |

---

## Architecture

```
                                  +------------------------------------+
                                  |         End-User Browser           |
                                  +-----------------+------------------+
                                                    | HTTP (Port 5000)
                                                    v
+----------------------------- Docker Compose Environment -----------------------------+
|                                                                                      |
|  +--------------------------------+               +-------------------------------+  |
|  |     Flask Web App Container    | -- Logs to -> |     PostgreSQL 15 Container   |  |
|  |  (Upload UI, Dashboard, CAM)   |               |     (Audit Trail Database)    |  |
|  +---------------+----------------+               +-------------------------------+  |
|                  |                                                                   |
+------------------|-------------------------------------------------------------------+
                   | HTTP POST /predict (x-api-key authenticated)
                   v
+--------------------------------------------------------------------------------------+
|                           AWS EC2 Cloud Microservice                                 |
|  +--------------------------------------------------------------------------------+  |
|  |  Docker Container (Port 80) -> Gunicorn -> Flask Inference API                 |  |
|  |  Loads Champion MobileNetV2 Architecture & Preprocessing Pipeline              |  |
|  +--------------------------------------------------------------------------------+  |
+--------------------------------------------------------------------------------------+

Remote Tracking Infrastructure (DagsHub):
- DVC Remote: S3-compatible remote storage for versioned raw datasets and artifacts.
- MLflow Tracking: Centralized experiment runs, hyperparameter sweeps, and Champion Model Registry.
```

---

## Repository Structure

```
23L-0801_A2_Ops/
├── .dvc/                      # DVC configuration and tracking metadata
├── .github/workflows/
│   └── ci.yml                 # Multi-python matrix CI pipeline (lint, test, docker build)
├── api/                       # Standalone cloud inference microservice
│   ├── app.py                 # Lightweight Flask inference service (/health, /predict)
│   ├── Dockerfile             # Production container definition for cloud deployment
│   ├── requirements.txt       # Cloud API dependencies
│   └── model/
│       └── champion.keras     # Champion model weights package
├── app/                       # Front-end Flask web application
│   ├── app.py                 # Web routes, prediction handlers, and DB logging
│   ├── static/                # Stylesheets (Linear-inspired aesthetic)
│   └── templates/             # Jinja2 templates (upload, dashboard, audit trail)
├── data/
│   └── raw.dvc                # DVC pointer tracking raw dataset on DagsHub storage
├── model/
│   └── .gitkeep               # Model directory placeholder (model tracked via DVC/registry)
├── outputs/
│   └── test_done.json         # Evaluation manifest and benchmark metrics
├── samples/                   # Sample evaluation images (AI & Real)
├── splits/                    # Deterministic split manifests (train, val, test CSVs)
├── src/                       # Core ML pipeline and automation scripts
│   ├── audit.py               # Dataset integrity and format audit
│   ├── aws_deploy.py          # AWS EC2 remote deployment orchestrator
│   ├── aws_setup_host.py      # AWS EC2 provisioning and security group configurator
│   ├── check_versions.py      # Environment dependency verifier
│   ├── data.py                # tf.data input pipeline and augmentations
│   ├── download_champion.py   # Registry model downloader
│   ├── eda.py                 # Exploratory data analysis generator
│   ├── errors.py              # Error analysis and misclassification inspection
│   ├── evaluate.py            # Comprehensive test evaluation pipeline
│   ├── experiments.py         # Multi-architecture experiment runner
│   ├── gradcam.py             # Grad-CAM visual explainability heatmap generator
│   ├── make_splits.py         # Stratified dataset split generator
│   ├── models.py              # CNN and MobileNetV2 architecture definitions
│   ├── predict_by_alias.py    # Registry alias inference utility
│   ├── register.py            # MLflow champion promotion utility
│   ├── show_batch.py          # Batch visualization utility
│   ├── show_summary.py        # Model architecture inspector
│   ├── toy_mlflow.py          # DagsHub MLflow remote connectivity test
│   ├── train.py               # Champion training pipeline
│   └── tuner.py               # Hyperparameter optimization sweep
├── tests/                     # Unit and integration test suite
│   ├── test_inference.py      # Model inference verification tests
│   └── test_routes.py         # Web application and API route tests
├── .dockerignore              # Docker build exclusions
├── .dvcignore                 # DVC tracking exclusions
├── .env.example               # Example environment configuration template
├── .gitignore                 # Git ignore rules (scratch, credentials, caches)
├── conftest.py                # Pytest configuration and shared fixtures
├── docker-compose.yml         # Multi-container orchestration (Web App + Postgres)
├── Dockerfile                 # Root production Dockerfile for web application
├── pyproject.toml             # Ruff and tool configuration
├── requirements-prod.txt      # Production runtime dependencies
└── requirements.txt           # Base project requirements
```

---

## Quickstart

### 1. Clone & Setup Environment
```bash
git clone https://github.com/shayanO-O/23L-0801_A2_Ops.git
cd 23L-0801_A2_Ops

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
Copy the example environment configuration template:
```bash
cp .env.example .env
```
Update `.env` with your desired configuration or credentials (e.g. database credentials, API keys).

### 3. Launch via Docker Compose
Build and run the multi-container application (Flask Web Application + PostgreSQL DB):
```bash
docker compose up -d --build
```
Access the application dashboard at: `http://localhost:5000`

To tear down the containers:
```bash
docker compose down
```

---

## Reproducing the ML Pipeline

To reproduce the complete pipeline from raw data to registered model:

1. **Pull Dataset via DVC**:
   ```bash
   dvc pull
   ```
2. **Audit Dataset Integrity**:
   ```bash
   python src/audit.py
   ```
3. **Generate Exploratory Data Analysis (EDA)**:
   ```bash
   python src/eda.py
   ```
4. **Create Deterministic Splits**:
   ```bash
   python src/make_splits.py
   ```
5. **Train Champion Model**:
   ```bash
   python src/train.py
   ```
6. **Evaluate Model Performance**:
   ```bash
   python src/evaluate.py
   ```
7. **Register Champion Model in MLflow**:
   ```bash
   python src/register.py
   ```

---

## Tests & Continuous Integration

### Local Testing
Run the automated test suite locally:
```bash
pytest -v tests/
```

### Continuous Integration (CI) Pipeline
Every push and pull request triggers GitHub Actions CI (`.github/workflows/ci.yml`):
- **Matrix Testing**: Executes automated test suites across **Python 3.10** and **Python 3.11**.
- **Code Quality**: Enforces strict linting and code hygiene via `ruff`.
- **Docker Verification**: Validates container image build reliability in a clean Linux runner.

---

## Cloud Deployment (Task 7)

The production inference microservice is deployed on **AWS EC2**:
- **Host**: AWS EC2 `t3.small` instance running Ubuntu 24.04 LTS.
- **Microservice**: Containerized Flask API served with Gunicorn on port **80**.
- **Security**: Inbound traffic secured via AWS Security Groups; API endpoints are protected by API key authentication (`x-api-key` header).
- **Live Endpoint**: `http://3.24.214.144`
- **Health Check**: `http://3.24.214.144/health`

---

## Assignment Information
- **Student Roll Number**: `23L-0801`
- **Course**: MLOps (CS / AI)
- **Assignment**: Assignment 2: End-to-End MLOps Pipeline & Production Deployment
