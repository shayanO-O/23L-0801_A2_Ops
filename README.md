# VisionGuard: AI vs Real Image Classification

[![CI](https://github.com/shayanO-O/23L-0801_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/shayanO-O/23L-0801_A2_Ops/actions/workflows/ci.yml)

VisionGuard is an end-to-end MLOps project that classifies whether an image is AI-generated or real. Data and models are versioned with DVC and MLflow on DagsHub, a Flask web app logs predictions to PostgreSQL, and the champion model is served by a separate inference API deployed on AWS EC2.

## Links

| Resource | URL |
|---|---|
| GitHub Repository | https://github.com/shayanO-O/23L-0801_A2_Ops |
| DagsHub Project (DVC + MLflow) | https://dagshub.com/shayanO-O/23L-0801_A2_Ops |
| Live Inference API | http://3.24.214.144 |
| API Health Check | http://3.24.214.144/health |

## Repository Structure

```
23L-0801_A2_Ops/
├── .dvc/                  # DVC configuration
├── .github/workflows/     # CI pipeline
├── api/                   # Cloud inference API (deployed on AWS EC2)
├── app/                   # Flask web app
├── data/                  # DVC-tracked raw dataset pointer
├── model/                 # Local model directory
├── outputs/               # Evaluation manifest
├── samples/               # Sample test images
├── splits/                # Train/val/test split manifests
├── src/                   # Training, evaluation and registry scripts
├── tests/                 # Test suite
├── Dockerfile             # Web app image
├── docker-compose.yml     # Web app + PostgreSQL
├── requirements.txt       # Local development & test dependencies (includes -r requirements-prod.txt)
├── requirements-prod.txt  # Core production & container dependencies (used by Dockerfile & CI)
├── requirements-train.txt # Model training & experiment tracking dependencies
├── pyproject.toml         # Ruff and pytest config
├── conftest.py            # Pytest config
└── .env.example           # Environment template
```

## Setup

### 1. Run the web app (Docker)

No local Python installation or virtual environment is required to run the full application stack (Flask web app + PostgreSQL audit database):

```bash
docker compose up -d --build
```

Open http://localhost:5000 in your browser. To stop the application:

```bash
docker compose down
```

### 2. Run tests and lint

To execute the automated test suite and code quality checks locally, create a virtual environment and install dependencies in one command:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pytest -v tests/
ruff check app/ src/ tests/
```

