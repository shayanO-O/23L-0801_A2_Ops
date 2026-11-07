# VisionGuard: AI vs Real Image Classification and Deployment

[![CI](https://github.com/shayanO-O/23L-0801_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/shayanO-O/23L-0801_A2_Ops/actions/workflows/ci.yml)

VisionGuard is an end-to-end machine learning system that classifies whether an image is AI-generated or real. It features an automated data and model versioning pipeline with DVC and MLflow on DagsHub, a containerized Flask web application with PostgreSQL logging, and a cloud-deployed inference API.

## Project Structure
- `src/`: Data processing, audit, EDA, model training, evaluation, and registry scripts.
- `app/`: Flask web application and UI.
- `api/`: Cloud-ready inference API containerized for Hugging Face Spaces.
- `tests/`: Automated test suite for validation and routes.
- `splits/`: Train, validation, and test split manifests.
- `samples/`: Test image samples for evaluation and manual inspection.
- `docker-compose.yml`: Local multi-container deployment (Web App + PostgreSQL).

## Quickstart

### 1. Local Environment
```bash
python -m venv .venv
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run with Docker Compose
```bash
docker compose up -d --build
```
Access the application at `http://localhost:5000`.
