import io
import os
import pytest
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    # Verify Web Interface Serves HTML With Required Start Line Tag
    response = client.get("/")
    assert response.status_code == 200
    html_text = response.data.decode("utf-8")
    assert "23L-0801 A2 Ops" in html_text


def test_health_route(client):
    # Verify Health Endpoint Schema
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["model_name"] == "23L-0801_A2_Ops_visionguard"
    assert data["model_alias"] == "champion"
    assert "database" in data
    assert "uptime_seconds" in data


def test_predict_no_file(client):
    # Verify 400 When File Parameter Is Missing
    response = client.post("/predict")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_predict_valid_sample(client):
    # Verify Full Predict Flow Logs To DB And Returns JSON
    sample_path = os.path.join("samples", "test_ai_01.jpeg")
    assert os.path.exists(sample_path), f"Missing {sample_path}"

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    data = {
        "file": (io.BytesIO(img_bytes), "test_ai_01.jpeg"),
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()

    assert "verdict" in payload
    assert payload["verdict"] in ["AI", "Real"]
    assert "confidence" in payload
    assert "raw_score" in payload
    assert "threshold" in payload
    assert "latency_ms" in payload
    assert payload["latency_ms"] > 0


def test_history_route(client):
    # Verify Prediction Audit History Route
    response = client.get("/history?limit=5")
    assert response.status_code == 200
    history = response.get_json()
    assert isinstance(history, list)
