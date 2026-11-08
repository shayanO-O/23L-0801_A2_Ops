import os
import pytest
from app.inference import predict_image_bytes, DEFAULT_THRESHOLD


def test_inference_ai_sample():
    # Test Local Inference On Known AI Sample Image
    sample_path = os.path.join("samples", "test_ai_01.jpeg")
    assert os.path.exists(sample_path), f"Sample image missing at {sample_path}"

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    result, file_hash = predict_image_bytes(
        img_bytes,
        filename="test_ai_01.jpeg",
        threshold=DEFAULT_THRESHOLD,
        model_mode="local",
    )

    assert result["verdict"] in ["AI", "Real"]
    assert 0.0 <= result["raw_score"] <= 1.0
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["threshold"] == DEFAULT_THRESHOLD
    assert result["model_alias"] == "champion"
    assert result["model_version"] == "2"
    assert len(file_hash) == 64  # SHA-256 hex length


def test_inference_real_sample():
    # Test Local Inference On Known Real Sample Image
    sample_path = os.path.join("samples", "test_real_01.jpg")
    assert os.path.exists(sample_path), f"Sample image missing at {sample_path}"

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    result, file_hash = predict_image_bytes(
        img_bytes,
        filename="test_real_01.jpg",
        threshold=DEFAULT_THRESHOLD,
        model_mode="local",
    )

    assert result["verdict"] in ["AI", "Real"]
    assert 0.0 <= result["raw_score"] <= 1.0
    assert 0.5 <= result["confidence"] <= 1.0
    assert len(file_hash) == 64
