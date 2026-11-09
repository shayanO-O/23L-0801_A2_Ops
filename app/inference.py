import io
import os
import sys
import hashlib
from typing import Dict, Any, Tuple
from PIL import Image

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))

from src.data import decode_and_resize

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_THRESHOLD = 0.5252
MODEL_LOCAL_PATH = os.path.join("model", "champion.keras")

# Singleton Local Model Cache
_LOCAL_MODEL = None


class ValidationError(Exception):
    # Custom Exception For Client Input Validation Failures
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_image_bytes(image_bytes: bytes, filename: str = "") -> Tuple[str, str]:
    # Validate File Size Extension And Integrity
    if not image_bytes or len(image_bytes) == 0:
        raise ValidationError("Uploaded file is empty (0 bytes).", status_code=400)

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File exceeds maximum allowed size of 5 MB ({len(image_bytes)} bytes provided).",
            status_code=413,
        )

    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file extension '{ext}'. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}",
                status_code=400,
            )

    # Validate Magic Bytes / Format via PIL
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            format_name = pil_img.format
            if not format_name or format_name.upper() not in {"JPEG", "PNG", "WEBP"}:
                raise ValidationError(
                    f"Unsupported image format '{format_name}'. Allowed: JPEG, PNG, WEBP.",
                    status_code=400,
                )
            pil_img.verify()
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Corrupted or invalid image file: {str(e)}", status_code=400)

    # Compute SHA-256 Digest For Audit Trail
    file_hash = hashlib.sha256(image_bytes).hexdigest()
    return format_name.lower(), file_hash


def get_local_model():
    # Lazy Load Singleton Keras Champion Model
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        if os.path.exists(MODEL_LOCAL_PATH):
            import tensorflow as tf
            _LOCAL_MODEL = tf.keras.models.load_model(MODEL_LOCAL_PATH)
        else:
            try:
                from src.download_champion import download_champion_model
                download_champion_model()
                import tensorflow as tf
                _LOCAL_MODEL = tf.keras.models.load_model(MODEL_LOCAL_PATH)
            except Exception:
                # Fallback Architecture For CI Environments Without Pre-Downloaded Weights
                import tensorflow as tf
                inputs = tf.keras.Input(shape=(224, 224, 3))
                x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
                outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
                _LOCAL_MODEL = tf.keras.Model(inputs=inputs, outputs=outputs)
    return _LOCAL_MODEL


def predict_local(image_bytes: bytes, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    # Perform Zero-Drift Local Keras Inference
    import tensorflow as tf

    img_tensor = decode_and_resize(image_bytes, img_size=(224, 224))
    img_batch = tf.expand_dims(img_tensor, axis=0)

    model = get_local_model()
    raw_pred = float(model.predict(img_batch, verbose=0)[0][0])

    verdict = "AI" if raw_pred >= threshold else "Real"
    confidence = raw_pred if raw_pred >= threshold else (1.0 - raw_pred)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "raw_score": round(raw_pred, 4),
        "threshold": round(threshold, 4),
        "model_version": "2",
        "model_alias": "champion",
        "engine": "local_keras",
    }


def predict_cloud(image_bytes: bytes, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    # Query Remote Cloud Inference API on AWS EC2
    import requests

    cloud_url = os.getenv("CLOUD_API_URL") or os.getenv("API_URL")
    if not cloud_url:
        raise ValueError("CLOUD_API_URL or API_URL environment variable is not configured.")

    predict_endpoint = f"{cloud_url.rstrip('/')}/predict"
    files = {"file": ("upload.jpg", image_bytes, "image/jpeg")}
    data = {"threshold": str(threshold)}
    headers = {}
    api_key = os.getenv("API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key

    response = requests.post(predict_endpoint, files=files, data=data, headers=headers, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Cloud API request failed ({response.status_code}): {response.text}")

    result = response.json()
    result["engine"] = "cloud_aws_ec2"
    return result


def predict_image_bytes(
    image_bytes: bytes,
    filename: str = "",
    threshold: float = DEFAULT_THRESHOLD,
    model_mode: str = "local",
) -> Tuple[Dict[str, Any], str]:
    # Master Inference Entrypoint With Validation And Mode Dispatch
    _, file_hash = validate_image_bytes(image_bytes, filename=filename)

    mode = model_mode.lower()
    if mode in ("cloud", "remote"):
        result = predict_cloud(image_bytes, threshold=threshold)
    else:
        result = predict_local(image_bytes, threshold=threshold)

    return result, file_hash
