import os
import sys
import time
import io
import hashlib
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_PATH = os.path.join("model", "champion.keras")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.5252"))
API_KEY = os.getenv("API_KEY")
START_TIME = time.time()


@app.before_request
def authenticate_request():
    if request.path in ["/", "/health"]:
        return None
    if API_KEY:
        incoming_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
        if not incoming_key and request.headers.get("Authorization"):
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                incoming_key = auth.split(" ", 1)[1]
        if incoming_key != API_KEY:
            return jsonify({"error": "Unauthorized: Missing or invalid X-API-Key header."}), 401
    return None

# Zero Drift Decoding Op
def decode_and_resize(image_bytes, img_size=(224, 224)):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)
    img.set_shape([img_size[0], img_size[1], 3])
    return img


# Load Model At Startup
print(f"Loading champion model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Champion model loaded successfully.")


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "VisionGuard Cloud Inference API",
        "student_id": "23L-0801",
        "model_name": "23L-0801_A2_Ops_visionguard",
        "model_alias": "champion",
        "model_version": "2",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict (multipart/form-data with 'file')"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    uptime = time.time() - START_TIME
    return jsonify({
        "status": "healthy",
        "model_name": "23L-0801_A2_Ops_visionguard",
        "model_alias": "champion",
        "model_version": "2",
        "uptime_seconds": round(uptime, 2)
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Provide 'file' field."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if not image_bytes or len(image_bytes) == 0:
        return jsonify({"error": "Empty file."}), 400

    if len(image_bytes) > 5 * 1024 * 1024:
        return jsonify({"error": "File size exceeds 5 MB limit."}), 413

    # Validate Image
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            format_name = pil_img.format
            if not format_name or format_name.upper() not in {"JPEG", "PNG", "WEBP"}:
                return jsonify({"error": f"Unsupported format '{format_name}'. Allowed: JPEG, PNG, WEBP."}), 400
            pil_img.verify()
    except Exception as e:
        return jsonify({"error": f"Corrupt image file: {str(e)}"}), 400

    threshold = request.form.get("threshold", type=float)
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    try:
        img_tensor = decode_and_resize(image_bytes, img_size=(224, 224))
        img_batch = tf.expand_dims(img_tensor, axis=0)
        raw_pred = float(model.predict(img_batch, verbose=0)[0][0])
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

    latency_ms = (time.time() - t0) * 1000.0
    verdict = "AI" if raw_pred >= threshold else "Real"
    confidence = raw_pred if raw_pred >= threshold else (1.0 - raw_pred)
    file_hash = hashlib.sha256(image_bytes).hexdigest()

    return jsonify({
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "raw_score": round(raw_pred, 4),
        "threshold": round(threshold, 4),
        "model_version": "2",
        "model_alias": "champion",
        "file_hash_sha256": file_hash,
        "latency_ms": round(latency_ms, 2)
    }), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
