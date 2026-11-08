import os
import sys
import time
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))
load_dotenv(".env")

from app.db import insert_prediction, get_recent_predictions, get_db_health, init_db
from app.inference import predict_image_bytes, ValidationError, DEFAULT_THRESHOLD

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit for Flask request body

START_TIME = time.time()
MODEL_MODE = (os.getenv("MODEL_MODE") or os.getenv("INFERENCE_MODE") or "local").lower()


@app.route("/", methods=["GET"])
def index():
    # Render Interactive Inspection Dashboard
    history = get_recent_predictions(limit=15)
    return render_template("index.html", history=history, model_mode=MODEL_MODE)


@app.route("/predict", methods=["POST"])
def predict():
    # Handle File Upload Input Validation Inference And Audit Trail
    t0 = time.time()

    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Provide 'file' form-data field."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected for upload."}), 400

    filename = file.filename
    image_bytes = file.read()

    threshold = request.form.get("threshold", type=float)
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    mode = request.form.get("mode", MODEL_MODE).lower()

    try:
        result, file_hash = predict_image_bytes(
            image_bytes=image_bytes,
            filename=filename,
            threshold=threshold,
            model_mode=mode,
        )
    except ValidationError as ve:
        return jsonify({"error": ve.message}), ve.status_code
    except Exception as e:
        return jsonify({"error": f"Inference pipeline failure: {str(e)}"}), 500

    latency_ms = (time.time() - t0) * 1000.0

    # Persist In Audit Database
    rec_id = insert_prediction(
        filename=filename,
        file_hash=file_hash,
        verdict=result["verdict"],
        confidence=result["confidence"],
        raw_score=result["raw_score"],
        threshold=result["threshold"],
        model_mode=mode,
        latency_ms=round(latency_ms, 2),
    )

    response_payload = {
        "id": rec_id,
        "filename": filename,
        "file_hash_sha256": file_hash,
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "raw_score": result["raw_score"],
        "threshold": result["threshold"],
        "model_version": result.get("model_version", "2"),
        "model_alias": result.get("model_alias", "champion"),
        "model_mode": mode,
        "latency_ms": round(latency_ms, 2),
    }

    return jsonify(response_payload), 200


@app.route("/health", methods=["GET"])
def health():
    # Query Application Model And Database Health Endpoint
    db_health = get_db_health()
    uptime = time.time() - START_TIME

    status_code = 200 if db_health.get("status") == "healthy" else 503

    return (
        jsonify(
            {
                "status": "healthy" if status_code == 200 else "degraded",
                "model_name": "23L-0801_A2_Ops_visionguard",
                "model_alias": "champion",
                "model_version": "2",
                "model_mode": MODEL_MODE,
                "database": db_health,
                "uptime_seconds": round(uptime, 2),
            }
        ),
        status_code,
    )


@app.route("/history", methods=["GET"])
def history():
    # Return Recent Prediction Audit Records
    limit = request.args.get("limit", default=20, type=int)
    records = get_recent_predictions(limit=limit)
    return jsonify(records), 200


if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
