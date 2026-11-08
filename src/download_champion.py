import os
import sys

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv(".env")

REGISTERED_MODEL_NAME = "23L-0801_A2_Ops_visionguard"
ALIAS = "champion"
LOCAL_MODEL_PATH = os.path.join("model", "champion.keras")


def download_champion_model():
    # Download Champion Model From Remote Registry Using Champion Alias
    print(f"Querying Model Registry for '{REGISTERED_MODEL_NAME}@{ALIAS}'...")
    client = MlflowClient()

    try:
        model_ver = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)
    except Exception as e:
        print(f"Error fetching model by alias '{ALIAS}': {e}")
        return None

    run_id = model_ver.run_id
    version = model_ver.version
    print(f"Resolved Alias '{ALIAS}' -> Version {version} (Run ID: {run_id})")

    os.makedirs("model", exist_ok=True)

    # Download Artifact From Run
    artifacts = client.list_artifacts(run_id, path="model")
    artifact_paths = [a.path for a in artifacts]
    print(f"Artifacts available under 'model/': {artifact_paths}")

    target_artifact = None
    if "model/champion.keras" in artifact_paths:
        target_artifact = "model/champion.keras"
    elif "model/latest_checkpoint.h5" in artifact_paths:
        target_artifact = "model/latest_checkpoint.h5"
    else:
        # Fallback Search
        for a in artifacts:
            if a.path.endswith(".keras") or a.path.endswith(".h5"):
                target_artifact = a.path
                break

    if not target_artifact:
        raise FileNotFoundError(f"No trained model weights found in artifacts of run {run_id}")

    print(f"Downloading artifact '{target_artifact}' from run {run_id}...")
    downloaded_file = client.download_artifacts(run_id, target_artifact, dst_path="scratch")
    print(f"Downloaded artifact to: {downloaded_file}")

    # Copy To Local Destination
    if downloaded_file.endswith(".h5") and not LOCAL_MODEL_PATH.endswith(".h5"):
        import tensorflow as tf
        m = tf.keras.models.load_model(downloaded_file)
        m.save(LOCAL_MODEL_PATH)
    else:
        import shutil
        shutil.copyfile(downloaded_file, LOCAL_MODEL_PATH)

    print(f"Successfully saved standalone champion model to {LOCAL_MODEL_PATH} ({os.path.getsize(LOCAL_MODEL_PATH)} bytes)")
    return LOCAL_MODEL_PATH


if __name__ == "__main__":
    download_champion_model()
