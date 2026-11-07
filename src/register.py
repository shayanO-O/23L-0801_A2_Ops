import os
import sys

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv(".env")

REGISTERED_MODEL_NAME = "23L-0801_A2_Ops_visionguard"


def register_models():
    # Register Champion And Baseline Transfer Models In DagsHub Model Registry
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not set in environment")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    exp = client.get_experiment_by_name("23L-0801_A2_Ops")
    if not exp:
        raise ValueError("Experiment '23L-0801_A2_Ops' not found on DagsHub")

    runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=100)

    # 1. Locate Best Pre-Tuning Run (R6) And Final Manual Run
    r6_runs = [r for r in runs if r.data.tags.get("mlflow.runName") == "R6_effb0_frozen_lr1e-3" and r.info.status == "FINISHED"]
    manual_runs = [r for r in runs if r.data.tags.get("mlflow.runName") == "final_manual" and r.info.status == "FINISHED"]

    if not r6_runs:
        raise ValueError("Could not find FINISHED run for R6_effb0_frozen_lr1e-3")
    if not manual_runs:
        raise ValueError("Could not find FINISHED run for final_manual")

    r6_id = r6_runs[0].info.run_id
    manual_id = manual_runs[0].info.run_id

    print(f"Found Pre-tuning Best Run (R6): {r6_id}")
    print(f"Found Final Manual Run: {manual_id}")

    # Ensure Registered Model Exists
    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        print(f"Creating registered model '{REGISTERED_MODEL_NAME}'...")
        client.create_registered_model(REGISTERED_MODEL_NAME)

    # Check Existing Versions To Prevent Duplicate Creation
    reg = client.get_registered_model(REGISTERED_MODEL_NAME)
    existing_versions = {v.run_id: v.version for v in reg.latest_versions}

    # Register Model Version 1: Best of R1-R6 (R6)
    if r6_id not in existing_versions:
        print(f"\nRegistering Version 1 for R6 (Run ID: {r6_id})...")
        v1 = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=f"runs:/{r6_id}/model",
            run_id=r6_id,
            description="Best Pre-tuning Model: EfficientNetB0 Frozen Reference (Val ROC-AUC: 0.8157)",
        )
        print(f"Registered Version {v1.version} under {REGISTERED_MODEL_NAME}")
    else:
        print(f"Version for R6 already registered as Version {existing_versions[r6_id]}")

    # Register Model Version 2: final_manual (Champion)
    if manual_id not in existing_versions:
        print(f"\nRegistering Version 2 for final_manual (Run ID: {manual_id})...")
        v2 = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=f"runs:/{manual_id}/model",
            run_id=manual_id,
            description="Champion Model: Tuned MobileNetV2 with Unfrozen Block (Val ROC-AUC: 0.8420, Val F1: 0.8000)",
        )
        v2_version = v2.version
        print(f"Registered Version {v2_version} under {REGISTERED_MODEL_NAME}")
    else:
        v2_version = existing_versions[manual_id]
        print(f"Version for final_manual already registered as Version {v2_version}")

    # Assign Alias 'champion' To Version 2
    print(f"\nAssigning alias 'champion' to Version {v2_version}...")
    client.set_registered_model_alias(name=REGISTERED_MODEL_NAME, alias="champion", version=str(v2_version))

    # Verify Registration State
    reg = client.get_registered_model(REGISTERED_MODEL_NAME)
    print(f"\n=== REGISTERED MODEL SUMMARY: {reg.name} ===")
    print(f"Aliases: {reg.aliases}")
    for ver in reg.latest_versions:
        print(f" - Version {ver.version} (Run ID: {ver.run_id}, Status: {ver.status})")

    print("\nModel Registration Completed Successfully.")
    return reg


if __name__ == "__main__":
    register_models()
