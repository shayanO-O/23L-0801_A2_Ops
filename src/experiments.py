import argparse
import hashlib
import json
import os
import subprocess
import sys

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))

# Opt In To Filesystem Tracking Backend For Local MLflow Runs
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import mlflow.keras
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv

from src.data import create_dataset_from_csv
from src.models import build_baseline_cnn, build_transfer, unfreeze_last_block
from src.train import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_FINE_TUNE_EPOCHS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_REDUCE_LR_PATIENCE,
    SEED,
    plot_training_curves,
    set_reproducible_seeds,
    train_run,
)
from src.tuner import run_tuner_search

# Load Environment Configuration
load_dotenv(".env")

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "visionguard_experiments")


def compute_code_commit():
    # Compute Last Git Commit Hash For Training Relevant Code
    watched_files = [
        "src/data.py",
        "src/models.py",
        "src/train.py",
        "src/experiments.py",
        "src/tuner.py",
        "requirements-train.txt",
    ]
    try:
        cmd = ["git", "log", "-1", "--format=%H", "--"] + watched_files
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commit = res.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass
    return "unknown_commit"


def compute_dvc_data_md5():
    # Extract Raw Dataset MD5 From DVC Pointer File
    dvc_path = os.path.join("data", "raw.dvc")
    if os.path.exists(dvc_path):
        try:
            with open(dvc_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "md5:" in line:
                        return line.split("md5:")[1].strip()
        except Exception:
            pass
    return "unknown_dvc_md5"


def compute_config_hash(config_dict):
    # Compute Stable MD5 Hash Of Run Configuration
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def get_existing_run(client, experiment_id, run_name):
    # Locate Latest Finished Non Superseded Run By Name
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and tags.superseded != 'true'",
        order_by=["start_time DESC"],
    )
    for r in runs:
        if r.info.status == "FINISHED":
            return r
    return None


def mark_older_runs_superseded(client, experiment_id, run_name):
    # Mark Any Previous Runs With Identical Name As Superseded
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and tags.superseded != 'true'",
    )
    for r in runs:
        client.set_tag(r.info.run_id, "superseded", "true")


class ManualEpochMetricsCallback(tf.keras.callbacks.Callback):
    # Log Per Epoch Metrics Explicitly To Active MLflow Run
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for k, v in logs.items():
                mlflow.log_metric(k, float(v), step=epoch + 1)


def save_environment_artifacts(artifact_dir):
    # Export Reproducibility Artifacts Like Pip Freeze And Split Lists
    os.makedirs(artifact_dir, exist_ok=True)
    freeze_path = os.path.join(artifact_dir, "requirements_freeze.txt")
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        with open(freeze_path, "w", encoding="utf-8") as f:
            f.write(res.stdout)
    except Exception:
        with open(freeze_path, "w", encoding="utf-8") as f:
            f.write("# Pip Freeze Unavailable\n")

    return freeze_path


def execute_single_run(run_name, model_builder_fn, config, splits_dir="splits", smoke=False):
    # Orchestrate Single MLflow Logged Training Run With Idempotency Check
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    exp_id = experiment.experiment_id

    code_commit = compute_code_commit()
    dvc_md5 = compute_dvc_data_md5()
    cfg_hash = compute_config_hash(config)

    # Idempotency Verification
    existing = get_existing_run(client, exp_id, run_name)
    if existing and existing.info.status == "FINISHED":
        print(f"[IDEMPOTENT SKIP] Run '{run_name}' Already Completed. Skipping.")
        return existing.info.run_id

    print("\n========================================================")
    print(f" Executing Experiment Run: {run_name} (Smoke={smoke})")
    print("========================================================")

    mark_older_runs_superseded(client, exp_id, run_name)

    img_size = config.get("img_size", (224, 224))
    batch_size = config.get("batch_size", 32)
    augment = config.get("augment", True)
    epochs = 1 if smoke else config.get("epochs", DEFAULT_MAX_EPOCHS)
    patience = 1 if smoke else config.get("patience", DEFAULT_EARLY_STOPPING_PATIENCE)

    train_ds = create_dataset_from_csv(
        os.path.join(splits_dir, "train.csv"),
        img_size=img_size,
        batch_size=batch_size,
        is_training=True,
        augment=augment,
        seed=SEED,
    )
    val_ds = create_dataset_from_csv(
        os.path.join(splits_dir, "val.csv"),
        img_size=img_size,
        batch_size=batch_size,
        is_training=False,
        augment=False,
        seed=SEED,
    )

    if smoke:
        train_ds = train_ds.take(1)
        val_ds = val_ds.take(1)

    model = model_builder_fn()

    with mlflow.start_run(run_name=run_name) as active_run:
        run_id = active_run.info.run_id

        # Log Full Configuration Parameters
        mlflow.log_params(config)
        mlflow.log_param("img_size", str(img_size))
        mlflow.log_param("augment", augment)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("seed", SEED)

        # Set Provenance Tags
        mlflow.set_tag("git_commit", code_commit)
        mlflow.set_tag("dvc_data_md5", dvc_md5)
        mlflow.set_tag("config_hash", cfg_hash)
        mlflow.set_tag("seed", str(SEED))
        mlflow.set_tag("run_name", run_name)
        mlflow.set_tag("superseded", "false")

        metric_cb = ManualEpochMetricsCallback()
        checkpoint_dir = os.path.join("scratch", "checkpoints", run_name)
        ckpt_path = os.path.join(checkpoint_dir, "best_model.keras")

        train_cfg = {
            "epochs": epochs,
            "patience": patience,
            "reduce_lr_patience": config.get("reduce_lr_patience", DEFAULT_REDUCE_LR_PATIENCE),
            "checkpoint_path": ckpt_path,
            "seed": SEED,
        }

        history, best_epoch, best_val_loss = train_run(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            config=train_cfg,
            custom_callbacks=[metric_cb],
        )

        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Extract Metrics At Best Epoch
        hist = history.history
        idx = best_epoch - 1
        for k in hist.keys():
            mlflow.log_metric(f"best_{k}", float(hist[k][idx]))

        # Generate And Save Curves
        curves_path = os.path.join("scratch", "figures", f"{run_name}_curves.png")
        plot_training_curves(history, best_epoch, curves_path, title_prefix=run_name)
        mlflow.log_artifact(curves_path, artifact_path="figures")

        # Save Environment Artifacts
        artifacts_temp = os.path.join("scratch", "artifacts", run_name)
        freeze_file = save_environment_artifacts(artifacts_temp)
        mlflow.log_artifact(freeze_file, artifact_path="environment")
        mlflow.log_artifact(os.path.join(splits_dir, "train.csv"), artifact_path="splits")
        mlflow.log_artifact(os.path.join(splits_dir, "val.csv"), artifact_path="splits")

        # Log Trained Keras Model Artifact
        mlflow.keras.log_model(model, artifact_path="model")
        print(f"Run {run_name} Finished Successfully. MLflow Run ID: {run_id}")

        return run_id


def run_stage_runs(smoke=False):
    # Execute Core Model Runs R0 Through R6
    print("\n>>> Launching Stage: Runs (R0 to R6) <<<")

    # R0 Baseline Custom CNN
    execute_single_run(
        run_name="R0_baseline_cnn",
        model_builder_fn=lambda: build_baseline_cnn(learning_rate=1e-3, dropout_rate=0.3),
        config={"architecture": "baseline_cnn", "learning_rate": 1e-3, "dropout": 0.3},
        smoke=smoke,
    )

    # R1 MobileNetV2 Reference (Frozen, LR 1e-3)
    execute_single_run(
        run_name="R1_tl_frozen_lr1e-3",
        model_builder_fn=lambda: build_transfer("mobilenet_v2", learning_rate=1e-3, dropout_rate=0.3),
        config={"architecture": "mobilenet_v2", "learning_rate": 1e-3, "dropout": 0.3, "frozen": True},
        smoke=smoke,
    )

    # R2 MobileNetV2 Smaller Learning Rate (LR 1e-4)
    execute_single_run(
        run_name="R2_tl_frozen_lr1e-4",
        model_builder_fn=lambda: build_transfer("mobilenet_v2", learning_rate=1e-4, dropout_rate=0.3),
        config={"architecture": "mobilenet_v2", "learning_rate": 1e-4, "dropout": 0.3, "frozen": True},
        smoke=smoke,
    )

    # R3 MobileNetV2 Without Augmentation
    execute_single_run(
        run_name="R3_tl_frozen_noaug",
        model_builder_fn=lambda: build_transfer("mobilenet_v2", learning_rate=1e-3, dropout_rate=0.3),
        config={"architecture": "mobilenet_v2", "learning_rate": 1e-3, "dropout": 0.3, "frozen": True, "augment": False},
        smoke=smoke,
    )

    # R4 MobileNetV2 Smaller Image Resolution (160x160)
    execute_single_run(
        run_name="R4_tl_frozen_img160",
        model_builder_fn=lambda: build_transfer("mobilenet_v2", img_size=(160, 160), learning_rate=1e-3, dropout_rate=0.3),
        config={"architecture": "mobilenet_v2", "img_size": (160, 160), "learning_rate": 1e-3, "dropout": 0.3, "frozen": True},
        smoke=smoke,
    )

    # R5 MobileNetV2 Fine-Tuned Last Block
    def build_r5_model():
        base = build_transfer("mobilenet_v2", learning_rate=1e-3, dropout_rate=0.3)
        return unfreeze_last_block(base, fine_tune_lr=1e-4)

    execute_single_run(
        run_name="R5_tl_finetune_last_block",
        model_builder_fn=build_r5_model,
        config={
            "architecture": "mobilenet_v2_finetune",
            "learning_rate": 1e-4,
            "dropout": 0.3,
            "epochs": DEFAULT_FINE_TUNE_EPOCHS,
            "frozen": False,
        },
        smoke=smoke,
    )

    # R6 EfficientNetB0 Frozen Reference
    execute_single_run(
        run_name="R6_effb0_frozen_lr1e-3",
        model_builder_fn=lambda: build_transfer("efficientnetb0", learning_rate=1e-3, dropout_rate=0.3),
        config={"architecture": "efficientnetb0", "learning_rate": 1e-3, "dropout": 0.3, "frozen": True},
        smoke=smoke,
    )


def run_stage_tuner(smoke=False):
    # Execute Keras Tuner Random Search Stage
    print("\n>>> Launching Stage: Tuner (8 Trials) <<<")
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    exp_id = experiment.experiment_id

    code_commit = compute_code_commit()
    dvc_md5 = compute_dvc_data_md5()

    out_csv = os.path.join("report", "tables", "tuner_trials.csv")
    res = run_tuner_search(
        splits_dir="splits",
        output_csv=out_csv,
        seed=SEED,
        smoke=smoke,
    )

    if isinstance(res, tuple):
        best_trial, df_trials = res
    else:
        df_trials = res
        best_trial = df_trials.iloc[0].to_dict()

    mark_older_runs_superseded(client, exp_id, "tuner")

    with mlflow.start_run(run_name="tuner"):
        mlflow.log_params(best_trial)
        mlflow.set_tag("git_commit", code_commit)
        mlflow.set_tag("dvc_data_md5", dvc_md5)
        mlflow.set_tag("seed", str(SEED))
        mlflow.set_tag("run_name", "tuner")
        mlflow.set_tag("superseded", "false")
        mlflow.log_artifact(out_csv, artifact_path="tables")

    print("Tuner Search Finished. Best Configuration:")
    print(best_trial)
    return best_trial


def run_stage_finals(smoke=False):
    # Execute Final Comparison Runs With Best Tuner Settings
    print("\n>>> Launching Stage: Finals (final_manual vs final_autolog) <<<")
    tuner_csv = os.path.join("report", "tables", "tuner_trials.csv")
    if os.path.exists(tuner_csv):
        df_trials = pd.read_csv(tuner_csv)
        best_cfg = df_trials.iloc[0].to_dict()
        best_lr = float(best_cfg.get("learning_rate", 1e-3))
        best_dropout = float(best_cfg.get("dropout", 0.3))
        unfreeze_choice = int(best_cfg.get("unfreeze_blocks", 0))
    else:
        best_lr = 1e-3
        best_dropout = 0.3
        unfreeze_choice = 0

    def build_best_model():
        m = build_transfer("mobilenet_v2", learning_rate=best_lr, dropout_rate=best_dropout)
        if unfreeze_choice == 1:
            m = unfreeze_last_block(m, fine_tune_lr=best_lr / 10.0)
        return m

    # 1. Final Manual Logging
    execute_single_run(
        run_name="final_manual",
        model_builder_fn=build_best_model,
        config={
            "architecture": "mobilenet_v2_best",
            "learning_rate": best_lr,
            "dropout": best_dropout,
            "unfreeze_blocks": unfreeze_choice,
            "logging_type": "manual",
        },
        smoke=smoke,
    )

    # 2. Final Autolog Run
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()
    exp_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    mark_older_runs_superseded(client, exp_id, "final_autolog")

    code_commit = compute_code_commit()
    dvc_md5 = compute_dvc_data_md5()
    epochs = 1 if smoke else DEFAULT_MAX_EPOCHS

    train_ds = create_dataset_from_csv(
        os.path.join("splits", "train.csv"),
        is_training=True,
        augment=True,
        seed=SEED,
    )
    val_ds = create_dataset_from_csv(
        os.path.join("splits", "val.csv"),
        is_training=False,
        augment=False,
        seed=SEED,
    )

    if smoke:
        train_ds = train_ds.take(1)
        val_ds = val_ds.take(1)

    set_reproducible_seeds(SEED)
    model_autolog = build_best_model()

    # Enable MLflow Autologging
    try:
        mlflow.tensorflow.autolog(log_models=True, disable=False)
    except Exception:
        mlflow.keras.autolog(log_models=True)

    with mlflow.start_run(run_name="final_autolog"):
        mlflow.set_tag("git_commit", code_commit)
        mlflow.set_tag("dvc_data_md5", dvc_md5)
        mlflow.set_tag("seed", str(SEED))
        mlflow.set_tag("run_name", "final_autolog")
        mlflow.set_tag("logging_type", "autolog")
        mlflow.set_tag("superseded", "false")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1 if smoke else DEFAULT_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ]
        model_autolog.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        try:
            mlflow.keras.log_model(model_autolog, artifact_path="model")
        except Exception:
            pass

    # Disable Autolog Afterwards To Prevent Spillover
    try:
        mlflow.tensorflow.autolog(disable=True)
    except Exception:
        pass

    print("Final Runs Completed Successfully.")


def main():
    global EXPERIMENT_NAME
    parser = argparse.ArgumentParser(description="VisionGuard Model Training Orchestrator")
    parser.add_argument(
        "--stage",
        choices=["runs", "tuner", "finals", "all"],
        default="all",
        help="Training stage to execute",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke test with 1 batch and 1 epoch",
    )
    args = parser.parse_args()

    if args.smoke:
        os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
        mlflow.set_tracking_uri("file:./mlruns")
        EXPERIMENT_NAME = "smoke_local"

    if args.stage in ["runs", "all"]:
        run_stage_runs(smoke=args.smoke)
    if args.stage in ["tuner", "all"]:
        run_stage_tuner(smoke=args.smoke)
    if args.stage in ["finals", "all"]:
        run_stage_finals(smoke=args.smoke)


if __name__ == "__main__":
    main()
