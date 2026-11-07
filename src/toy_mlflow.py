import argparse
import os
import sys
import matplotlib.pyplot as plt

# Opt In To Filesystem Tracking Backend For MLflow 3.x Local File Store
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Configure UTF-8 Output To Prevent Emoji Crashes On Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import tensorflow as tf
from dotenv import load_dotenv

SEED = 42


def get_toy_data():
    # Load Small Fashion-MNIST Sample
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train[:500] / 255.0).astype("float32")
    y_train = y_train[:500]
    x_val = (x_val[:150] / 255.0).astype("float32")
    y_val = y_val[:150]
    return (x_train, y_train), (x_val, y_val)


def build_toy_classifier(learning_rate):
    # Lightweight Classifier For Tracking Experiments
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_toy_loss(history, run_name, output_path):
    # Plot Epoch Loss Metric For Artifact Logging
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss", lw=2)
    plt.plot(history.history["val_loss"], label="Val Loss", lw=2)
    plt.title(f"Loss Curve: {run_name}", weight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_local_toy():
    # Run Two Local Runs With Different Learning Rates
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("toy_local")

    (x_train, y_train), (x_val, y_val) = get_toy_data()
    lrs = [0.01, 0.001]

    for idx, lr in enumerate(lrs, start=1):
        run_name = f"toy_run_lr_{lr}"
        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Starting Local {run_name} ---")
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", 3)
            mlflow.log_param("batch_size", 32)
            mlflow.set_tag("developer", "Shayan Ahmed")
            mlflow.set_tag("stage", "toy_local")

            model = build_toy_classifier(learning_rate=lr)

            class MetricsCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    mlflow.log_metric("train_loss", float(logs.get("loss")), step=epoch + 1)
                    mlflow.log_metric("train_accuracy", float(logs.get("accuracy")), step=epoch + 1)
                    mlflow.log_metric("val_loss", float(logs.get("val_loss")), step=epoch + 1)
                    mlflow.log_metric("val_accuracy", float(logs.get("val_accuracy")), step=epoch + 1)

            hist = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=3,
                batch_size=32,
                callbacks=[MetricsCallback()],
                verbose=1,
            )

            fig_path = f"scratch/toy_loss_lr_{lr}.png"
            plot_toy_loss(hist, run_name, fig_path)
            mlflow.log_artifact(fig_path, artifact_path="plots")
            mlflow.keras.log_model(model, artifact_path="model")
            print(f"Completed {run_name}")


def run_remote_toy():
    # Run Toy Tracking And Model Registration Against DagsHub
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not tracking_uri or not username or not password:
        raise ValueError("Missing MLFLOW_TRACKING_* credentials in environment")

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "23L-0801_A2_Ops"
    mlflow.set_experiment(experiment_name)

    (x_train, y_train), (x_val, y_val) = get_toy_data()
    lr = 0.001
    run_name = "toy_remote_verification"
    registered_name = "23L-0801_A2_Ops_toy_model"

    with mlflow.start_run(run_name=run_name):
        print(f"\n--- Starting Remote {run_name} on DagsHub ---")
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", 2)
        mlflow.set_tag("environment", "dagshub_remote")

        model = build_toy_classifier(learning_rate=lr)
        hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, verbose=1)

        for ep in range(2):
            mlflow.log_metric("train_loss", float(hist.history["loss"][ep]), step=ep + 1)
            mlflow.log_metric("val_loss", float(hist.history["val_loss"][ep]), step=ep + 1)

        fig_path = "scratch/toy_remote_loss.png"
        plot_toy_loss(hist, run_name, fig_path)
        mlflow.log_artifact(fig_path, artifact_path="plots")

        print("Logging and registering model to DagsHub...")
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_name,
        )
        print("Model logged and registered successfully.")

    client = MlflowClient()
    model_obj = client.get_registered_model(registered_name)
    version = model_obj.latest_versions[0].version if model_obj.latest_versions else 1
    print(f"Setting alias 'champion' on version {version}...")
    client.set_registered_model_alias(name=registered_name, alias="champion", version=str(version))

    # De-risk Loading By Alias
    alias_uri = f"models:/{registered_name}@champion"
    print(f"Loading registered model via alias: {alias_uri}")
    loaded = mlflow.keras.load_model(alias_uri)
    preds = loaded.predict(x_val[:1])
    print(f"Prediction by alias test passed. Output shape: {preds.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="Run against DagsHub MLflow")
    args = parser.parse_args()

    if args.remote:
        run_remote_toy()
    else:
        run_local_toy()
