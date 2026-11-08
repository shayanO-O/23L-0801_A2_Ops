import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SEED = 42
DEFAULT_MAX_EPOCHS = 20
DEFAULT_EARLY_STOPPING_PATIENCE = 3
DEFAULT_REDUCE_LR_PATIENCE = 1
DEFAULT_FINE_TUNE_EPOCHS = 10
SMOKE_EPOCHS = 1


def set_reproducible_seeds(seed=SEED):
    # Enforce Deterministic Random Seeds Across Libraries
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)


def plot_training_curves(history, best_epoch, output_path, title_prefix="Training"):
    # Plot Loss And Accuracy Curves With Highlighted Best Epoch
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Curves
    ax1.plot(epochs_range, hist["loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs_range, hist["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax1.axvline(
        x=best_epoch,
        color="crimson",
        linestyle="--",
        label=f"Best Epoch ({best_epoch})",
    )
    ax1.scatter([best_epoch], [hist["val_loss"][best_epoch - 1]], color="crimson", s=80, zorder=5)
    ax1.set_title(f"{title_prefix}: Loss", weight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy Curves
    ax2.plot(epochs_range, hist["accuracy"], label="Train Accuracy", color="#1f77b4", lw=2)
    ax2.plot(epochs_range, hist["val_accuracy"], label="Val Accuracy", color="#ff7f0e", lw=2)
    ax2.axvline(
        x=best_epoch,
        color="crimson",
        linestyle="--",
        label=f"Best Epoch ({best_epoch})",
    )
    ax2.scatter([best_epoch], [hist["val_accuracy"][best_epoch - 1]], color="crimson", s=80, zorder=5)
    ax2.set_title(f"{title_prefix}: Accuracy", weight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_run(model, train_ds, val_ds, config, custom_callbacks=None):
    # Execute Model Training With Early Stopping And LR Reduction
    set_reproducible_seeds(config.get("seed", SEED))

    epochs = config.get("epochs", DEFAULT_MAX_EPOCHS)
    patience = config.get("patience", DEFAULT_EARLY_STOPPING_PATIENCE)
    reduce_lr_patience = config.get("reduce_lr_patience", DEFAULT_REDUCE_LR_PATIENCE)
    checkpoint_path = config.get("checkpoint_path", None)

    callbacks_list = []
    if custom_callbacks:
        callbacks_list.extend(custom_callbacks)

    callbacks_list.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    )
    callbacks_list.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        )
    )

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        callbacks_list.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )

    # Determine Best Performing Epoch By Validation Loss
    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1

    # Save Plots
    run_name = config.get("run_name", "run")
    curves_path = config.get("curves_path", f"report/figures/{run_name}_curves.png")
    plot_training_curves(history, best_epoch, curves_path, title_prefix=run_name)

    best_val_loss = float(val_losses[best_epoch - 1])
    return history, best_epoch, best_val_loss
