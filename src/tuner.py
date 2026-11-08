import os
import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from src.data import create_dataset_from_csv
from src.models import build_transfer, unfreeze_last_block

SEED = 42
MAX_TRIALS = 8
TUNER_EPOCHS = 5
TUNER_PATIENCE = 2


def build_tunable_model(hp):
    # Hyperparameter Search Space For Transfer Model
    lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    dropout = hp.Float("dropout", min_value=0.2, max_value=0.5, step=0.1)
    unfreeze_choice = hp.Choice("unfreeze_blocks", values=[0, 1])

    model = build_transfer(
        backbone_name="mobilenet_v2",
        img_size=(224, 224),
        dropout_rate=dropout,
        learning_rate=lr,
    )

    if unfreeze_choice == 1:
        model = unfreeze_last_block(model, fine_tune_lr=lr / 10.0)

    return model


def run_tuner_search(
    splits_dir="splits",
    output_csv="report/tables/tuner_trials.csv",
    seed=SEED,
    smoke=False,
):
    # Execute Keras Tuner RandomSearch Over 8 Trials
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    train_ds = create_dataset_from_csv(
        os.path.join(splits_dir, "train.csv"),
        is_training=True,
        augment=True,
        batch_size=32,
    )
    val_ds = create_dataset_from_csv(
        os.path.join(splits_dir, "val.csv"),
        is_training=False,
        batch_size=32,
    )

    if smoke:
        train_ds = train_ds.take(1)
        val_ds = val_ds.take(1)
        trials = 2
        epochs = 1
    else:
        trials = MAX_TRIALS
        epochs = TUNER_EPOCHS

    tuner = kt.RandomSearch(
        build_tunable_model,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=trials,
        executions_per_trial=1,
        directory="scratch/kt_visionguard",
        project_name="visionguard_search",
        overwrite=True,
        seed=seed,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=TUNER_PATIENCE,
            restore_best_weights=True,
        )
    ]

    print(f"\n--- Starting VisionGuard Keras Tuner Search ({trials} Trials) ---")
    tuner.search(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)

    # Compile Results Table For All Trials
    all_trials = tuner.oracle.get_best_trials(num_trials=trials)
    records = []
    for rank, trial in enumerate(all_trials, start=1):
        vals = trial.hyperparameters.values
        records.append(
            {
                "rank": rank,
                "trial_id": trial.trial_id,
                "learning_rate": vals.get("learning_rate"),
                "dropout": vals.get("dropout"),
                "unfreeze_blocks": vals.get("unfreeze_blocks"),
                "val_loss": trial.score,
                "val_accuracy": trial.metrics.get_best_value("val_accuracy"),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nTuner results table saved to {output_csv}")
    print(df.to_string(index=False))

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_params = {
        "learning_rate": best_hp.get("learning_rate"),
        "dropout": best_hp.get("dropout"),
        "unfreeze_blocks": best_hp.get("unfreeze_blocks"),
    }
    print(f"\nBest Hyperparameters Found: {best_params}\n")
    return best_params, df


if __name__ == "__main__":
    run_tuner_search()
