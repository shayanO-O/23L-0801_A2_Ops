import os
import keras_tuner as kt
import pandas as pd
import tensorflow as tf

SEED = 42


def build_toy_model(hp):
    # Parameterized Multi-Layer Perceptron For Hyperparameter Search
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Search Space: Units In Hidden Layer
    units = hp.Int("units", min_value=32, max_value=128, step=32)
    model.add(tf.keras.layers.Dense(units=units, activation="relu"))

    # Search Space: Learning Rate
    lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_toy_tuner(output_csv="report/tables/toy_tuner.csv"):
    # Run Quick Tuner Search On Fashion-MNIST Subset
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    # Load Fashion-MNIST
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    # Normalize And Subset For Speed
    x_train = x_train[:600] / 255.0
    y_train = y_train[:600]
    x_val = x_val[:200] / 255.0
    y_val = y_val[:200]

    tuner = kt.RandomSearch(
        build_toy_model,
        objective="val_accuracy",
        max_trials=4,
        executions_per_trial=1,
        directory="scratch/kt_toy",
        project_name="fashion_mnist_tuning",
        overwrite=True,
        seed=SEED,
    )

    tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val), verbose=1)

    trials = tuner.oracle.get_best_trials(num_trials=4)
    records = []
    for rank, trial in enumerate(trials, start=1):
        records.append(
            {
                "rank": rank,
                "trial_id": trial.trial_id,
                "units": trial.hyperparameters.values.get("units"),
                "learning_rate": trial.hyperparameters.values.get("learning_rate"),
                "val_accuracy": trial.score,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nToy Tuner results saved to {output_csv}")
    print(df.to_string(index=False))

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"\nBest Settings -> Units: {best_hp.get('units')}, LR: {best_hp.get('learning_rate')}")


if __name__ == "__main__":
    run_toy_tuner()
