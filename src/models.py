import pandas as pd
import tensorflow as tf
from tensorflow.keras import applications, layers, models

# Common Compilation Metric Setup
METRICS = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]


def build_baseline_cnn(input_shape=(224, 224, 3), learning_rate=1e-3, dropout_rate=0.3):
    # Construct Custom Convolutional Architecture
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Internal Normalization To [0, 1] Range
    x = layers.Rescaling(1.0 / 255.0, name="rescaling_norm")(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_transfer(
    backbone_name="mobilenet_v2",
    img_size=(224, 224),
    dropout_rate=0.3,
    learning_rate=1e-3,
):
    # Construct Transfer Learning Model With Pretrained Feature Extractor
    input_shape = (img_size[0], img_size[1], 3)
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Backbone Specific In-Model Preprocessing
    if backbone_name.lower() == "mobilenet_v2":
        # Maps [0, 255] To [-1, 1]
        x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_scaling")(inputs)
        base_model = applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    elif backbone_name.lower() == "efficientnetb0":
        # EfficientNet Has Built In Rescaling
        x = inputs
        base_model = applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Freeze Backbone For Feature Extraction
    base_model.trainable = False

    # Inference Mode During Training Keeps BatchNorm Statistics Stable
    features = base_model(x, training=False)

    # Custom Top Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(features)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"transfer_{backbone_name.lower()}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_last_block(model, fine_tune_lr=1e-5):
    # Unfreeze Only The Final Convolutional Block While Keeping BatchNorm Frozen
    # Locate The Base Pretrained Model Layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find nested base model layer for fine-tuning")

    base_model.trainable = True

    # Identify Name Of Backbone
    base_name = base_model.name.lower()
    if "mobilenet" in base_name:
        # MobileNetV2 Final Inverted Residual Block Starts At 'block_16'
        fine_tune_from = "block_16_expand"
    else:
        # Default To Top 20 Layers
        fine_tune_from = base_model.layers[-20].name

    unfreeze = False
    for layer in base_model.layers:
        if layer.name == fine_tune_from:
            unfreeze = True
        # Keep BatchNorm Layers Frozen To Preserve Population Mean/Variance
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = unfreeze

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def list_trainable_layers(model):
    # Enumerate Trainable Status Of Every Layer
    records = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub_layer in layer.layers:
                records.append(
                    {
                        "layer": f"{layer.name}/{sub_layer.name}",
                        "type": sub_layer.__class__.__name__,
                        "trainable": sub_layer.trainable,
                        "params": sub_layer.count_params(),
                    }
                )
        else:
            records.append(
                {
                    "layer": layer.name,
                    "type": layer.__class__.__name__,
                    "trainable": layer.trainable,
                    "params": layer.count_params(),
                }
            )
    return pd.DataFrame(records)
