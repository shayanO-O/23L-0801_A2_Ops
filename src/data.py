import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

SEED = 42
DEFAULT_IMG_SIZE = (224, 224)


def decode_and_resize(image_bytes, img_size=DEFAULT_IMG_SIZE):
    # Standard Decoding And Resizing Op Shared By Training And Serving
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size, method="bilinear")
    img = tf.cast(img, tf.float32)
    img.set_shape([img_size[0], img_size[1], 3])
    return img


def load_image_from_path(relpath, raw_dir="data/raw", img_size=DEFAULT_IMG_SIZE):
    # Read File Bytes And Run Standard Preprocessing
    file_path = tf.strings.join([raw_dir, "/", relpath])
    img_bytes = tf.io.read_file(file_path)
    return decode_and_resize(img_bytes, img_size=img_size)


def get_augmentation_layer(seed=SEED):
    # Training Data Augmentation Applied After Cache
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, seed=seed),
            layers.RandomZoom(0.05, seed=seed),
            layers.RandomContrast(0.1, seed=seed),
        ],
        name="data_augmentation",
    )


def create_dataset_from_csv(
    csv_path,
    raw_dir="data/raw",
    img_size=DEFAULT_IMG_SIZE,
    batch_size=32,
    is_training=False,
    augment=True,
    seed=SEED,
):
    # Build High Performance tf.data Input Pipeline
    df = pd.read_csv(csv_path)
    relpaths = df["relpath"].values
    labels = df["label"].values.astype("float32")

    paths_ds = tf.data.Dataset.from_tensor_slices(relpaths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)

    def _process_path(path):
        return load_image_from_path(path, raw_dir=raw_dir, img_size=img_size)

    images_ds = paths_ds.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = tf.data.Dataset.zip((images_ds, labels_ds))

    if is_training:
        # Cache Raw Decoded Features Before Random Augmentation
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=min(len(df), 1000), seed=seed)
        ds = ds.batch(batch_size)
        if augment:
            aug_layer = get_augmentation_layer(seed=seed)
            ds = ds.map(
                lambda x, y: (aug_layer(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
