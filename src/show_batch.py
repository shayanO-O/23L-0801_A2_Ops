import os
import sys

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))

import matplotlib.pyplot as plt
import numpy as np
from src.data import create_dataset_from_csv


def save_augmented_batch(
    csv_path="splits/train.csv",
    output_path="report/figures/augmented_batch.png",
    n_images=9,
):
    # Load One Augmented Training Batch
    ds = create_dataset_from_csv(csv_path, is_training=True, augment=True, batch_size=32)
    images, labels = next(iter(ds))

    images = images.numpy()
    labels = labels.numpy()

    cols = 3
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    label_names = {0: "Real", 1: "AI-Generated"}

    for i in range(n_images):
        ax = axes[i]
        # Images Are In [0, 255] Range
        img = np.clip(images[i], 0, 255).astype("uint8")
        ax.imshow(img)
        lbl = int(labels[i])
        ax.set_title(f"{label_names.get(lbl, str(lbl))} ({lbl})", fontsize=11, weight="bold")
        ax.axis("off")

    for j in range(n_images, len(axes)):
        axes[j].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.suptitle("Augmented Training Batch (3x3 Grid, 224x224)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Augmented batch saved to {output_path}")


if __name__ == "__main__":
    save_augmented_batch()
