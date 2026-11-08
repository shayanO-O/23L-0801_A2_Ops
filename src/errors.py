import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))
os.environ["PYTHONIOENCODING"] = "utf-8"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data import create_dataset_from_csv


def generate_error_analysis():
    # Analyze Model Failures On Sealed Test Split
    print("Generating Error Analysis on Test Split...")
    test_df = pd.read_csv("splits/test.csv")
    y_test = test_df["label"].values

    test_ds = create_dataset_from_csv("splits/test.csv", is_training=False, augment=False, batch_size=32)

    model_path = os.path.join("model", "champion.keras")
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(test_ds, verbose=0).flatten()

    threshold = 0.5252
    preds = (probs >= threshold).astype(int)

    # Identify Misclassifications
    misclassified_indices = np.where(preds != y_test)[0]
    print(f"Total Misclassified Test Samples: {len(misclassified_indices)} out of {len(y_test)}")

    # Separate False Positives (Real labeled as AI) and False Negatives (AI labeled as Real)
    fps = [idx for idx in misclassified_indices if y_test[idx] == 0 and preds[idx] == 1]
    fns = [idx for idx in misclassified_indices if y_test[idx] == 1 and preds[idx] == 0]

    print(f" - False Positives (Real misclassified as AI): {len(fps)}")
    print(f" - False Negatives (AI misclassified as Real): {len(fns)}")

    # Select 5 FPs and 5 FNs for Balanced 10-Image Grid
    selected_indices = fps[:5] + fns[:5]
    if len(selected_indices) < 10:
        remaining = [idx for idx in misclassified_indices if idx not in selected_indices]
        selected_indices.extend(remaining[: 10 - len(selected_indices)])

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices[:10]):
        row = test_df.iloc[idx]
        img_path = os.path.join("data", "raw", row["relpath"])
        p = probs[idx]
        gt_label = "Real" if row["label"] == 0 else "AI"
        err_type = "False Positive" if row["label"] == 0 else "False Negative"
        color = "#DC2626" if err_type == "False Positive" else "#9333EA"

        try:
            img = Image.open(img_path).convert("RGB")
            axes[i].imshow(img)
        except Exception as e:
            axes[i].text(0.5, 0.5, "Image Load\nError", ha="center", va="center")

        title = f"True: {gt_label}\nPred P(AI): {p:.2f}\n[{err_type}]"
        axes[i].set_title(title, fontsize=9, fontweight="bold", color=color)
        axes[i].axis("off")

    plt.suptitle("VisionGuard Champion Error Analysis: Top 10 Test Misclassifications", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join("report", "figures", "misclassified_grid.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Successfully saved error analysis grid to {out_path}.")
    return out_path


if __name__ == "__main__":
    generate_error_analysis()
