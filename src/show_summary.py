import os
import sys

# Ensure Workspace Root In Python Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.models import build_baseline_cnn


def export_baseline_summary(output_csv="report/tables/baseline_summary.csv"):
    # Build Baseline Model And Print Architecture Summary
    model = build_baseline_cnn()

    print("=================================================================")
    print("                VISIONGUARD BASELINE CNN SUMMARY                ")
    print("=================================================================")
    model.summary()
    print("=================================================================")

    # Extract Per-Layer Output Shape And Parameter Count
    records = []
    for layer in model.layers:
        try:
            out_shape = str(layer.output.shape)
        except Exception:
            out_shape = str(getattr(layer, "output_shape", "N/A"))

        records.append(
            {
                "layer_name": layer.name,
                "layer_type": layer.__class__.__name__,
                "output_shape": out_shape,
                "param_count": layer.count_params(),
                "trainable": layer.trainable,
            }
        )

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nBaseline layer summary table saved to {output_csv}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum(layer.count_params() for layer in model.layers if layer.trainable):,}")


if __name__ == "__main__":
    export_baseline_summary()
