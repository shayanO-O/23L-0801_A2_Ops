import importlib.metadata
import json
import os
import sys

# Pinned Core Packages To Enforce
PIN_PACKAGES = ["tensorflow", "keras", "mlflow"]

# All Inspected Packages For CP-1.1
DISPLAY_PACKAGES = [
    "tensorflow",
    "keras",
    "numpy",
    "pandas",
    "pillow",
    "scikit-learn",
    "mlflow",
    "dvc",
    "flask",
]


def load_pins(pins_path="pins.json"):
    # Read Expected Pins From File
    with open(pins_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_gpu():
    # Detect GPU Through TensorFlow
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        return f"Detected {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}" if gpus else "No GPU detected (CPU mode)"
    except Exception as exc:
        return f"GPU check failed or TensorFlow not loaded: {exc}"


def get_version(pkg_name):
    # Lookup Installed Package Version
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        return "Not installed"


def run_version_report(output_file="report/tables/versions.txt"):
    # Generate Formatted Version And GPU Summary
    python_ver = sys.version.split()[0]
    lines = [
        f"Python Version: {python_ver}",
        "--- Installed Packages ---",
    ]
    for pkg in DISPLAY_PACKAGES:
        lines.append(f"{pkg}: {get_version(pkg)}")
    lines.append("--- GPU Check ---")
    lines.append(check_gpu())

    report_str = "\n".join(lines)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_str + "\n")
    print(report_str)
    return report_str


def verify_against_pins(pins_path="pins.json"):
    # Check Required Packages Match Exact Pins
    pins = load_pins(pins_path)
    mismatches = []
    for pkg, expected in pins.items():
        actual = get_version(pkg)
        if actual != expected:
            mismatches.append(f"{pkg}: expected {expected}, found {actual}")
    if mismatches:
        print("[ERROR] Version mismatch detected:")
        for msg in mismatches:
            print(f"  - {msg}")
        return False
    print("[SUCCESS] All pinned package versions match pins.json")
    return True


if __name__ == "__main__":
    run_version_report()
    if os.path.exists("pins.json"):
        verify_against_pins()
