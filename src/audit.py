import hashlib
import os
import pandas as pd
from PIL import Image


def compute_sha256(file_path):
    # Compute SHA-256 Hash Of File
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_image(file_path, raw_dir):
    # Inspect Single Image File Metadata
    relpath = os.path.relpath(file_path, raw_dir).replace("\\", "/")
    label = os.path.basename(os.path.dirname(file_path))
    file_size = os.path.getsize(file_path)
    sha256_hash = compute_sha256(file_path)

    width, height, mode, img_format = None, None, None, None
    is_corrupt = False
    error_msg = ""

    try:
        with Image.open(file_path) as img:
            img.verify()
        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode
            img_format = img.format
    except Exception as exc:
        is_corrupt = True
        error_msg = str(exc)

    return {
        "relpath": relpath,
        "class": label,
        "file_size": file_size,
        "width": width,
        "height": height,
        "mode": mode,
        "format": img_format,
        "sha256": sha256_hash,
        "is_corrupt": is_corrupt,
        "error": error_msg,
    }


def run_audit(raw_dir="data/raw", tables_dir="report/tables"):
    # Walk Dataset And Build Audit Report
    os.makedirs(tables_dir, exist_ok=True)
    records = []

    for root, _, files in os.walk(raw_dir):
        for f in files:
            full_path = os.path.join(root, f)
            records.append(audit_image(full_path, raw_dir))

    df = pd.DataFrame(records)
    csv_path = os.path.join(tables_dir, "audit.csv")
    df.to_csv(csv_path, index=False)

    # Detect Duplicates By Hash
    hash_counts = df.groupby("sha256")["class"].unique()
    duplicates_mask = df["sha256"].duplicated(keep=False)
    num_duplicates = duplicates_mask.sum()

    # Identify Label Conflicts
    conflict_hashes = hash_counts[hash_counts.apply(len) > 1].index.tolist()
    df["has_conflict"] = df["sha256"].isin(conflict_hashes)
    num_conflicts = df["has_conflict"].sum()

    summary_lines = [
        "# Dataset Audit Summary",
        "",
        "## Overview",
        f"- Total image records scanned: {len(df)}",
        f"- Corrupt or unreadable files: {df['is_corrupt'].sum()}",
        f"- Total duplicate instances: {num_duplicates}",
        f"- Cross-class conflicting instances: {num_conflicts}",
        "",
        "## Class Distribution",
    ]

    valid_df = df[~df["is_corrupt"]]
    for cls_name, grp in valid_df.groupby("class"):
        summary_lines.extend(
            [
                f"### Class: `{cls_name}`",
                f"- Count: {len(grp)}",
                f"- Width (min / median / max): {grp['width'].min()} / {grp['width'].median()} / {grp['width'].max()}",
                f"- Height (min / median / max): {grp['height'].min()} / {grp['height'].median()} / {grp['height'].max()}",
                f"- File Size Bytes (min / median / max): {grp['file_size'].min()} / {int(grp['file_size'].median())} / {grp['file_size'].max()}",
                f"- Formats: {grp['format'].value_counts().to_dict()}",
                f"- Modes: {grp['mode'].value_counts().to_dict()}",
                "",
            ]
        )

    summary_md_path = os.path.join(tables_dir, "audit_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Audit completed: {len(df)} files processed.")
    print(f"Summary saved to {summary_md_path}")
    return df


if __name__ == "__main__":
    run_audit()
