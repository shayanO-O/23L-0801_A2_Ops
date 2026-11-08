import os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
LABEL_MAP = {
    "RealArt": 0,
    "AiArtData": 1,
}


def create_splits(
    audit_csv="report/tables/audit.csv",
    splits_dir="splits",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=SEED,
):
    # Ensure Splits Output Directory Exists
    os.makedirs(splits_dir, exist_ok=True)

    df = pd.read_csv(audit_csv)

    # 1. Drop Corrupt Files
    clean_df = df[~df["is_corrupt"]].copy()

    # 2. Identify And Drop Cross-Class Duplicate Conflicts
    hash_classes = clean_df.groupby("sha256")["class"].nunique()
    conflict_hashes = hash_classes[hash_classes > 1].index
    num_conflicts = clean_df["sha256"].isin(conflict_hashes).sum()
    if num_conflicts > 0:
        print(f"Dropping {num_conflicts} records due to cross-class label conflicts.")
        clean_df = clean_df[~clean_df["sha256"].isin(conflict_hashes)]

    # 3. Deduplicate Within Class (Keep First Instance)
    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["sha256"], keep="first")
    print(f"Removed {before_dedup - len(clean_df)} duplicate copies; retained {len(clean_df)} unique images.")

    # 4. Map Integer Labels
    clean_df["label"] = clean_df["class"].map(LABEL_MAP)
    clean_df = clean_df.dropna(subset=["label"])
    clean_df["label"] = clean_df["label"].astype(int)

    # 5. Stratified 70/15/15 Split
    # Split Into Train (70%) And Temp (30%)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        clean_df,
        test_size=temp_ratio,
        stratify=clean_df["label"],
        random_state=seed,
    )

    # Split Temp (30%) Evenly Into Val (15%) And Test (15%)
    val_share = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share),
        stratify=temp_df["label"],
        random_state=seed,
    )

    # 6. Save Relpath And Label Columns
    split_cols = ["relpath", "label"]
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")

    train_df[split_cols].to_csv(train_path, index=False)
    val_df[split_cols].to_csv(val_path, index=False)
    test_df[split_cols].to_csv(test_path, index=False)

    print("\n--- Split Manifest Summary ---")
    print(f"Train set: {len(train_df)} samples | Distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"Val set:   {len(val_df)} samples | Distribution: {val_df['label'].value_counts().to_dict()}")
    print(f"Test set:  {len(test_df)} samples | Distribution: {test_df['label'].value_counts().to_dict()}")
    print(f"Splits saved to {splits_dir}/\n")

    return train_df, val_df, test_df


if __name__ == "__main__":
    create_splits()
