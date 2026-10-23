"""
Data Pipeline: Download, Ingestion, Text Cleaning, Split Management & Imbalance Handling.

"""

import logging
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(url: str, destination_path) -> None:
    """
    Downloads raw dataset files using standard cross-platform HTTP requests.

    """
    if not destination_path.exists():
        logger.info(f"Downloading from {url} to {destination_path}...")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            with open(destination_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Successfully downloaded {destination_path.name}")
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
    else:
        logger.debug(f"File {destination_path.name} already exists. Skipping download.")


def ensure_datasets_present() -> tuple[config.Path, config.Path]:
    """Ensures both train-data.tsv and valid-data.tsv are downloaded."""
    download_dataset(config.TRAIN_DATA_URL, config.TRAIN_FILE)
    download_dataset(config.VALID_DATA_URL, config.VALID_FILE)
    return config.TRAIN_FILE, config.VALID_FILE


def clean_text(text: str) -> str:
    """
    Applies domain-aware token normalization to an SMS string.

    Transformations:
    - Lowercase conversion
    - Normalizes URLs -> <URL>
    - Normalizes Phone Numbers -> <PHONE>
    - Normalizes Currency/Prizes -> <MONEY>
    - Cleans excessive whitespace

    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # 1. Normalize URLs
    text = re.sub(r"https?://\S+|www\.\S+", f" {config.URL_TOKEN} ", text)

    # 2. Normalize Currency & Monetary mentions
    text = re.sub(r"(?:[\$£€]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:pounds?|quid|pence|dollars?|cents?|euros?|cash|prize)\b)", f" {config.MONEY_TOKEN} ", text)

    # 3. Normalize Phone numbers
    text = re.sub(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{3,5}\b", f" {config.PHONE_TOKEN} ", text)

    # 4. Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw freeCodeCamp train and validation dataframes.
    """
    ensure_datasets_present()
    train_df = pd.read_csv(config.TRAIN_FILE, sep="\t", header=None, names=["label", "message"])
    valid_df = pd.read_csv(config.VALID_FILE, sep="\t", header=None, names=["label", "message"])

    # Ensure labels are binary (1 for spam, 0 for ham)
    train_df["target"] = (train_df["label"] == "spam").astype(int)
    valid_df["target"] = (valid_df["label"] == "spam").astype(int)

    return train_df, valid_df


def inspect_leakage_and_duplicates(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict[str, int]:
    """
    Analyzes duplicate messages within and across dataset splits.

    """
    train_dupes = int(train_df.duplicated(subset=["message"]).sum())
    valid_dupes = int(valid_df.duplicated(subset=["message"]).sum())

    train_msgs = set(train_df["message"])
    valid_msgs = set(valid_df["message"])
    leakage_count = len(train_msgs.intersection(valid_msgs))

    report = {
        "train_rows": len(train_df),
        "train_duplicates": train_dupes,
        "valid_rows": len(valid_df),
        "valid_duplicates": valid_dupes,
        "train_valid_overlapping_messages": leakage_count,
    }

    logger.info(
        f"Data Leakage Audit: {train_dupes} train duplicates, "
        f"{valid_dupes} valid duplicates, {leakage_count} overlapping messages across splits."
    )
    return report


def get_stratified_3way_splits(
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    clean: bool = True,
    seed: int = config.SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combines all available data, drops exact duplicates, and generates a leak-free,
    stratified 3-way split: (X_train, y_train), (X_val, y_val), (X_test, y_test).

    """
    train_df, valid_df = load_raw_data()
    combined_df = pd.concat([train_df, valid_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    if clean:
        combined_df["clean_message"] = combined_df["message"].apply(clean_text)
        X = np.array(combined_df["clean_message"].tolist(), dtype=object)
    else:
        X = np.array(combined_df["message"].tolist(), dtype=object)

    y = np.array(combined_df["target"].tolist(), dtype=int)

    # Step 1: Split into train and temp (val + test)
    val_test_ratio = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_test_ratio, random_state=seed, stratify=y
    )

    # Step 2: Split temp equally into validation and test sets
    relative_test_ratio = test_size / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_ratio, random_state=seed, stratify=y_temp
    )

    logger.info(
        f"Stratified Splits: Train={len(X_train)} (Spam={sum(y_train)}), "
        f"Val={len(X_val)} (Spam={sum(y_val)}), Test={len(X_test)} (Spam={sum(y_test)})"
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_fcc_raw_splits(clean: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns train and valid data adhering strictly to the freeCodeCamp split structure.
    Used for direct backward-compatibility verification.
    """
    train_df, valid_df = load_raw_data()

    if clean:
        train_texts = np.array(train_df["message"].apply(clean_text).tolist(), dtype=object)
        valid_texts = np.array(valid_df["message"].apply(clean_text).tolist(), dtype=object)
    else:
        train_texts = np.array(train_df["message"].tolist(), dtype=object)
        valid_texts = np.array(valid_df["message"].tolist(), dtype=object)

    return train_texts, np.array(train_df["target"].tolist(), dtype=int), valid_texts, np.array(valid_df["target"].tolist(), dtype=int)


def get_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Calculates balanced class weights: n_samples / (n_classes * np.bincount(y)).

    """
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    return {cls: float(weight) for cls, weight in zip(np.unique(y), weights)}


def oversample_minority_class(X: np.ndarray, y: np.ndarray, seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs random oversampling on the minority class (spam) to balance the training distribution.

    """
    np.random.seed(seed)
    spam_indices = np.where(y == 1)[0]
    ham_indices = np.where(y == 0)[0]

    oversampled_spam_idx = np.random.choice(spam_indices, size=len(ham_indices), replace=True)
    combined_idx = np.concatenate([ham_indices, oversampled_spam_idx])
    np.random.shuffle(combined_idx)

    return X[combined_idx], y[combined_idx]


if __name__ == "__main__":
    train_df, valid_df = load_raw_data()
    print("Train shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)
    leakage_stats = inspect_leakage_and_duplicates(train_df, valid_df)
    print("Leakage stats:", leakage_stats)
