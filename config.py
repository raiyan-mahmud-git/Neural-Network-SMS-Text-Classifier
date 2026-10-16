"""
Project Configuration and Reproducibility Setup.

WHY THIS MATTERS:
Centralizing configuration avoids 'magic numbers' scattered across scripts,
enables clean CLI overrides, and ensures all components (training, evaluation,
serving) reference identical paths, tokens, and sequence dimensions.
"""

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Directories & File Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"

# FreeCodeCamp raw data URLs
TRAIN_DATA_URL = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
VALID_DATA_URL = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"

TRAIN_FILE = DATA_DIR / "train-data.tsv"
VALID_FILE = DATA_DIR / "valid-data.tsv"
CLEANED_DATA_FILE = DATA_DIR / "sms_cleaned.parquet"

# Best saved model path for serving/FCC evaluation
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
BEST_TFLITE_PATH = MODELS_DIR / "best_model.tflite"

# Ensure runtime directories exist
for folder in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Modeling & Preprocessing Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42

# Vocabulary & Text Vectorization
VOCAB_SIZE = 8000               # Sufficient to cover SMS jargon without severe OOV
MAX_SEQUENCE_LENGTH = 120       # Covers 99%+ of SMS message lengths
EMBEDDING_DIM = 64              # Rich enough representation without overfitting small data

# Training
BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE_EARLY_STOPPING = 4     # Restore best weights when val_loss plateaus
PATIENCE_REDUCE_LR = 2          # Halve LR when loss stagnates

# Classification & Thresholding
DEFAULT_THRESHOLD = 0.5         # Tunable on validation set to balance FP vs FN

# Special entity tokens (Preserves critical spam signals)
URL_TOKEN = "<URL>"
PHONE_TOKEN = "<PHONE>"
MONEY_TOKEN = "<MONEY>"


def set_seed(seed: int = SEED) -> None:
    """
    Sets deterministic seeds across Python, NumPy, and TensorFlow.

    WHY THIS MATTERS:
    Neural network training contains multiple stochastic sources: weight initialization,
    batch shuffling, and dropout masks. Without explicit seeding across all three
    libraries and OS hash seeds, two runs will yield diverging results, making it
    impossible to scientifically verify whether an architectural upgrade truly helped.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Note: on GPU, deterministic ops can also be forced if needed:
        # tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass
