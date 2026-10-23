"""
Project Configuration and Reproducibility Setup.

"""

import os
import random
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"

TRAIN_DATA_URL = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
VALID_DATA_URL = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"

TRAIN_FILE = DATA_DIR / "train-data.tsv"
VALID_FILE = DATA_DIR / "valid-data.tsv"
CLEANED_DATA_FILE = DATA_DIR / "sms_cleaned.parquet"

BEST_MODEL_PATH = Path(os.getenv("SMS_MODEL_PATH", str(MODELS_DIR / "best_model.keras")))
BEST_TFLITE_PATH = MODELS_DIR / "best_model.tflite"
MODEL_MANIFEST_FILE = ARTIFACTS_DIR / "model_manifest.json"

for folder in [DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)



SEED = 42

VOCAB_SIZE = 8000               # Sufficient to cover SMS jargon
MAX_SEQUENCE_LENGTH = 120       # Covers 99%+ of SMS message lengths
EMBEDDING_DIM = 64              

BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE_EARLY_STOPPING = 4     # Restore best weights when val_loss plateaus
PATIENCE_REDUCE_LR = 2          # Halve LR when loss stagnates

# best_model.keras is the tuned BiLSTM artifact; keep its validated serving
# threshold aligned with artifacts/model_manifest.json.
DEFAULT_THRESHOLD = 0.95

URL_TOKEN = "<URL>"
PHONE_TOKEN = "<PHONE>"
MONEY_TOKEN = "<MONEY>"


def set_seed(seed: int = SEED) -> None:
    """
    Sets deterministic seeds across Python, NumPy, and TensorFlow.

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
    except ImportError:
        pass
