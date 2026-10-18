"""
Hyperparameter Optimization Pipeline.

Sweeps:
- Vocabulary Size: [5000, 8000]
- Embedding Dimension: [32, 64]
- Sequence Length: [80, 120]
- Recurrent/Hidden Units: [16, 32, 64]
- Dropout Rate: [0.2, 0.4]
- Learning Rate: [5e-4, 1e-3, 2e-3]

import itertools
import json
import logging
from typing import Any

import numpy as np
from tensorflow import keras

import config
import data
import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_hyperparameter_sweep(
    max_trials: int = 8,
    epochs_per_trial: int = 6,
    seed: int = config.SEED
) -> dict[str, Any]:
    """
    Executes a structured hyperparameter exploration on the validation set.
    """
    config.set_seed(seed)
    X_train, y_train, X_val, y_val, _, _ = data.get_stratified_3way_splits(seed=seed)
    cw = data.get_class_weights(y_train)

    param_grid = {
        "vocab_size": [5000, 8000],
        "embed_dim": [32, 64],
        "lstm_units": [16, 32],
        "dropout": [0.2, 0.4],
        "lr": [1e-3, 2e-3]
    }

    # Generate parameter combinations
    keys, values = zip(*param_grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    np.random.seed(seed)
    np.random.shuffle(all_combos)
    selected_combos = all_combos[:max_trials]

    logger.info(f"Starting Hyperparameter Sweep: evaluating {len(selected_combos)} candidate configurations...")

    trials = []
    best_pr_auc = -1.0
    best_config = None

    for idx, params in enumerate(selected_combos, 1):
        logger.info(f"--- Trial {idx}/{len(selected_combos)}: {params} ---")

        # Create vectorizer
        vectorizer = models.create_text_vectorizer(
            train_texts=X_train,
            vocab_size=params["vocab_size"],
            max_length=config.MAX_SEQUENCE_LENGTH
        )

        # parameterized BiLSTM
        model = models.build_bilstm_model(
            vectorizer=vectorizer,
            embedding_dim=params["embed_dim"],
            lstm_units=params["lstm_units"],
            dropout_rate=params["dropout"],
            learning_rate=params["lr"]
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=0
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_per_trial,
            batch_size=config.BATCH_SIZE,
            class_weight=cw,
            callbacks=[early_stopping],
            verbose=0
        )

        val_metrics = model.evaluate(X_val, y_val, verbose=0)
        metrics_dict = dict(zip(model.metrics_names, [float(x) for x in val_metrics]))

        pr_auc = metrics_dict.get("pr_auc", 0.0)
        val_loss = metrics_dict.get("loss", 1.0)
        val_acc = metrics_dict.get("accuracy", 0.0)

        trial_record = {
            "trial_id": idx,
            "params": params,
            "val_pr_auc": pr_auc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "stopped_epoch": len(history.history["loss"])
        }
        trials.append(trial_record)

        logger.info(f"Trial {idx} Results -> Val PR-AUC: {pr_auc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_config = trial_record

    results_summary = {
        "best_trial": best_config,
        "all_trials": trials
    }

    out_file = config.ARTIFACTS_DIR / "tuning_results.json"
    with open(out_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"Tuning complete. Best config: {best_config['params']} with Val PR-AUC: {best_config['val_pr_auc']:.4f}")
    logger.info(f"Saved tuning logs to {out_file}")

    return results_summary


if __name__ == "__main__":
    run_hyperparameter_sweep(max_trials=6, epochs_per_trial=5)
