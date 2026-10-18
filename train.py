"""
Model Training Pipeline
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
import data
import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_learning_curves(history: keras.callbacks.History, model_name: str, save_path: Path) -> None:
    """
    Plots training and validation loss and PR-AUC curves to diagnose overfitting
    
    """
    epochs_range = range(1, len(history.history["loss"]) + 1)

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    axes[0].plot(epochs_range, history.history["loss"], "b-o", label="Training Loss")
    axes[0].plot(epochs_range, history.history["val_loss"], "r--s", label="Validation Loss")
    axes[0].set_title(f"{model_name}: Binary Crossentropy Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # PR-AUC / ROC-AUC subplot
    metric_key = "pr_auc" if "pr_auc" in history.history else "roc_auc"
    val_metric_key = f"val_{metric_key}"
    if metric_key in history.history and val_metric_key in history.history:
        axes[1].plot(epochs_range, history.history[metric_key], "b-o", label=f"Train {metric_key.upper()}")
        axes[1].plot(epochs_range, history.history[val_metric_key], "r--s", label=f"Val {metric_key.upper()}")
        axes[1].set_title(f"{model_name}: Area Under Precision-Recall Curve (PR-AUC)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel(metric_key.upper())
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Saved learning curves to {save_path}")


def train_single_neural_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    vectorizer: tf.keras.layers.TextVectorization,
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    imbalance_strategy: str = "class_weight"
) -> tuple[keras.Model, dict[str, Any]]:
    """
    Trains a selected neural model with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.

    WHY CALLBACKS MATTER:
    - EarlyStopping: Halts training when validation loss stops improving, avoiding memorization of noise.
    - restore_best_weights=True: Automatically reverts weights to the checkpoint with the lowest val_loss,
      so the user never has to guess which epoch was optimal.
    - ReduceLROnPlateau: Gradually reduces step size when stuck in a plateau to settle into deeper minima.
    """
    logger.info(f"--- Training {model_name} (imbalance strategy: {imbalance_strategy}) ---")
    model = models.get_model_by_name(model_name, vectorizer)

    checkpoint_path = config.MODELS_DIR / f"{model_name}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.PATIENCE_EARLY_STOPPING,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.PATIENCE_REDUCE_LR,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        )
    ]

    # class imbalance
    class_weights = None
    if imbalance_strategy == "class_weight":
        class_weights = data.get_class_weights(y_train)
        logger.info(f"Using class weights: {class_weights}")
    elif imbalance_strategy == "oversample":
        X_train, y_train = data.oversample_minority_class(X_train, y_train)
        logger.info(f"Oversampled training set size: {len(X_train)} samples")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    plot_path = config.PLOTS_DIR / f"{model_name}_learning_curve.png"
    plot_learning_curves(history, model_name, plot_path)

    # Save final model
    model.save(checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")

    val_res = model.evaluate(X_val, y_val, verbose=0)
    val_metrics = dict(zip(model.metrics_names, [float(x) for x in val_res]))

    return model, val_metrics


def train_baseline_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "nb"
) -> tuple[Any, dict[str, float]]:
    """Trains classical TF-IDF baseline and calculates validation scores."""
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logger.info(f"--- Training Classical Baseline: TF-IDF + {model_type.upper()} ---")
    pipeline = models.build_baseline_tfidf(model_type=model_type)
    pipeline.fit(X_train, y_train)

    val_preds_prob = pipeline.predict_proba(X_val)[:, 1]
    val_preds = (val_preds_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, val_preds)),
        "precision": float(precision_score(y_val, val_preds, zero_division=0)),
        "recall": float(recall_score(y_val, val_preds, zero_division=0)),
        "f1": float(f1_score(y_val, val_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, val_preds_prob)),
        "pr_auc": float(average_precision_score(y_val, val_preds_prob)),
    }

    baseline_path = config.MODELS_DIR / f"baseline_{model_type}.joblib"
    joblib.dump(pipeline, baseline_path)
    logger.info(f"Baseline saved to {baseline_path}. Val F1: {metrics['f1']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}")

    return pipeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="SMS Spam Classifier Training CLI")
    parser.add_argument("--model", type=str, default="bilstm",
                        choices=["baseline", "original", "cnn", "bilstm", "transformer", "all"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--imbalance", type=str, default="class_weight",
                        choices=["none", "class_weight", "oversample"],
                        help="Class imbalance mitigation strategy")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed")
    args = parser.parse_args()

    config.set_seed(args.seed)

    # leak-free
    X_train, y_train, X_val, y_val, _X_test, _y_test = data.get_stratified_3way_splits(seed=args.seed)

    # Create & adapt TextVectorization layer
    logger.info("Adapting TextVectorization on training corpus...")
    vectorizer = models.create_text_vectorizer(X_train)

    models_to_train = [args.model] if args.model != "all" else ["baseline", "original", "cnn", "bilstm", "transformer"]

    results = {}

    for name in models_to_train:
        if name == "baseline":
            _, val_metrics = train_baseline_pipeline(X_train, y_train, X_val, y_val, model_type="nb")
            results["baseline_nb"] = val_metrics
        else:
            _trained_model, val_metrics = train_single_neural_model(
                model_name=name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                vectorizer=vectorizer,
                epochs=args.epochs,
                batch_size=args.batch_size,
                imbalance_strategy=args.imbalance
            )
            results[name] = val_metrics

    logger.info("Training complete. Validation metrics summary:")
    logger.info(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
