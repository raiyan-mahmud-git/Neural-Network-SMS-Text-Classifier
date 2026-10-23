"""
Comprehensive Evaluation, Error Analysis, Threshold Tuning & Model Explainability.

"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

import config
import data
import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = config.DEFAULT_THRESHOLD) -> dict[str, Any]:
    """
    Computes a comprehensive suite of binary classification metrics.

    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }


def tune_decision_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    fp_cost_weight: float = 3.0
) -> tuple[float, pd.DataFrame]:
    """
    Sweeps decision thresholds [0.05, 0.95] to find the optimal operating point.

    """
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []

    best_score = -1.0
    best_threshold = 0.5

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, _tp = cm.ravel()

       
        # score = F1 - (fp_cost_weight * (fp / len(y_val)))
        cost_penalized_score = f1 - (fp_cost_weight * (fp / max(1, (tn + fp))))

        records.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "FP": fp,
            "FN": fn,
            "cost_penalized_score": cost_penalized_score
        })

        if cost_penalized_score > best_score:
            best_score = cost_penalized_score
            best_threshold = t

    df_sweep = pd.DataFrame(records)
    logger.info(f"Optimal threshold chosen: {best_threshold:.2f} (Validation F1 at this threshold: {df_sweep.loc[df_sweep['threshold'] == best_threshold, 'f1'].values[0]:.4f})")
    return float(best_threshold), df_sweep


def plot_threshold_curves(df_sweep: pd.DataFrame, save_path: Path) -> None:
    """Plots Precision, Recall, and F1 across decision thresholds."""
    plt.figure(figsize=(9, 5))
    plt.plot(df_sweep["threshold"], df_sweep["precision"], label="Precision", color="blue", lw=2)
    plt.plot(df_sweep["threshold"], df_sweep["recall"], label="Recall", color="green", lw=2)
    plt.plot(df_sweep["threshold"], df_sweep["f1"], label="F1-Score", color="purple", lw=2, linestyle="--")
    plt.axvline(0.5, color="grey", linestyle=":", label="Default Threshold (0.50)")
    plt.title("Precision, Recall, and F1 Trade-off Across Decision Thresholds", fontsize=12, fontweight="bold")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_learning_curves(history: keras.callbacks.History, model_name: str, save_path: Path) -> None:
    """
    Plots training and validation loss and PR-AUC curves to diagnose overfitting/underfitting.
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


def plot_calibration_curves(models_dict: dict[str, tuple[np.ndarray, np.ndarray]], save_path: Path) -> None:
    """
    Generates a Reliability Diagram for evaluated models.

    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration (Ideal)")

    for name, (y_true, y_prob) in models_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{name}")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (True Frequency)")
    plt.title("Reliability Diagram (Probability Calibration)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def perform_error_analysis(
    texts: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
    top_n: int = 10
) -> dict[str, list[dict[str, Any]]]:
    """
    Identifies and analyzes top False Positives and False Negatives.

    """
    preds = (y_prob >= threshold).astype(int)

    fp_mask = (y_true == 0) & (preds == 1)
    fn_indices = np.where((y_true == 1) & (preds == 0))[0]

    # Top false positives
    fp_indices = np.where(fp_mask)[0]
    fp_sorted = fp_indices[np.argsort(-y_prob[fp_indices])][:top_n]

    # Top false negatives
    fn_sorted = fn_indices[np.argsort(y_prob[fn_indices])][:top_n]

    false_positives = [
        {"text": texts[i], "true_label": "ham", "predicted_prob": float(y_prob[i])}
        for i in fp_sorted
    ]

    false_negatives = [
        {"text": texts[i], "true_label": "spam", "predicted_prob": float(y_prob[i])}
        for i in fn_sorted
    ]

    return {"false_positives": false_positives, "false_negatives": false_negatives}


def explain_prediction_occlusion(text: str, predict_fn, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Word-level occlusion attribution:
    Measures the drop in spam probability when each word is removed.

    """
    words = text.split()
    if not words:
        return []

    base_prob = float(predict_fn([text])[0])
    attributions = []

    for i, word in enumerate(words):
        # Create text with word i omitted
        occluded_words = words[:i] + words[i+1:]
        occluded_text = " ".join(occluded_words) if occluded_words else ""
        new_prob = float(predict_fn([occluded_text])[0])
        importance = base_prob - new_prob
        attributions.append((word, importance))

    # Sort by absolute impact
    attributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return attributions[:top_k]


def run_stratified_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "baseline",
    n_splits: int = 5,
    seed: int = config.SEED
) -> dict[str, tuple[float, float]]:
    """
    Performs 5-fold stratified cross-validation for a model.

    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    f1_scores = []
    pr_auc_scores = []
    roc_auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_v, y_v = X[val_idx], y[val_idx]

        if model_type == "baseline":
            pipe = models.build_baseline_tfidf(model_type="nb")
            pipe.fit(X_tr, y_tr)
            probs = pipe.predict_proba(X_v)[:, 1]
        elif model_type == "bilstm":
            vec = models.create_text_vectorizer(X_tr)
            mod = models.build_bilstm_model(vec)
            cw = data.get_class_weights(y_tr)
            mod.fit(X_tr, y_tr, epochs=8, batch_size=config.BATCH_SIZE, class_weight=cw, verbose=0)
            probs = mod.predict(X_v, verbose=0).ravel()
        else:
            raise ValueError(f"Unsupported model_type {model_type}")

        preds = (probs >= 0.5).astype(int)
        f1_scores.append(f1_score(y_v, preds, zero_division=0))
        pr_auc_scores.append(average_precision_score(y_v, probs))
        roc_auc_scores.append(roc_auc_score(y_v, probs))

    return {
        "f1_mean_std": (float(np.mean(f1_scores)), float(np.std(f1_scores))),
        "pr_auc_mean_std": (float(np.mean(pr_auc_scores)), float(np.std(pr_auc_scores))),
        "roc_auc_mean_std": (float(np.mean(roc_auc_scores)), float(np.std(roc_auc_scores))),
    }
