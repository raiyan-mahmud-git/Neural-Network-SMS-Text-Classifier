"""
Experiment: Class Imbalance Mitigation Strategies Comparison.

Compares:
1. No Handling (Unweighted loss)
2. Algorithmic Reweighting (class_weight)
3. Data-Level Resampling (Oversampling minority class)

"""

import json
import logging

import pandas as pd
from tensorflow import keras

import config
import data
import models
from evaluate import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_imbalance_experiment(epochs: int = 10) -> pd.DataFrame:
    config.set_seed(config.SEED)
    X_train, y_train, X_val, y_val, X_test, y_test = data.get_stratified_3way_splits()

    vectorizer = models.create_text_vectorizer(X_train)

    strategies = ["no_handling", "class_weight", "oversampling"]
    results = []

    for strat in strategies:
        logger.info(f"Running Imbalance Strategy: {strat}")
        config.set_seed(config.SEED)

        model = models.build_cnn_model(vectorizer)

        cw = None
        X_tr, y_tr = X_train, y_train

        if strat == "class_weight":
            cw = data.get_class_weights(y_train)
        elif strat == "oversampling":
            X_tr, y_tr = data.oversample_minority_class(X_train, y_train)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=0
        )

        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            class_weight=cw,
            callbacks=[early_stopping],
            verbose=0
        )

        test_probs = model.predict(X_test, verbose=0).ravel()
        metrics = compute_all_metrics(y_test, test_probs, threshold=0.5)

        results.append({
            "Strategy": strat,
            "Accuracy": round(metrics["accuracy"] * 100, 2),
            "Precision": round(metrics["precision"] * 100, 2),
            "Recall": round(metrics["recall"] * 100, 2),
            "F1-Score": round(metrics["f1"], 4),
            "PR-AUC": round(metrics["pr_auc"], 4),
            "ROC-AUC": round(metrics["roc_auc"], 4),
            "False Positives (FP)": metrics["confusion_matrix"]["FP"],
            "False Negatives (FN)": metrics["confusion_matrix"]["FN"],
        })

    df_results = pd.DataFrame(results)
    out_path = config.ARTIFACTS_DIR / "imbalance_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Class Imbalance Comparison:\n{df_results.to_markdown(index=False)}")
    return df_results


if __name__ == "__main__":
    run_imbalance_experiment()
