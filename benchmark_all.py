"""
Master Benchmark Runner: Comprehensive Model Comparison, Evaluation & Error Analysis.

Trains and evaluates:
1. Baseline: TF-IDF + Multinomial Naive Bayes
2. Rebuilt Original FCC Model (TextVectorization + AvgPool)
3. 1D CNN
4. Bidirectional LSTM
5. Transformer-Lite

Outputs:
- artifacts/metrics.json
- artifacts/plots/calibration_curves.png
- artifacts/plots/threshold_tuning.png
- artifacts/plots/<model>_learning_curve.png
- Top 10 False Positives & False Negatives Error Audit
- 5-Fold Stratified Cross-Validation for Best Classical & Neural Models
- Saves artifacts/models/best_model.keras
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

import config
import data
import evaluate
import models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_full_benchmark():
    config.set_seed(config.SEED)

    # 1. Load leak-free 3-way stratified splits
    X_train, y_train, X_val, y_val, X_test, y_test = data.get_stratified_3way_splits(seed=config.SEED)

    # 2. In-graph TextVectorization layer adapted on X_train only
    logger.info("Fitting TextVectorization layer on training set...")
    vectorizer = models.create_text_vectorizer(X_train)

    # Class weights for training neural nets
    class_weights = data.get_class_weights(y_train)

    metrics_registry = {}
    test_predictions_dict = {}

    
    # 1. Classical Baseline: TF-IDF + Naive Bayes
    
    logger.info("=== 1/5 Training Classical Baseline (TF-IDF + Naive Bayes) ===")
    baseline_pipe = models.build_baseline_tfidf(model_type="nb")
    baseline_pipe.fit(X_train, y_train)

    joblib.dump(baseline_pipe, config.MODELS_DIR / "baseline_nb.joblib")

    baseline_pipe.predict_proba(X_val)[:, 1]
    test_probs_base = baseline_pipe.predict_proba(X_test)[:, 1]

    metrics_registry["Baseline (TF-IDF + Naive Bayes)"] = evaluate.compute_all_metrics(
        y_test, test_probs_base, threshold=config.DEFAULT_THRESHOLD
    )
    test_predictions_dict["Baseline (NB)"] = (y_test, test_probs_base)

   
    # 2. Rebuilt Original FCC Model
    
    logger.info("=== 2/5 Training Rebuilt Original FCC Model ===")
    config.set_seed(config.SEED)
    model_orig = models.build_original_model(vectorizer)
    cb_orig = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE_EARLY_STOPPING, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config.PATIENCE_REDUCE_LR, min_lr=1e-5, verbose=0)
    ]
    hist_orig = model_orig.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=15, batch_size=config.BATCH_SIZE, class_weight=class_weights,
        callbacks=cb_orig, verbose=0
    )
    evaluate.plot_learning_curves(hist_orig, "FCC_Original_Upgraded", config.PLOTS_DIR / "original_learning_curve.png")
    model_orig.save(config.MODELS_DIR / "original.keras")

    model_orig.predict(X_val, verbose=0).ravel()
    test_probs_orig = model_orig.predict(X_test, verbose=0).ravel()
    metrics_registry["Original Rebuilt (AvgPool)"] = evaluate.compute_all_metrics(
        y_test, test_probs_orig, threshold=config.DEFAULT_THRESHOLD
    )
    test_predictions_dict["Original (AvgPool)"] = (y_test, test_probs_orig)

   
    # 3. 1D CNN Model
   
    logger.info("=== 3/5 Training 1D CNN ===")
    config.set_seed(config.SEED)
    model_cnn = models.build_cnn_model(vectorizer)
    cb_cnn = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE_EARLY_STOPPING, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config.PATIENCE_REDUCE_LR, min_lr=1e-5, verbose=0)
    ]
    hist_cnn = model_cnn.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=15, batch_size=config.BATCH_SIZE, class_weight=class_weights,
        callbacks=cb_cnn, verbose=0
    )
    evaluate.plot_learning_curves(hist_cnn, "1D_CNN", config.PLOTS_DIR / "cnn_learning_curve.png")
    model_cnn.save(config.MODELS_DIR / "cnn.keras")

    model_cnn.predict(X_val, verbose=0).ravel()
    test_probs_cnn = model_cnn.predict(X_test, verbose=0).ravel()
    metrics_registry["1D CNN (N-Gram Filters)"] = evaluate.compute_all_metrics(
        y_test, test_probs_cnn, threshold=config.DEFAULT_THRESHOLD
    )
    test_predictions_dict["1D CNN"] = (y_test, test_probs_cnn)

   
    # 4. Bidirectional LSTM
    
    logger.info("=== 4/5 Training Bidirectional LSTM ===")
    config.set_seed(config.SEED)
    model_bilstm = models.build_bilstm_model(vectorizer)
    cb_bilstm = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE_EARLY_STOPPING, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config.PATIENCE_REDUCE_LR, min_lr=1e-5, verbose=0)
    ]
    hist_bilstm = model_bilstm.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=15, batch_size=config.BATCH_SIZE, class_weight=class_weights,
        callbacks=cb_bilstm, verbose=0
    )
    evaluate.plot_learning_curves(hist_bilstm, "BiLSTM", config.PLOTS_DIR / "bilstm_learning_curve.png")
    model_bilstm.save(config.MODELS_DIR / "bilstm.keras")

    val_probs_bilstm = model_bilstm.predict(X_val, verbose=0).ravel()
    test_probs_bilstm = model_bilstm.predict(X_test, verbose=0).ravel()
    metrics_registry["Bidirectional LSTM"] = evaluate.compute_all_metrics(
        y_test, test_probs_bilstm, threshold=config.DEFAULT_THRESHOLD
    )
    test_predictions_dict["BiLSTM"] = (y_test, test_probs_bilstm)

    
    # 5. Transformer-Lite
    
    logger.info("=== 5/5 Training Transformer-Lite ===")
    config.set_seed(config.SEED)
    model_trans = models.build_transformer_model(vectorizer)
    cb_trans = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE_EARLY_STOPPING, restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config.PATIENCE_REDUCE_LR, min_lr=1e-5, verbose=0)
    ]
    hist_trans = model_trans.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=15, batch_size=config.BATCH_SIZE, class_weight=class_weights,
        callbacks=cb_trans, verbose=0
    )
    evaluate.plot_learning_curves(hist_trans, "Transformer_Lite", config.PLOTS_DIR / "transformer_learning_curve.png")
    model_trans.save(config.MODELS_DIR / "transformer.keras")

    model_trans.predict(X_val, verbose=0).ravel()
    test_probs_trans = model_trans.predict(X_test, verbose=0).ravel()
    metrics_registry["Transformer-Lite (Self-Attention)"] = evaluate.compute_all_metrics(
        y_test, test_probs_trans, threshold=config.DEFAULT_THRESHOLD
    )
    test_predictions_dict["Transformer-Lite"] = (y_test, test_probs_trans)

    
    # Threshold Tuning on Best Model (Validation Set Sweep)
    
    logger.info("=== Decision Threshold Sweep & Cost Optimization (Validation Set) ===")
    best_threshold, df_threshold_sweep = evaluate.tune_decision_threshold(
        y_val, val_probs_bilstm, fp_cost_weight=3.0
    )
    evaluate.plot_threshold_curves(df_threshold_sweep, config.PLOTS_DIR / "threshold_tuning.png")

    # Evaluate BiLSTM
    metrics_registry["Bidirectional LSTM (Tuned Threshold)"] = evaluate.compute_all_metrics(
        y_test, test_probs_bilstm, threshold=best_threshold
    )

   
    # Reliability / Calibration Diagram
    
    logger.info("Generating Probability Calibration Diagram...")
    evaluate.plot_calibration_curves(test_predictions_dict, config.PLOTS_DIR / "calibration_curves.png")

    
    # Error Analysis: Top 10 False Positives & False Negatives
    
    logger.info("Performing In-Depth Error Analysis on Test Set...")
    error_report = evaluate.perform_error_analysis(
        texts=X_test,
        y_true=y_test,
        y_prob=test_probs_bilstm,
        threshold=best_threshold,
        top_n=10
    )
    with open(config.ARTIFACTS_DIR / "error_analysis.json", "w") as f:
        json.dump(error_report, f, indent=2)

    
    # Stratified 5-Fold Cross-Validation for Best Classical and Best Neural
    
    logger.info("Running Stratified 5-Fold Cross-Validation for Classical vs BiLSTM...")
    # Combine train + val for CV
    X_cv = np.concatenate([X_train, X_val])
    y_cv = np.concatenate([y_train, y_val])

    cv_baseline = evaluate.run_stratified_cross_validation(X_cv, y_cv, model_type="baseline", n_splits=5)
    cv_bilstm = evaluate.run_stratified_cross_validation(X_cv, y_cv, model_type="bilstm", n_splits=5)

    cv_results = {
        "baseline_naive_bayes_5fold": cv_baseline,
        "bilstm_5fold": cv_bilstm
    }
    with open(config.ARTIFACTS_DIR / "cross_validation.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    
    # Save Best Model to best_model.keras
    
    model_bilstm.save(config.BEST_MODEL_PATH)
    logger.info(f"Designated Bidirectional LSTM as production best_model.keras at {config.BEST_MODEL_PATH}")

    
    # Save Metrics JSON & Output Comparison Markdown Table
    
    with open(config.METRICS_FILE, "w") as f:
        json.dump(metrics_registry, f, indent=2)

    # Build markdown summary table
    summary_rows = []
    for model_name, m in metrics_registry.items():
        summary_rows.append({
            "Model Architecture": model_name,
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Recall": f"{m['recall'] * 100:.2f}%",
            "F1-Score": f"{m['f1']:.4f}",
            "PR-AUC": f"{m['pr_auc']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "Brier Score": f"{m['brier_score']:.4f}",
            "FP / FN": f"{m['confusion_matrix']['FP']} / {m['confusion_matrix']['FN']}"
        })

    df_summary = pd.DataFrame(summary_rows)
    logger.info("\n" + "=" * 90 + "\nOFFICIAL BENCHMARK EVALUATION RESULTS (TEST SET)\n" + "=" * 90)
    print(df_summary.to_markdown(index=False))
    print("=" * 90 + "\n")

    return df_summary, error_report, cv_results


if __name__ == "__main__":
    run_full_benchmark()
