"""
Inference API & FreeCodeCamp Verification Interface.

Provides:
1. `predict_message(pred_text)`: Returns `[probability, "spam" | "ham"]` (exact FCC signature).
2. CLI interface for quick interactive predictions.
3. `test_predictions()`: Exact FCC verification suite.
"""

import argparse
import logging
import sys

import numpy as np
from tensorflow import keras

import config
import data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global cache for the loaded model
_CACHED_MODEL = None
_CACHED_THRESHOLD = config.DEFAULT_THRESHOLD


def get_inference_model() -> keras.Model:
    """
    Loads and caches the best saved Keras model.
    Falls back to building and training a quick model if no saved model is found.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    # Check for best_model.keras, or any trained model in artifacts/models
    candidate_paths = [
        config.BEST_MODEL_PATH,
        config.MODELS_DIR / "bilstm.keras",
        config.MODELS_DIR / "cnn.keras",
        config.MODELS_DIR / "original.keras",
        config.MODELS_DIR / "transformer.keras"
    ]

    for model_file in candidate_paths:
        if model_file.exists():
            try:
                # Custom objects for Transformer if needed
                import models
                custom_objects = {
                    "PositionalEmbedding": models.PositionalEmbedding,
                    "TransformerBlock": models.TransformerBlock
                }
                _CACHED_MODEL = keras.models.load_model(model_file, custom_objects=custom_objects)
                logger.info(f"Loaded trained model from {model_file.name}")
                return _CACHED_MODEL
            except Exception as e: # noqa: BLE001
                logger.warning(f"Could not load {model_file}: {e}")

    # If no model found on disk, train a lightweight model on the fly
    logger.info("No saved model found. Training lightweight FCC model for instant inference...")
    import models
    X_train, y_train, X_val, y_val, _, _ = data.get_stratified_3way_splits()
    vec = models.create_text_vectorizer(X_train)
    _CACHED_MODEL = models.build_original_model(vec)
    cw = data.get_class_weights(y_train)
    _CACHED_MODEL.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, class_weight=cw, verbose=0)
    _CACHED_MODEL.save(config.BEST_MODEL_PATH)
    return _CACHED_MODEL


def predict_message(pred_text: str, threshold: float = config.DEFAULT_THRESHOLD) -> list[float | str]:
    """
    Predicts whether a message is spam or ham.

    Signature required by freeCodeCamp:
    Returns [probability, "spam" or "ham"]
    """
    model = get_inference_model()

    # Apply token normalization (<URL>, <PHONE>, <MONEY>)
    cleaned = data.clean_text(pred_text)

    # Input tensor of shape (1,)
    input_tensor = np.array([cleaned], dtype=object)

    raw_prob = float(model.predict(input_tensor, verbose=0)[0][0])
    label = "spam" if raw_prob >= threshold else "ham"

    return [raw_prob, label]


def test_predictions() -> bool:
    """
    Official freeCodeCamp validation suite.
    Runs the 7 standard test messages and prints result.
    """
    test_messages = [
        "how are you doing today",
        "sale today! to stop texts call 98912460324",
        "i dont want to go. can we try it a different day? available sat",
        "our new mobile video service is live. just install on your phone to start watching.",
        "you have won £1000 cash! call to claim your prize.",
        "i'll bring it tomorrow. don't forget the milk.",
        "wow, is your arm alright. that happened to me one time too"
    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]

    passed = True
    print("\n" + "=" * 65)
    print("RUNNING FREECODECAMP VERIFICATION SUITE")
    print("=" * 65)

    for msg, expected in zip(test_messages, test_answers):
        prob, pred_label = predict_message(msg)
        is_correct = (pred_label == expected)
        status = "PASSED [OK]" if is_correct else "FAILED [X]"
        if not is_correct:
            passed = False
        print(f"[{status}] Prob: {prob:6.4f} | Pred: {pred_label:4s} | Expected: {expected:4s} | Text: '{msg}'")

    print("=" * 65)
    if passed:
        print("RESULT: You passed the challenge. Great job!")
    else:
        print("RESULT: You haven't passed yet. Keep trying.")
    print("=" * 65 + "\n")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spam/ham for an SMS text")
    parser.add_argument("text", nargs="?", type=str, help="Text message to classify")
    parser.add_argument("--test", action="store_true", help="Run freeCodeCamp verification tests")
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD, help="Classification threshold")
    args = parser.parse_args()

    if args.test or not args.text:
        success = test_predictions()
        sys.exit(0 if success else 1)

    result = predict_message(args.text, threshold=args.threshold)
    print(f"\nMessage: '{args.text}'")
    print(f"Spam Probability: {result[0]:.4f}")
    print(f"Classification:   {result[1].upper()}\n")


if __name__ == "__main__":
    main()
