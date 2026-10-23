"""
Robustness & Adversarial Testing Suite for SMS Spam Classifier.

Tests:
1. Obfuscated / Leetspeak spam ("fr33 pr1ze", "c@sh", "w!n")
2. Spacing evasion ("f r e e", "c a l l   n o w")
3. Mixed-case evasion ("fReE", "PrIzE")
4. Emoji injection ("🎁 Win $1000 cash! 💰 Click now!")
5. Out-of-Distribution (OOD): Code snippets, Shakespeare, legal disclaimers.

"""

import logging
import re

from predict import predict_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LEET_MAP = {
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "@": "a", "$": "s", "!": "i"
}

ADVERSARIAL_SPAM_SAMPLES = [
    ("Standard Spam", "congratulations you have won a 1000 cash prize call 09061701461 to claim"),
    ("Leetspeak Spam", "c0ngratulati0ns y0u have w0n a 1000 c@sh pr1ze c@ll 09061701461 t0 cla1m"),
    ("Spaced Out Spam", "c o n g r a t u l a t i o n s   y o u   h a v e   w o n   a   p r i z e"),
    ("Mixed Case Evasion", "CoNgRaTuLaTiOnS YoU HaVe WoN A 1000 CaSh PrIzE CaLl 09061701461"),
    ("Emoji Injected Spam", "🎉 CONGRATS! 🎁 You won £1000 cash! 💰 Call 09061701461 now! 🚀"),
    ("URL Shortener Spam", "Urgent alert: your parcel is held. Claim at http://bit.ly/spam-gift-99"),
]

OUT_OF_DISTRIBUTION_SAMPLES = [
    ("Python Code", "def calculate_loss(y_true, y_pred): return -np.mean(y_true * np.log(y_pred))"),
    ("Shakespeare Quote", "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer"),
    ("Legal Disclaimer", "This email and any attachments are confidential and intended solely for the addressee."),
    ("Git Commit Message", "fix(data): resolve off-by-one error in stratified k-fold cross validation split"),
]


def deobfuscate_leetspeak(text: str) -> str:
    """
    Applies character-level deobfuscation.
    """
    normalized = []
    for char in text.lower():
        normalized.append(LEET_MAP.get(char, char))
    reconstructed = "".join(normalized)

    reconstructed = re.sub(r"(?<=\b\w)\s(?=\w\b)", "", reconstructed)
    return reconstructed


def run_robustness_benchmark() -> dict[str, list[dict[str, any]]]:
    """Evaluates the model on adversarial perturbations and OOD samples."""
    logger.info("--- Evaluating Adversarial Robustness ---")

    adv_results = []
    for category, sample in ADVERSARIAL_SPAM_SAMPLES:
        raw_prob, raw_label = predict_message(sample)

        deobf_text = deobfuscate_leetspeak(sample)
        mitigated_prob, mitigated_label = predict_message(deobf_text)

        adv_results.append({
            "category": category,
            "sample": sample,
            "raw_prob": raw_prob,
            "raw_label": raw_label,
            "mitigated_prob": mitigated_prob,
            "mitigated_label": mitigated_label,
        })
        logger.info(f"[{category}] Raw: {raw_label} ({raw_prob:.4f}) | Mitigated: {mitigated_label} ({mitigated_prob:.4f})")

    logger.info("--- Evaluating Out-of-Distribution (OOD) Samples ---")
    ood_results = []
    for category, sample in OUT_OF_DISTRIBUTION_SAMPLES:
        prob, label = predict_message(sample)
        ood_results.append({
            "category": category,
            "sample": sample,
            "predicted_prob": prob,
            "predicted_label": label
        })
        logger.info(f"[{category}] Predicted: {label} (Prob: {prob:.4f})")

    return {"adversarial": adv_results, "ood": ood_results}


if __name__ == "__main__":
    run_robustness_benchmark()
