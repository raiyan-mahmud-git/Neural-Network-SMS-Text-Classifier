"""
Unit and Integration Test Suite for SMS Spam Classifier.

Tests:
1. Exact 7 FreeCodeCamp benchmark test cases
2. Output format & probability range [0.0, 1.0]
3. Edge cases: empty string, whitespace only
4. Stress tests: extremely long messages (1,000+ words)
5. Out-of-domain / Non-English inputs
6. Preprocessing regex token normalizations (<URL>, <PHONE>, <MONEY>)
"""

import numpy as np
import pytest

import config
from data import clean_text
from predict import predict_message

FCC_TEST_CASES = [
    ("how are you doing today", "ham"),
    ("sale today! to stop texts call 98912460324", "spam"),
    ("i dont want to go. can we try it a different day? available sat", "ham"),
    ("our new mobile video service is live. just install on your phone to start watching.", "spam"),
    ("you have won £1000 cash! call to claim your prize.", "spam"),
    ("i'll bring it tomorrow. don't forget the milk.", "ham"),
    ("wow, is your arm alright. that happened to me one time too", "ham"),
]


@pytest.mark.parametrize("message, expected_label", FCC_TEST_CASES)
def test_fcc_official_cases(message: str, expected_label: str):
    """Verifies that all 7 official FCC test messages are correctly classified."""
    result = predict_message(message)
    prob, label = result[0], result[1]

    assert isinstance(prob, (float, np.floating)), f"Probability must be float, got {type(prob)}"
    assert label in ["spam", "ham"], f"Label must be 'spam' or 'ham', got {label}"
    assert label == expected_label, f"Message '{message}' predicted as '{label}', expected '{expected_label}' (Prob: {prob:.4f})"




def test_prediction_output_structure():
    """Ensures predict_message adheres strictly to [probability, label]."""
    result = predict_message("Check this out!")
    assert isinstance(result, list), "Return value must be a list"
    assert len(result) == 2, "Return value must have exactly 2 elements"
    prob, label = result[0], result[1]
    assert 0.0 <= prob <= 1.0, f"Probability must lie in [0.0, 1.0], got {prob}"
    assert label in ["spam", "ham"], f"Label must be 'spam' or 'ham', got {label}"




@pytest.mark.parametrize("empty_input", ["", "   ", "\n\t\r"])
def test_empty_and_whitespace_inputs(empty_input: str):
    """Empty or whitespace-only inputs should not throw exceptions and should default to ham."""
    result = predict_message(empty_input)
    prob, label = result[0], result[1]
    assert 0.0 <= prob <= 1.0
    assert label == "ham", f"Empty text should not be flagged as spam, got {label}"




def test_very_long_message():
    """Models must handle messages exceeding max_length without dimension or shape mismatch errors."""
    long_msg = "hello friend how are you " * 500
    result = predict_message(long_msg)
    assert len(result) == 2
    assert 0.0 <= result[0] <= 1.0




@pytest.mark.parametrize("foreign_text", [
    "Hola amigo, ¿quieres ir a almorzar mañana?",
    "Bonjour mon ami, comment ça va aujourd'hui?",
    "你好，今天晚上一起去吃饭吗？",
    "Guten Tag, treffen wir uns heute Nachmittag?",
])
def test_multilingual_inputs(foreign_text: str):
    """Non-English text should be handled gracefully without decoding or OOV crashes."""
    result = predict_message(foreign_text)
    assert len(result) == 2
    assert 0.0 <= result[0] <= 1.0




def test_clean_text_normalizations():
    """Verifies that regex normalization reliably extracts spam-indicative entity tokens."""
    url_text = "Visit https://spamprize.com or www.freegift.org for your reward"
    cleaned_url = clean_text(url_text)
    assert config.URL_TOKEN in cleaned_url
    assert "https://" not in cleaned_url

    phone_text = "Call 08712460324 or +447808726822 right now"
    cleaned_phone = clean_text(phone_text)
    assert config.PHONE_TOKEN in cleaned_phone

    money_text = "Claim your £1000 cash or $500 reward today"
    cleaned_money = clean_text(money_text)
    assert config.MONEY_TOKEN in cleaned_money
