# 🛡️ Production SMS Spam Classifier & Explainability Pipeline

[![CI Pipeline](https://github.com/example/sms-spam-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/example/sms-spam-classifier/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.x%20%2F%20Keras%203-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-ready Natural Language Processing (NLP) system that transforms the classic **freeCodeCamp SMS Spam Classifier** into a modular, portfolio-grade deep learning project.

---

## 📌 1. Problem Statement & Motivation

SMS spam is not merely an annoyance; it is a vector for smishing, financial fraud, and credential theft. Building an effective spam filter requires solving three real-world challenges:

1. **Severe Class Imbalance**: In real-world communication (~86.6% Ham vs ~13.4% Spam), a trivial model predicting "Ham" every time scores **86.6% accuracy** while letting 100% of malicious messages through.
2. **Asymmetric Error Costs**: A **False Positive** (blocking a bank 2FA code or hospital appointment) destroys user trust; a **False Negative** (a spam promotion reaching the inbox) is a mild annoyance.
3. **Adversarial Obfuscation**: Attackers deliberately disguise keywords (`"fr33 pr1ze"`, `"w i n"`, emoji spam) to bypass simple keyword filters.

This project implements modern deep learning architectures, in-graph tokenization, automated threshold tuning, probability calibration, and word-level occlusion explainability.

---

## 🏗️ 2. System Architecture

```mermaid
flowchart TD
    A["Raw SMS Text Input"] --> B["Domain Token Normalization\n(<URL>, <PHONE>, <MONEY>)"]
    B --> C["In-Graph TextVectorization Layer\n(Embedded in Computation Graph)"]
    C --> D["Token & Positional Embeddings"]
    D --> E["Feature Extraction Block\n(TF-IDF Baseline / 1D CNN / BiLSTM / Transformer-Lite)"]
    E --> F["Dense Layers + Dropout + LayerNorm"]
    F --> G["Sigmoid Output: P(Spam) in [0.0, 1.0]"]
    G --> H{"Cost-Tuned Threshold T\n(Default: 0.50 | Tuned: 0.95)"}
    H -->|P >= T| I["🚨 Flagged as SPAM"]
    H -->|P < T| J["✅ Delivered as HAM"]
    G --> K["Word-Level Occlusion Attribution\n(Explainability Engine)"]
```

### Why Preprocessing Lives Inside the Model (`TextVectorization`)
In legacy Keras code, `tf.keras.preprocessing.text.Tokenizer` lived as a separate Python pickle object. This introduced **train-serve skew**: if the tokenizer file was missing or out of sync with production, the model would fail silently or produce nonsense predictions. By moving tokenization into Keras 3's native `TextVectorization` layer, the entire pipeline is self-contained: the saved `.keras` or `.tflite` model ingests raw strings directly.

---

## 📊 3. Official Benchmark Results (Test Set)

All models were evaluated on the exact same leak-free, stratified test split ($N = 774$, 97 Spam, 677 Ham):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Brier Score | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1D CNN (N-Gram Filters)** | **99.10%** | **94.12%** | **98.97%** | **0.9648** | **0.9958** | **0.9994** | **0.0090** | 6 | **1** |
| **Baseline (TF-IDF + Naive Bayes)** | 98.97% | **96.84%** | 94.85% | 0.9583 | 0.9894 | 0.9980 | 0.0091 | **3** | 5 |
| **Bidirectional LSTM (Tuned T=0.95)**| 98.84% | 92.31% | **98.97%** | 0.9552 | 0.9903 | 0.9986 | 0.0190 | 8 | **1** |
| **Transformer-Lite (Self-Attention)**| 98.58% | 90.57% | **98.97%** | 0.9458 | 0.9939 | 0.9991 | 0.0112 | 10 | **1** |
| **Bidirectional LSTM (Default T=0.50)**| 97.80% | 85.09% | **100.00%** | 0.9194 | 0.9903 | 0.9986 | 0.0190 | 17 | **0** |
| **Original Rebuilt (AvgPool Unweighted)**| 12.53% | 12.53% | 100.00% | 0.2227 | 0.1253 | 0.5000 | 0.2501 | 677 | 0 |

> **Key Architectural Takeaways**:
> 1. **1D CNN** achieves the best overall balance (**96.48% F1**, **0.9958 PR-AUC**), missing only a single spam message out of 97 while keeping false alarms to 6. Convolutions act as local trigram detectors, catching signatures like `"claim your prize"` anywhere in the text.
> 2. **Naive Bayes** sets an exceptionally high bar (98.97% accuracy, 96.84% precision), proving why classical baselines must always be benchmarked first.
> 3. **Threshold Tuning**: Adjusting the decision threshold from 0.50 to 0.95 on the BiLSTM cut False Positives by **over 52%** (from 17 down to 8) while preserving **98.97% recall**.

---

## ⚖️ 4. Class Imbalance Mitigation Study

On an imbalanced dataset (86.6% Ham), how you penalize errors determines operational behavior:

| Strategy | Accuracy | Precision | Recall | F1-Score | PR-AUC | False Negatives (Missed Spam) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Handling (Standard Loss)** | 98.32% | 92.00% | 94.85% | 0.9340 | 0.9878 | 5 missed |
| **Algorithmic `class_weight`** | **99.10%** | **94.12%** | **98.97%** | **0.9648** | **0.9958** | **1 missed** |
| **Oversampling Minority Class** | 98.84% | 94.00% | 96.91% | 0.9543 | 0.9933 | 3 missed |

**Why `class_weight` wins**:
Reweighting modifies the gradient update step by assigning $\approx 6.5\times$ greater penalty to errors on the spam minority class without duplicating data or bloating epoch training duration.

---

## 🔍 5. Error Analysis & Failure Modes

Inspecting aggregate metrics hides specific failure modes. We audited the top False Positives and False Negatives:

### False Positives (Legitimate Ham Flagged as Spam)
1. *"please protect yourself from e-threats. sib never asks for sensitive information like passwords,atm/sms pin thru email. never share your password with anybody."* (Prob: `0.9999`)
   - **Root Cause**: This is a genuine bank security warning. Because its vocabulary (`"sensitive information"`, `"password"`, `"sms pin"`, `"email"`) overlaps heavily with phishing lures, the model flags it.
   - **Remediation**: Whitelisting official shortcodes or adding transactional security warning heuristics.
2. *"hey...great deal...farm tour 9am to 5pm <MONEY> /pax, <MONEY> deposit by 16 may"* (Prob: `0.9992`)
   - **Root Cause**: A friend discussing group trip pricing with deadlines and currency symbols.

### False Negatives (Spam Escaping to Inbox)
1. *"dear voucher holder 2 claim your 1st class airport lounge passes when using your holiday voucher call <PHONE> . when booking quote 1st class x 2"* (Prob: `0.9098`)
   - **Root Cause**: When the threshold is tuned to $T=0.95$ to strictly eliminate false alarms, messages scoring ~0.91 slip through. At standard $T=0.50$, it is caught instantly.

---

## 🛡️ 6. Robustness & Adversarial Testing

Attackers use leetspeak, spacing, and emojis to evade text filters:

| Attack Category | Adversarial Message | Raw Prediction | Mitigated Prediction | Defense Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **Spaced-Out Spam** | `"c o n g r a t u l a t i o n s   y o u   w o n"` | ❌ HAM (`0.0048`) | ✅ **SPAM (`0.9456`)** | Spacing collapse regex |
| **Leetspeak** | `"c0ngratulati0ns y0u have w0n a c@sh pr1ze"` | ✅ SPAM (`0.9994`) | ✅ **SPAM (`0.9999`)** | Character mapping (`0->o, @->a`) |
| **Emoji Injection** | `"🎉 CONGRATS! 🎁 You won £1000 cash! 💰"` | ✅ SPAM (`0.9999`) | ✅ **SPAM (`0.9987`)** | In-graph entity normalization |
| **Out-of-Domain Code** | `"def calculate_loss(y_true, y_pred): ..."` | ✅ HAM (`0.4356`) | ✅ HAM (`0.4356`) | Low perplexity on conversational vocab |

---

## ⚡ 7. Deployment: Format, Latency & Edge Optimization

We benchmarked model export formats across 100 benchmark inferences:

| Model Format | Target Environment | Artifact Size | Latency per Sample | Throughput |
| :--- | :--- | :---: | :---: | :---: |
| **Keras 3 Archive (`.keras`)** | Cloud Server / Python Backend | ~1,200 KB | ~8.4 ms | ~119 samples/sec |
| **TensorFlow Lite (`.tflite`)** | Mobile (iOS/Android) / Edge IoT | **~420 KB** | **~3.8 ms** | **~263 samples/sec** |

---

## 🚀 8. Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/example/sms-spam-classifier.git
cd sms-spam-classifier
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify FreeCodeCamp Tests
```bash
python predict.py --test
```
Output:
```
=================================================================
RUNNING FREECODECAMP VERIFICATION SUITE
=================================================================
[PASSED [OK]] Prob: 0.0036 | Pred: ham  | Expected: ham  | Text: 'how are you doing today'
[PASSED [OK]] Prob: 0.9825 | Pred: spam | Expected: spam | Text: 'sale today! to stop texts call 98912460324'
...
RESULT: You passed the challenge. Great job!
=================================================================
```

### 3. Run Unit Tests (Pytest)
```bash
pytest tests/ -v
```

### 4. Train Models via CLI
```bash
# Train Bidirectional LSTM
python train.py --model bilstm --epochs 15 --imbalance class_weight

# Train 1D CNN
python train.py --model cnn --epochs 15

# Train classical baseline
python train.py --model baseline
```

### 5. Launch Interactive Streamlit App
```bash
streamlit run app.py
```

### 6. Run via Docker
```bash
docker build -t sms-classifier .
docker run -p 8000:8000 sms-classifier
# Test endpoint: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"message": "Win $1000 now!"}'
```

---

## 📈 9. Project Directory Structure

```
sms-spam-classifier/
├── README.md                  # Comprehensive portfolio documentation
├── requirements.txt           # Pinned production dependencies
├── config.py                  # Central hyperparameter & path configurations
├── data.py                    # Cross-platform download, regex normalizer, 3-way split
├── models.py                  # Baseline, CNN, BiLSTM, and Transformer-Lite architectures
├── train.py                   # CLI training pipeline with EarlyStopping & ReduceLROnPlateau
├── evaluate.py                # Metrics, calibration curves, threshold tuning, occlusion XAI
├── benchmark_all.py           # Master benchmarking suite for all models
├── predict.py                 # FreeCodeCamp verification & importable predict_message()
├── robustness.py              # Adversarial leetspeak and out-of-distribution evaluation
├── tune.py                    # Systematic hyperparameter optimization sweep
├── export.py                  # .keras and TFLite export & latency benchmarking
├── app.py                     # Interactive Streamlit dashboard with word attribution
├── api.py                     # Production FastAPI REST microservice with Pydantic validation
├── Dockerfile                 # Containerization specification
├── pytest.ini                 # Pytest configuration
├── notebooks/
│   └── exploration.ipynb      # Complete exploratory data analysis & visualization
├── tests/
│   └── test_predict.py        # 17 unit and edge case tests
└── artifacts/
    ├── models/                # Saved .keras & .tflite weights
    ├── plots/                 # High-resolution ROC/PR, loss, and calibration charts
    └── metrics.json           # Comprehensive evaluation registry
```
