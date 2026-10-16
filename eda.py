"""
Exploratory Data Analysis (EDA) Script: Generates publication-quality charts
and extracts real dataset statistics.

Saves:
- artifacts/plots/class_balance.png
- artifacts/plots/message_length_distribution.png
- artifacts/plots/top_keywords_per_class.png
"""

import logging
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

import config
import data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_eda_artifacts():
    train_df, valid_df = data.load_raw_data()
    df = pd.concat([train_df, valid_df], ignore_index=True)

    # 1. Class Balance Analysis
    counts = df["label"].value_counts()
    percentages = df["label"].value_counts(normalize=True) * 100

    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"], width=0.45)
    plt.title("SMS Dataset Class Balance (Minority Class: Spam)", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Messages", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 60, f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    class_plot_path = config.PLOTS_DIR / "class_balance.png"
    plt.savefig(class_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved class balance plot to {class_plot_path}")

    # 2. Message Length Distribution (Characters and Words)
    df["char_len"] = df["message"].apply(len)
    df["word_cnt"] = df["message"].apply(lambda s: len(str(s).split()))

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df[df["label"] == "ham"]["char_len"], bins=40, alpha=0.6, color="#2ecc71", label="Ham (Normal)", density=True)
    axes[0].hist(df[df["label"] == "spam"]["char_len"], bins=40, alpha=0.6, color="#e74c3c", label="Spam", density=True)
    axes[0].set_title("Message Character Length Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Character Count")
    axes[0].set_ylabel("Probability Density")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].hist(df[df["label"] == "ham"]["word_cnt"], bins=30, alpha=0.6, color="#2ecc71", label="Ham (Normal)", density=True)
    axes[1].hist(df[df["label"] == "spam"]["word_cnt"], bins=30, alpha=0.6, color="#e74c3c", label="Spam", density=True)
    axes[1].set_title("Message Word Count Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Probability Density")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    length_plot_path = config.PLOTS_DIR / "message_length_distribution.png"
    plt.savefig(length_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved length distribution plot to {length_plot_path}")

    # 3. Top Keywords Analysis
    df["cleaned"] = df["message"].apply(data.clean_text)

    # Filter generic stopwords
    stopwords = {"to", "you", "a", "i", "the", "and", "is", "in", "for", "of", "my", "it", "on", "me", "your", "that", "have", "at", "are", "do", "from", "with", "be"}
    spam_tokens = [w for w in " ".join(df[df["label"] == "spam"]["cleaned"]).split() if w not in stopwords and len(w) > 1]
    ham_tokens = [w for w in " ".join(df[df["label"] == "ham"]["cleaned"]).split() if w not in stopwords and len(w) > 1]

    top_spam = Counter(spam_tokens).most_common(12)
    top_ham = Counter(ham_tokens).most_common(12)

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    s_words, s_counts = zip(*top_spam)
    axes[0].barh(s_words[::-1], s_counts[::-1], color="#e74c3c")
    axes[0].set_title("Top Spam Predictive Tokens (Normalized)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Frequency")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    h_words, h_counts = zip(*top_ham)
    axes[1].barh(h_words[::-1], h_counts[::-1], color="#2ecc71")
    axes[1].set_title("Top Ham Conversational Tokens", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Frequency")
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    keywords_plot_path = config.PLOTS_DIR / "top_keywords_per_class.png"
    plt.savefig(keywords_plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved top keywords plot to {keywords_plot_path}")

    stats = {
        "total_messages": len(df),
        "ham_count": int(counts["ham"]),
        "spam_count": int(counts["spam"]),
        "spam_percentage": float(percentages["spam"]),
        "ham_avg_char_length": float(df[df["label"] == "ham"]["char_len"].mean()),
        "spam_avg_char_length": float(df[df["label"] == "spam"]["char_len"].mean()),
        "top_spam_tokens": top_spam[:5],
        "top_ham_tokens": top_ham[:5]
    }
    logger.info(f"EDA Summary Stats: {stats}")
    return stats


if __name__ == "__main__":
    generate_eda_artifacts()
