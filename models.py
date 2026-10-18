"""
Model Architectures for SMS Spam Classification.

Includes:
1. Classical Baseline: TF-IDF + Multinomial Naive Bayes / Logistic Regression
2. FCC Model: Embedding + GlobalAveragePooling1D + Dense
3. 1D CNN: Local n-gram feature detection
4. Bidirectional LSTM: Sequential context learning with gating
5. Transformer-Lite: Multi-head self-attention with positional embeddings

"""

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers

import config



# 1. Classical Scikit-Learn Baseline

def build_baseline_tfidf(model_type: str = "nb") -> Pipeline:
    """
    TF-IDF feature extraction pipeline paired with Naive Bayes or Logistic Regression.

    """
    if model_type == "nb":
        clf = MultinomialNB(alpha=0.1)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.SEED)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=config.VOCAB_SIZE,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )),
        ("classifier", clf)
    ])
    return pipeline



# 2. TextVectorization Layer Setup

def create_text_vectorizer(train_texts: np.ndarray, vocab_size: int = config.VOCAB_SIZE, max_length: int = config.MAX_SEQUENCE_LENGTH) -> layers.TextVectorization:
    """
    Creates and adapts a Keras TextVectorization layer.

    """
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_length,
        standardize="lower_and_strip_punctuation"
    )
    # Adapt vectorizer
    vectorizer.adapt(train_texts)
    return vectorizer


# Helper: Standard Model Compiler

def _compile_model(model: keras.Model, learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """Compiles Keras model with Adam optimizer, binary crossentropy, and multi-metric tracking."""
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc", curve="ROC"),
            keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )
    return model


# 3. FCC Architecture

def build_original_model(
    vectorizer: layers.TextVectorization,
    embedding_dim: int = 32,
    dense_units: int = 24,
    learning_rate: float = config.LEARNING_RATE
) -> keras.Model:
    """
    Rebuilds freeCodeCamp's original model:
    Raw String -> TextVectorization -> Embedding -> GlobalAveragePooling1D -> Dense -> Output

    """
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="text_input")
    x = vectorizer(inputs)
    x = layers.Embedding(input_dim=config.VOCAB_SIZE, output_dim=embedding_dim, mask_zero=True, name="embedding")(x)
    x = layers.GlobalAveragePooling1D(name="avg_pooling")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FCC_Original_Upgraded")
    return _compile_model(model, learning_rate=learning_rate)


# 4. 1D Convolutional Neural Network (CNN)

def build_cnn_model(
    vectorizer: layers.TextVectorization,
    embedding_dim: int = config.EMBEDDING_DIM,
    filters: int = 64,
    kernel_size: int = 3,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = config.LEARNING_RATE
) -> keras.Model:
    """
    Builds a 1D CNN for Text Classification:
    Raw String -> TextVectorization -> Embedding -> Conv1D -> GlobalMaxPooling1D -> Dropout -> Dense -> Output

    """
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="text_input")
    x = vectorizer(inputs)
    x = layers.Embedding(input_dim=config.VOCAB_SIZE, output_dim=embedding_dim, name="embedding")(x)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation="relu", name="conv1d")(x)
    x = layers.GlobalMaxPooling1D(name="max_pooling")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN_1D")
    return _compile_model(model, learning_rate=learning_rate)


# 5. Bidirectional LSTM

def build_bilstm_model(
    vectorizer: layers.TextVectorization,
    embedding_dim: int = config.EMBEDDING_DIM,
    lstm_units: int = 32,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = config.LEARNING_RATE
) -> keras.Model:
    """
    Builds a Bidirectional LSTM Network:
    Raw String -> TextVectorization -> Embedding -> SpatialDropout1D -> BiLSTM -> Dense -> Output

    """
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="text_input")
    x = vectorizer(inputs)
    x = layers.Embedding(input_dim=config.VOCAB_SIZE, output_dim=embedding_dim, mask_zero=True, name="embedding")(x)
    x = layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=0.2), name="bilstm")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM")
    return _compile_model(model, learning_rate=learning_rate)


# 6. Transformer-Lite

class PositionalEmbedding(layers.Layer):
    """
    Combines token embeddings with learnable positional embeddings.

    """
    def __init__(self, max_length: int, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embed")
        self.position_embeddings = layers.Embedding(input_dim=max_length, output_dim=embed_dim, name="pos_embed")

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        token_embeds = self.token_embeddings(inputs)
        position_embeds = self.position_embeddings(positions)
        return token_embeds + position_embeds

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            "max_length": self.max_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config_dict


class TransformerBlock(layers.Layer):
    """
    Single Transformer Encoder Layer:
    MultiHeadAttention -> Add & Norm -> FeedForward -> Add & Norm

    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config_dict


def build_transformer_model(
    vectorizer: layers.TextVectorization,
    embedding_dim: int = 48,
    num_heads: int = 2,
    ff_dim: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.2,
    learning_rate: float = config.LEARNING_RATE
) -> keras.Model:
    """
    Builds a lightweight Transformer Encoder:
    Raw String -> TextVectorization -> PositionalEmbedding -> TransformerBlock -> GlobalAvgPool -> Dense -> Output
    """
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="text_input")
    x = vectorizer(inputs)
    x = PositionalEmbedding(max_length=config.MAX_SEQUENCE_LENGTH, vocab_size=config.VOCAB_SIZE, embed_dim=embedding_dim)(x)
    x = TransformerBlock(embed_dim=embedding_dim, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)(x)
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="spam_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Transformer_Lite")
    return _compile_model(model, learning_rate=learning_rate)


def get_model_by_name(name: str, vectorizer: layers.TextVectorization) -> keras.Model:
    """Factory function to build model by name."""
    name_clean = name.lower()
    if name_clean in ["original", "fcc"]:
        return build_original_model(vectorizer)
    elif name_clean in ["cnn", "conv1d"]:
        return build_cnn_model(vectorizer)
    elif name_clean in ["bilstm", "lstm", "rnn"]:
        return build_bilstm_model(vectorizer)
    elif name_clean in ["transformer", "transformer_lite"]:
        return build_transformer_model(vectorizer)
    else:
        raise ValueError(f"Unknown model name '{name}'. Choices: original, cnn, bilstm, transformer")
