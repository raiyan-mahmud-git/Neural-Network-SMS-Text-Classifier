# import libraries
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import os

print(tf.__version__)

# download data if not exists
if not os.path.exists("train-data.tsv"):
    os.system("wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv")
if not os.path.exists("valid-data.tsv"):
    os.system("wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv")

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# load datasets
train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# split data
train_data = train_df['message']
train_labels = train_df['label']

test_data = test_df['message']
test_labels = test_df['label']

# convert labels
train_labels = np.array([1 if label == "spam" else 0 for label in train_labels])
test_labels = np.array([1 if label == "spam" else 0 for label in test_labels])

# preprocessing
vocab_size = 10000
max_length = 120

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data)

train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding='post')

# build model
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 32),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# train
model.fit(
    train_padded,
    train_labels,
    epochs=15,
    validation_data=(test_padded, test_labels),
    verbose=2
)

# prediction function
def predict_message(pred_text):
    seq = tokenizer.texts_to_sequences([pred_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, padding='post')

    prediction = model.predict(padded, verbose=0)[0][0]
    label = "spam" if prediction >= 0.5 else "ham"

    return [prediction, label]

# test function (FCC requirement)
def test_predictions():
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

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

# run test
test_predictions()