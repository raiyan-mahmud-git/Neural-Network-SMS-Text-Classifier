# Neural-Network-SMS-Text-Classifier

This project builds a machine learning model to classify SMS messages as either spam or ham (not spam) using TensorFlow and natural language processing techniques.
The dataset is provided by freeCodeCamp and contains labeled SMS messages. The model processes text data by converting messages into numerical sequences using tokenization and padding. It then trains a neural network with an embedding layer and dense layers to learn patterns in spam messages.

After training, the model can predict whether a new message is spam or not through the predict_message function, which returns both the prediction probability and the final label.

The model is evaluated using a predefined test set, and it successfully classifies unseen messages, meeting the requirements of the freeCodeCamp challenge.

# Technologies Used
Python
TensorFlow / Keras
NumPy
Pandas

# Features
Text preprocessing (tokenization and padding)
Binary classification model
Spam prediction function
Automated testing for validation

# Result
The model achieves high accuracy on validation data and correctly classifies the test messages required to pass the challenge.
