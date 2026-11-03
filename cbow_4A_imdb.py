import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")
print(df.shape)
print(df.head())

# Clean text: lowercase, keep only letters, split into words, remove stop words
def clean_text(text):
    text = re.sub(r"[^a-z\s]", "", str(text).lower())
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]

sentences = df["review"].apply(clean_text).tolist()
print(sentences[0])

# Encode labels (negative = 0, positive = 1)
encoder = LabelEncoder()
y = encoder.fit_transform(df["sentiment"])

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=42)

# Train CBOW (sg=0 means CBOW)
w2v = Word2Vec(sentences=x_train, vector_size=100, window=5, min_count=2, sg=0, epochs=10)
print("Vocabulary size:", len(w2v.wv))

# Document vector = average of all word vectors in the document
def document_vector(sentence):
    words = [w for w in sentence if w in w2v.wv]
    if len(words) == 0:
        return np.zeros(100)
    return np.mean(w2v.wv[words], axis=0)

X_train = np.array([document_vector(s) for s in x_train])
X_test = np.array([document_vector(s) for s in x_test])

# Feed forward neural network
model = Sequential([
    Dense(64, activation="relu", input_shape=(100,)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
