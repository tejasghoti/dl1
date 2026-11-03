import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec

from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("twitter_dataset.csv")
print(df.shape)
print(df.head())

# Clean text: lowercase, keep only letters, split into words, remove stop words
def clean_text(text):
    text = re.sub(r"[^a-z\s]", "", str(text).lower())
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]

sentences = df["Text"].apply(clean_text).tolist()
print(sentences[0])

# Train CBOW (sg=0 means CBOW)
w2v = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, sg=0, epochs=10)
print("Vocabulary size:", len(w2v.wv))

# Document vector = average of all word vectors in the document
def document_vector(sentence):
    words = [w for w in sentence if w in w2v.wv]
    if len(words) == 0:
        return np.zeros(100)
    return np.mean(w2v.wv[words], axis=0)

doc_vectors = np.array([document_vector(s) for s in sentences])

# Recommend: cosine similarity between user input and every tweet
def recommend(user_input, top_n=5):
    input_vector = document_vector(clean_text(user_input))
    scores = cosine_similarity([input_vector], doc_vectors)[0]
    top = np.argsort(scores)[::-1][:top_n]
    for i in top:
        print(round(scores[i], 3), "->", df["Text"][i])

user_input = "money economy business"
print("Recommendations for:", user_input)
recommend(user_input)
