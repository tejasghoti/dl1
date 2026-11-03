import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec

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

# Evaluate: check learned embeddings with examples
print("Vector of 'money':", w2v.wv["money"])
print("Similar to 'money':", w2v.wv.most_similar("money", topn=5))
print("Similar to 'film':", w2v.wv.most_similar("film", topn=5))
print("Similarity (money, economy):", w2v.wv.similarity("money", "economy"))
