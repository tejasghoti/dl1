import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec

# Load text file and split into sentences on full stop
text = open("document3.txt").read()
print(text)
lines = text.split(".")

# Clean text: lowercase, keep only letters, split into words, remove stop words
def clean_text(text):
    text = re.sub(r"[^a-z\s]", "", str(text).lower())
    return [w for w in text.split() if w not in ENGLISH_STOP_WORDS]

sentences = [clean_text(line) for line in lines if line.strip() != ""]
print(sentences[0])

# Train CBOW (sg=0 means CBOW) -- small text, so min_count=1 and more epochs
w2v = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=0, epochs=100)
print("Vocabulary size:", len(w2v.wv))

# Evaluate: check learned embeddings with examples
print("Vector of 'covid':", w2v.wv["covid"])
print("Similar to 'covid':", w2v.wv.most_similar("covid", topn=5))
print("Similar to 'influenza':", w2v.wv.most_similar("influenza", topn=5))
print("Similarity (covid, influenza):", w2v.wv.similarity("covid", "influenza"))
