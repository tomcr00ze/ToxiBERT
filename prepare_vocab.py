import os
import pandas as pd
import pickle
import string
from collections import Counter

print("1. Loading dataset...")
csv_path = os.path.join('jigsaw-toxic-comment-classification-challenge', 'train.csv', 'train.csv')
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found.")
    exit(1)

df = pd.read_csv(csv_path)

print("2. Cleaning text and counting word frequencies (pure Python, fast)...")
# Prepare punctuation translation table (removes all string.punctuation characters)
translator = str.maketrans('', '', string.punctuation)

counter = Counter()

# Loop through all comments and count words
for text in df['comment_text'].astype(str).values:
    # Lowercase and remove punctuation
    clean_text = text.lower().translate(translator)
    # Split by whitespace and count
    counter.update(clean_text.split())

print("3. Building vocabulary list...")
# Keras TextVectorization defaults:
# Index 0 is reserved for padding: ""
# Index 1 is reserved for Out Of Vocabulary: "[UNK]"
# The rest are the most common words up to MAX_FEATURES - 2
MAX_FEATURES = 200000
most_common = counter.most_common(MAX_FEATURES - 2)

vocab = ["", "[UNK]"] + [word for word, count in most_common]

print(f"Total unique words found: {len(counter)}")
print(f"Vocabulary size: {len(vocab)}")

print("4. Saving to 'vocab.pkl'...")
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print("Success! Vocabulary created in seconds.")
