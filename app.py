import os
import pandas as pd
import tensorflow as tf
import numpy as np
import gradio as gr
import pickle
from tensorflow.keras.layers import TextVectorization

MAX_FEATURES = 200000 
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')

# Labels predicted by the model
y_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

vocab_path = 'vocab.pkl'
if os.path.exists(vocab_path):
    print("Loading pre-calculated vocabulary from 'vocab.pkl' (instant)...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vectorizer.set_vocabulary(vocab)
    print("Vocabulary loaded successfully!")
else:
    print("Pre-calculated vocabulary not found. Building it from dataset...")
    csv_path = os.path.join('jigsaw-toxic-comment-classification-challenge', 'train.csv', 'train.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df['comment_text']
        print("Adapting vectorizer (this will take 1-2 minutes)...")
        vectorizer.adapt(X.values)
        print("Vocabulary built! Saving to 'vocab.pkl' for next time...")
        vocab = vectorizer.get_vocabulary()
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
    else:
        print(f"Error: {csv_path} not found. Cannot build vocabulary.")
        exit(1)

print("Vocabulary built successfully!")
print("Loading model weights...")

# 2. Recreate Model Architecture and Load Weights (Keras 3 compatibility)
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(MAX_FEATURES + 1, 32),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh')),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')
])
model.build(input_shape=(None, 1800))
model.load_weights('toxicity.h5')

# 3. Define Score Function
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(y_columns):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)
    
    return text

# 4. Launch Gradio
print("Launching Gradio Web Interface...")
interface = gr.Interface(fn=score_comment, 
                         inputs=gr.components.Textbox(lines=2, placeholder='Comment to score'),
                         outputs='text')

interface.launch(share=True)
