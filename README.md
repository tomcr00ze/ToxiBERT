# ☢️ ToxiBERT: Toxic Comment Classification

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)

Welcome to **ToxiBERT**! 🚀 This project is an end-to-end Deep Learning application designed to detect and classify toxic text online. It is trained on the Jigsaw Toxic Comment Classification Challenge dataset and predicts multiple types of toxicity.

---

## ✨ Features

- **Multi-Label Classification:** Predicts 6 different categories of toxicity simultaneously:
  - 🤬 Toxic
  - 👿 Severe Toxic
  - 🛑 Obscene
  - 🔪 Threat
  - 🖕 Insult
  - 🚫 Identity Hate
- **Deep Learning Architecture:** Uses a powerful Bidirectional LSTM (Long Short-Term Memory) neural network.
- **Interactive Web App:** Includes a built-in Gradio interface so you can test your own sentences directly in your browser! 🌐

## 🧠 Model Architecture

The neural network is built using Keras and consists of:
1. **TextVectorization Layer:** Converts raw text into a vocabulary of up to 200,000 words.
2. **Embedding Layer:** Learns the contextual relationships between words.
3. **Bidirectional LSTM:** Captures text context going both forwards and backwards.
4. **Dense Feature Extractors:** Fully connected layers to process the sequential patterns.
5. **Output Layer:** A Dense layer with a Sigmoid activation function to output independent probabilities for all 6 toxicity classes.

## ⚡ Production Optimizations (Training vs. Serving)

This repository demonstrates a clean separation between **Model Training** and **Optimized Web Deployment**—showcasing software engineering best practices for machine learning workflows:

1. **Jupyter Notebook (`Toxicity.ipynb`):** Contains the research and training pipeline, utilizing TensorFlow and GPU acceleration to preprocess the dataset, build, train, and validate the model.
2. **Production Application (`app.py` & `prepare_vocab.py`):** Contains the deployment pipeline, optimized to run with maximum efficiency on standard CPU environments.

### Key Deployment Optimizations:
- **Instant Boot Caching:** Re-running the Keras `TextVectorization.adapt()` step on 159,000 comments during app startup was extremely slow (taking up to 10 minutes on CPU). I implemented a custom, pure-Python vocabulary generator (`prepare_vocab.py`) that executes in under 10 seconds and saves the result to a serialized file (`vocab.pkl`). The web app (`app.py`) loads this cache instantly, reducing startup latency to **less than 1 second** without needing the heavy 65 MB dataset in memory.
- **Keras 3 Compatibility Layer:** Rebuilt the legacy Keras 2 Sequential layers in code and loaded weights directly into the structure to avoid namespace errors and `TypeError` bugs when migrating older Keras models into newer TensorFlow environments.

---

## 🛠️ Setup & Installation

Follow these steps to get the project running on your local machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tomcr00ze/ToxiBERT.git
   cd ToxiBERT
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install tensorflow pandas matplotlib scikit-learn gradio jinja2
   ```

3. **Get the Dataset:**
   Ensure you have the Jigsaw dataset inside the folder `jigsaw-toxic-comment-classification-challenge/train.csv/`. *(Note: Due to file size limits, large `.csv` and `.h5` model files may not be included in the repository).*

---

## 🚀 How to Run

### Option A: Launch the Web App (Instant & Recommended)
1. Run the vocabulary pre-calculation script once (takes seconds):
   ```bash
   python prepare_vocab.py
   ```
2. Launch the Gradio web interface:
   ```bash
   python app.py
   ```
3. Open the local link (e.g. `http://127.0.0.1:7860`) in your browser to test comments! 🧪

### Option B: Run the Jupyter Notebook (Training Mode)
1. Open `Toxicity.ipynb` in Jupyter Notebook, VS Code, or Google Colab.
2. Run the cells step-by-step to load the data, train the model, and evaluate its performance.

---
*Made with ❤️ using TensorFlow and Keras.*
