# Transformer

Implementation of a Decoder-only Transformer (the architecture behind ChatGPT) built entirely in PyTorch.

This project is designed as a learning tool. Instead of using massive pre-trained models, we build the neural network block-by-block to understand exactly how information flows through a Transformer.

## 🎯 Project Goal

This model reads a Markdown file (e.g., `data.md`) containing company information or Q&A pairs, trains on that text, and generates answers to questions based *only* on what it learned.

## 🧠 Architecture Mapping

If you look at a standard Transformer diagram, here is how the code maps to the blocks:

- **Input Embedding (Pink Box):** `GPTEmbedding` class. Converts characters into numbers.

- **Positional Encoding (+ Circle):** `GPTEmbedding` class. Adds position information to the embeddings.

- **Masked Multi-Head Attention (Orange Box):** `MultiHeadAttention` class. Allows words to look at previous words to understand context (masking prevents it from "cheating" and looking at future words).

- **Feed Forward (Blue Box):** `FeedForward` class. Processes each word independently to learn complex patterns.

- **Add & Norm (Yellow Boxes):** `Block` class. Residual connections (`x + ...`) and Layer Normalization.

- **Nx (Repeating Blocks):** `MiniGPT` class. The Transformer block is repeated `n_layer` times.

- **Linear & Softmax (Top Boxes):** `MiniGPT.lm_head` and `F.softmax`. Projects the internal vectors back to our vocabulary size to predict the next character.

## 🛠️ Prerequisites

You need Python 3.8 or higher. You will also need to install PyTorch.

bash

pip install torch

## 🚀 How to Run

### 1. Prepare your Data

Create a file named `data.md` in the same folder. Add your training text (Q&A pairs, company policies, etc.):

markdown

Q: What is the company policy on remote work?
A: Employees can work remotely up to 3 days a week.
Q: How do I request time off?
A: Submit a request in the HR portal two weeks in advance.

### 2. Configure the Model

Open the Python script and adjust the hyperparameters at the bottom of the file:

python

block_size = 32 # How many characters the model looks at at once
n_embd = 64     # How big the internal vectors are
n_head = 4      # Number of attention heads
n_layer = 4     # Number of Transformer blocks

### 3. Train the Model

Run the script. The model will start studying the `data.md` file.

bash

python train.py

*You will see the `Loss` score decrease as the model learns the patterns in your text.*

### 4. Generate a Response

After training, the script will automatically run the `generate` function and output a response to a prompt.

## ⚠️ Important Limitations (The Reality Check)

This is a **character-level** model trained on a tiny dataset.

- **It is an "Autocomplete", not a genius:** It learns to predict the next character. If it says something smart, it's because it memorized it from the Markdown file.

- **It will hallucinate:** If you ask a question that isn't in the text, it will generate random letters that look like English but mean nothing.

- **Real-world use:** To build a production-ready Q&A bot, you should use **RAG (Retrieval-Augmented Generation)** or fine-tune an existing model (like Llama 3 or GPT-4) rather than training a tiny Transformer from scratch.

## 📚 What's Next?

Want to improve this model?

1. **Word-level Tokenizer:** Instead of predicting characters, build a tokenizer that predicts whole words.

2. **More Data:** Feed it megabytes of text instead of a few kilobytes.

3. **Scale Up:** Increase `n_embd` to 256, `n_head` to 8, and `n_layer` to 6.

## 🤝 Contributing

This repo is for learning. Feel free to fork it, add comments, break the code, and fix it. The best way to learn Transformers is to break them and see why they fail!
