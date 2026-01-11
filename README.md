
# Decoder-Only Transformer: From Scratch in PyTorch

> A faithful, byte-level reimplementation of the Decoder-Only Transformer architecture from *“Attention Is All You Need”* (Vaswani et al., 2017), built entirely from first principles without high-level wrapper libraries.

---

## 📖 Overview

This repository hosts a complete, from-scratch implementation of a Generative Pre-trained Transformer (GPT) style model. Unlike standard implementations that rely on `HuggingFace` or `torch.nn.Transformer` abstractions, this project explicitly defines every component of the architecture—from the attention mechanism to the training loop.

The goal was to demystify the "black box" of LLMs by engineering the internals manually, validating the architecture through rigorous training on the **TinyStories** dataset, and optimizing performance through hyperparameter tuning.

---

## ✨ Key Highlights

* **✅ Pure PyTorch Implementation:** No shortcuts. All core building blocks (Tokenization, Attention, LayerNorm) are written by hand.
* **✅ Transparent & Educational:** Includes a clean, executable Jupyter Notebook (`decoder-transformer.ipynb`) that documents the entire pipeline from data loading to inference.
* **✅ Rigorous Engineering:** Features a custom training loop with **Teacher Forcing**, gradient clipping, and learning rate scheduling (warmup + decay).
* **✅ Performance Driven:** Achieved measurable gains in generation quality through systematic hyperparameter tuning.

---

## 🛠️ Technical Architecture

I implemented the following components based on the original Transformer specifications:

### 1. Data Processing

* **Tokenizer:** Custom Byte-Pair Encoding (BPE) / Byte-level tokenizer optimized for the dataset.
* **Positional Encodings:** Implementation of both standard Sinusoidal encodings and Learnable Position Embeddings.

### 2. The Decoder Block

* **Masked Multi-Head Self-Attention:** Manually implemented `Scaled Dot-Product Attention` with a causal mask (lower triangular matrix) to prevent "peeking" at future tokens.
* **Feed-Forward Networks:** Position-wise MLP with GELU/ReLU activation.
* **Residual Connections:** Applied strictly as  (Pre-LN variant) for training stability.

### 3. Training Pipeline

* **Optimization:** AdamW optimizer with weight decay.
* **Scheduling:** Custom scheduler implementing linear warmup followed by cosine decay.
* **Evaluation:** Automated pipeline computing **BLEU scores** to objectively measure generation quality.

---

## 🚀 Hyperparameter Tuning & Results

To push the model beyond baseline performance, I conducted a series of experiments tuning the **Learning Rate**, **Dropout**, and **Batch Size**.

### The Experiment

* **Baseline Model:** 120M Parameters, Standard configuration.
* **Tuned Model:** "Aggressive" configuration (Higher LR: `5e-4`, Higher Dropout: `0.2`).
* **Metric:** BLEU Score (evaluated on unseen validation data).

### 🏆 Impact

> Hyperparameter tuning resulted in a **19.9% relative improvement** in the BLEU score compared to the initial baseline configuration 

