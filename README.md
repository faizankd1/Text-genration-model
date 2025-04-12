# Text-genration-model
This text generation model is trained on the Tiny Shakespeare dataset, a compact corpus of Shakespeare's plays and dialogues consisting of approximately 1 million characters. The model learns the stylistic and linguistic patterns of Elizabethan English, enabling it to generate text that mimics Shakespearean dialogue and structure.
# 🎭 Tiny Shakespeare Text Generator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

A lightweight text generation model trained on the Tiny Shakespeare dataset (~1MB of Shakespeare's plays). It generates poetic, dramatic, and stylized text in the flavor of the Bard himself — perfect for creative projects, AI writing experiments, or just having fun with Renaissance English.

---

## ✨ Features

- 🧠 Character-level text generation
- 🕰️ Learned from real Shakespearean dialogue
- 💬 Generate scenes, monologues, or quotes
- ⚡ Fast training and inference on low-end machines

---

## 📂 Dataset

We use the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), which contains:

- Dialogue from various plays
- Stage directions and scene formatting
- Authentic Shakespearean language (~1 million characters)

---

## 🏗️ Model Details

| Parameter       | Value           |
|----------------|------------------|
| Model Type      | LSTM / GRU / Transformer *(edit as needed)* |
| Training Type   | Character-level |
| Input Length    | 100 characters |
| Framework       | PyTorch *(or TensorFlow if applicable)* |
| Output          | Shakespearean-style text |

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/faizankd1/tiny-shakespeare-generator.git
cd tiny-shakespeare-generator
