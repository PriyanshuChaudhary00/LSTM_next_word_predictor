# 🧠 Next Word Predictor — LSTM (PyTorch)

A next-word prediction model built **from scratch** using a custom LSTM architecture in PyTorch. Given a seed sentence, the model predicts the next N words auto-regressively.

---

## 📌 Demo

```
Enter number of token: 7
Enter Your text:  hello my name
hello my name is that mr. mccarthy with a start.
```

---

## 🏗️ Architecture

```
Input Text → Tokenizer → Embedding Layer (dim=100)
           → LSTM (hidden=150) → Linear Layer → Predicted Word
```

| Component      | Detail                         |
|----------------|-------------------------------|
| Embedding dim  | 100                           |
| LSTM hidden    | 150                           |
| Optimizer      | Adam (lr=0.001)               |
| Loss function  | CrossEntropyLoss              |
| Batch size     | 64                            |
| Epochs         | 120                           |
| Device         | Apple Silicon MPS / CPU       |

---

## 📁 Project Structure

```
├── model.py        # LSTM model definition
├── dataset.py      # Tokenizer, vocab builder, Dataset & DataLoader
├── train.py        # Training loop + loss curve
├── predict.py      # Inference script (interactive CLI)
├── data1.txt       # Training corpus
├── results/        # Training loss graphs
└── checkpoint/     # Saved model weights (.pth)
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/PriyanshuChaudhary00/LSTM_next_word_predictor.git
cd LSTM_next_word_predictor
```

### 2. Install dependencies
```bash
pip install torch numpy matplotlib
```

### 3. Train the model
```bash
python train.py
```

### 4. Run predictions
```bash
python predict.py
```

---

## 📈 Training Loss

The model was trained for 120 epochs. Loss curve saved to `results/loss_graph.png`.

---

## 🔧 What I'd Improve Next

- [ ] Stacked / Bidirectional LSTM layers
- [ ] Beam search decoding (instead of greedy argmax)
- [ ] Larger and more diverse training corpus
- [ ] Temperature sampling for creative outputs
- [ ] Streamlit or Gradio web UI

---

## 🛠️ Tech Stack

- Python 3
- PyTorch (with MPS support for Apple Silicon)
- NumPy
- Matplotlib

---

## 📄 License

MIT License
