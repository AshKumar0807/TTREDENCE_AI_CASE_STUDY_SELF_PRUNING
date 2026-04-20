# Self-Pruning Neural Network — Tredence AI Engineering Case Study

This repository implements a **Soft-Pruning Neural Network** for CIFAR-10 classification.
Instead of pruning after training, the model **learns to prune itself during training** using learnable sigmoid gates.

---

## Quick Start

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Train and evaluate across multiple lambda values
python train.py
```

All outputs (plots, JSON results, model checkpoints) are saved in:

```
./outputs/
```

---

## Architecture Highlights

### PrunableLinear

A drop-in replacement for `nn.Linear`.

Each weight has a learnable gate:

```
W_eff = W ⊙ sigmoid(G)
```

---

### Sparsity Loss

Total loss:

```
Loss = CrossEntropy + λ * Σ sigmoid(G)
```

* Higher λ → more sparsity
* Gates pushed toward 0 → weights effectively removed

---

### Training Dynamics

* Small λ → dense network
* Large λ → sparse network
* Model learns which weights are important

---

## Results Summary

| λ (Sparsity Coeff) | Accuracy | Sparsity |
| ------------------ | -------- | -------- |
| 1e-05              | 63.44%   | 64.86%   |
| 0.0001             | 63.59%   | 92.08%   |
| 0.0005             | 63.41%   | 99.08%   |

👉 Even with **99% pruning**, accuracy remains ~63%
👉 Shows massive parameter redundancy

---

## Repository Structure

| File      | Description                     |
| --------- | ------------------------------- |
| train.py  | Model + training + evaluation   |
| REPORT.md | Detailed theory & analysis      |
| outputs/  | Graphs, histograms, checkpoints |

---

## Key Insight

> A neural network can lose **99% of its weights** and still perform competitively.

---
