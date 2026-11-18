# Adaptive Federated Aggregation for Disease Prediction using Heterogeneous Healthcare Data

A federated learning framework for binary disease classification (**Pleural Effusion**) across multi-institutional chest radiograph datasets (**NIH ChestX-ray14**, **CheXpert**, and **PadChest**).

This project investigates institutional sample volume disparity and disease prevalence skew in healthcare federations, introducing **Capped FedAvg** to mitigate client dominance while preserving global model convergence.

---

## 📌 Problem & Motivation

In cross-silo healthcare federations, participating institutions exhibit severe heterogeneity along two primary axes:
- **Sample Volume Disparity:** Major medical centers dwarf regional facilities in patient volume. Under standard Federated Averaging (FedAvg), aggregation weights are strictly proportional to sample size ($p_k \propto n_k$), allowing oversized nodes to dominate global model updates.
- **Institutional Prevalence Skew:** Disease incidence strongly reflects local clinical workflows (e.g., inpatient ICU cohorts vs. outpatient screening), inducing severe label distribution shifts across clients.

---

## ⚙️ Methodology: Capped FedAvg

**Capped FedAvg** introduces an adaptive pre-normalization threshold $c \in (0, 1]$ to bound individual client influence, followed by mass renormalization:

1. **Empirical Client Proportion:**
   $$p_k = \frac{n_k}{\sum_{j=1}^K n_j}$$
2. **Clipped Proportion:**
   $$\tilde{p}_k = \min(p_k, c)$$
3. **Renormalized Aggregation Weight:**
   $$q_k = \frac{\tilde{p}_k}{\sum_{j=1}^K \tilde{p}_j}$$
4. **Global Parameter Aggregation:**
   $$\theta_{t+1} = \sum_{k=1}^K q_k \theta_{t+1}^{(k)}$$

*When $c \ge \max_k(p_k)$, the algorithm identically recovers standard FedAvg. When $c < \max_k(p_k)$, the dominant client's update weight is clipped at $c$, and the excess mass is redistributed proportionally to minority clinical partners.*

---

## 🏗️ Model Architecture & Parity

To ensure performance differences stem strictly from aggregation dynamics rather than architectural divergence:
- **Backbone:** ResNet-50 (`torchvision.models.resnet50`, 23.51M parameters, 320 state dictionary tensors).
- **Classification Head:** Linear projection head ($\mathbf{w} \in \mathbb{R}^{2048}, b \in \mathbb{R}$) predicting a single logit.
- **Input Preprocessing:** $224 \times 224 \times 3$ RGB radiographs with standard ImageNet normalization constants ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Optimization:** AdamW optimizer with Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`).

---

## 📁 Repository Structure

```text
├── configs/               # Client and federated training configurations
├── notebooks/             # Data preprocessing, label standardization & leakage audits
├── scripts/               # Synthetic federated simulation and diagnostic runners
├── src/
│   ├── config.py          # Central project configuration and hyperparameter schemas
│   ├── data/              # Client dataset loaders, samplers, and path resolvers
│   ├── federated/         # Aggregation algorithms (FedAvg, Capped FedAvg), client & server logic
│   ├── models/            # ResNet-50 binary classification architecture
│   └── trainer.py         # Standardized local client training pipeline
└── tests/                 # Comprehensive test suite (91 pytest test cases)
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify model parity, aggregation math, and pipeline invariants:
```bash
python -m pytest tests/
```

### 3. Run Synthetic Federated Benchmark
Execute the controlled multi-round simulation comparing standard FedAvg against Capped FedAvg across cap thresholds ($c \in [0.30, 0.60]$):
```bash
python scripts/run_synthetic_demonstration.py
```

---

## 🔒 Privacy & Data Ethics

This repository adheres to strict privacy standards for decentralized clinical AI:
- **Zero Raw Data Exchange:** Medical institutions never share raw patient radiographs or protected health information (PHI). Only model weights ($\theta^{(k)}$) and sample counts ($n_k$) are communicated.
- **No Tracked Datasets:** Raw DICOM/PNG images and proprietary model checkpoints are excluded from version control.
