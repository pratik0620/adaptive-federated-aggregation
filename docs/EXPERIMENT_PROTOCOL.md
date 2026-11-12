# Federated Learning Experiment Protocol: FedAvg vs. Capped FedAvg

**Project Title:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
**Prediction Task:** Binary Pleural Effusion Classification on Chest Radiographs  
**Institutions / Silos:**
1. **Client 01:** NIH ChestX-ray14 (`client_01_NIH`)
2. **Client 02:** CheXpert — Stanford Hospital (`client_02_CheXpert`)
3. **Client 03:** PadChest — BIMCV / Hospital San Juan (`client_03_PadChest`)

---

## 1. Research Question & Empirical Hypothesis

### 1.1 Context & Problem Formulation
In real-world cross-silo healthcare federations, participating hospitals exhibit extreme heterogeneity in:
- **Sample Scale Disparity:** Large academic medical centers (e.g. CheXpert with >90,000 frontal radiographs) dwarf regional hospitals or specialized archives by orders of magnitude.
- **Label Skew (Prevalence Shift):** Disease prevalence varies dramatically by clinical referral setting. In CheXpert (inpatient ICU bias), Pleural Effusion prevalence is **73.7%**, whereas in NIH ChestX-ray14 (outpatient screening bias) it is **14.7%**, and in PadChest it is **16.4%**.

Under standard Federated Averaging (FedAvg), aggregation weights are proportional to sample count ($p_k = n_k / N$). Consequently, a high-volume client can command 70–85% of the total aggregation weight, pulling the shared model's posterior calibration and gradient trajectory overwhelmingly toward its local institutional prevalence. This can lead to elevated false-positive rates on lower-prevalence partner nodes.

### 1.2 Research Question
Does capping the maximum allowable aggregation weight of any single clinical institution (**Capped FedAvg**) mitigate institutional dominance and improve multi-institutional validation performance and fairness across heterogeneous healthcare nodes compared to standard **FedAvg**?

### 1.3 Hypothesis (Empirical Stance)
We hypothesize that applying a pre-normalization clipping threshold $c \in (0, 1.0]$ to raw sample-fraction weights prior to renormalization:
1. Attenuates the dominance of any single high-volume client over global model convergence.
2. Improves discrimination (AUROC, PR-AUC) and balanced accuracy on partner hospitals with lower prevalence.
3. Preserves overall global diagnostic performance without destabilizing convergence.

> [!NOTE]
> **Scientific Integrity Notice:**
> The efficacy of Capped FedAvg is an **open empirical question**, not an established fact. A formal literature review of adaptive federated weighting schemes (e.g. FedProx, q-FFL, Agnostic Federated Learning, and contribution-fair aggregation) must be conducted as part of the research paper to position this heuristic within the broader field.

---

## 2. Mathematical Formulations

Let $K = 3$ be the number of participating clients. At communication round $t$, each client $k$ trains locally on private data and reports sample count $n_k$ and updated model parameters $\theta_{t+1}^{(k)}$. Total sample count is $N = \sum_{k=1}^K n_k$.

### 2.1 Baseline: Standard FedAvg
In standard FedAvg (McMahan et al., 2017), client weights are the exact unconstrained empirical sample fractions:

$$p_k^{\text{FedAvg}} = \frac{n_k}{N}, \quad \text{where } \sum_{k=1}^K p_k^{\text{FedAvg}} = 1.0$$

The central server updates the global parameters via:

$$\theta_{t+1} = \sum_{k=1}^K p_k^{\text{FedAvg}} \theta_{t+1}^{(k)}$$

### 2.2 Proposed Modification: Capped FedAvg
Let $c \in (0, 1.0]$ denote the pre-normalization clipping threshold applied to each client's empirical sample fraction before renormalization.

#### Pre-Normalization Clipping Algorithm:
1. Compute raw sample fractions:
   $$p_k = \frac{n_k}{\sum_{j=1}^K n_j}$$
2. Clip each raw weight at the threshold $c$:
   $$\tilde{p}_k = \min(p_k, c)$$
3. Renormalize clipped weights so they sum to $1.0$:
   $$q_k = \frac{\tilde{p}_k}{\sum_{j=1}^K \tilde{p}_j}$$
4. Aggregate client parameters:
   $$\theta_{t+1}^{\text{Capped}} = \sum_{k=1}^K q_k \theta_{t+1}^{(k)}$$

> [!NOTE]
> **Pre-Normalization Clipping Semantics:** Because renormalization ($q_k = \tilde{p}_k / \sum_j \tilde{p}_j$) divides by $\sum_j \tilde{p}_j \le 1.0$, the final normalized aggregation weight $q_k$ may exceed $c$. This is the intended pre-normalization clipping formulation (distinct from iterative water-filling or strict post-normalization capping). If $c \ge \max_k p_k$ (including $c = 1.0$), Capped FedAvg degenerates to standard FedAvg; if $c \le \min_k p_k$, all clients are clipped equally and $q_k = 1/K$.

---

## 3. Cap Selection Protocol: Validation-Only Tuning

To prevent statistical overfitting and data snooping:

### 3.1 Candidate Cap Grid
For a 3-client federation ($K = 3$):
$$\mathcal{C} = \{0.40, 0.50, 0.60, 0.70\}$$

### 3.2 Selection Criterion
The optimal cap value $C^*$ will be selected exclusively using **client validation sets** according to:

$$C^* = \arg\max_{C \in \mathcal{C}} \text{Score}_{\text{val}}(C)$$

Where candidate selection scores are:
1. **Primary: Macro-Average Validation AUROC:**
   $$\text{Macro-Val-AUROC} = \frac{1}{K} \sum_{k=1}^K \text{AUROC}_{\text{val}}^{(k)}$$
2. **Secondary: Worst-Client Validation AUROC (Fairness Gate):**
   $$\text{Min-Val-AUROC} = \min_{k \in \{1, \dots, K\}} \text{AUROC}_{\text{val}}^{(k)}$$

### 3.3 Strict Test Set Isolation Rule
> [!IMPORTANT]
> **Cardinal Research Rule:**
> Held-out test sets (`test.csv`) must **NEVER** be evaluated during communication rounds, hyperparameter searches, threshold selection, or cap tuning. The test split is evaluated exactly **once** at the final designated evaluation stage on the single best checkpoint selected by validation performance.

---

## 4. Controlled Experimental Parity

To ensure that performance differences arise solely from the aggregation algorithm:

| Dimension | Standard FedAvg Baseline | Capped FedAvg | Parity Guarantee |
| :--- | :---: | :---: | :--- |
| **Initial Global Model** | ImageNet ResNet-50 (`seed`) | ImageNet ResNet-50 (`seed`) | Identical initial state dictionary distributed to all nodes. |
| **Model Architecture** | `ResNet50Binary` | `ResNet50Binary` | Single-logit head (`Linear(2048, 1)`), `dropout = 0.0`. |
| **Input Preprocessing** | $224 \times 224$ RGB, ImageNet norm | $224 \times 224$ RGB, ImageNet norm | Identical channel replication and normalization constants. |
| **Local Optimizer** | AdamW (`lr=1e-4, wd=1e-4`) | AdamW (`lr=1e-4, wd=1e-4`) | Identical optimizer type and hyperparameters. |
| **Optimizer Reset** | Fresh optimizer each round | Fresh optimizer each round | Zero momentum drift across communication rounds. |
| **Local Epochs ($E$)** | $E = 1$ (or $E = 2$) | $E = 1$ (or $E = 2$) | Identical local computation per round. |
| **Batch Size ($B$)** | $B = 32$ | $B = 32$ | Identical mini-batch geometry. |
| **Communication Rounds ($T$)** | $T = 5$ (or $T = 10$) | $T = 5$ (or $T = 10$) | Synchronous round termination at identical $T$. |
| **Loss Function** | `BCEWithLogitsLoss()` | `BCEWithLogitsLoss()` | Unweighted binary cross-entropy. |
| **Random Seeds** | 3 seeded runs (42, 123, 456) | 3 seeded runs (42, 123, 456) | Pairwise seed-matched comparisons. |

---

## 5. Client Cohorts and Patient-Level Isolation

All three client datasets use patient-level partitions to guarantee zero leakage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Institutional Silos                             │
├──────────────────────┬──────────────────────┬───────────────────────────────┤
│ Client 01: NIH       │ Client 02: CheXpert  │ Client 03: PadChest           │
│ 53,323 Train images  │ 92,164 Frontal Train │ 34,367 Frontal Train images   │
│ 20,977 unique pts    │ 36,726 unique pts    │ 26,534 unique pts             │
│ Prevalence: 14.7%    │ Prevalence: 73.7%    │ Prevalence: 16.4%             │
└──────────────────────┴──────────────────────┴───────────────────────────────┘
```

- **Patient-Level Isolation:** Verified by set intersection ($\text{train\_patients} \cap \text{val\_patients} = \emptyset$).
- **Frontal View Standardization:** AP and PA projections only (CheXpert and PadChest lateral/costal views excluded).
- **Pilot Experiments:** When `--patient_limit` is set, sampling slices unique patients, never raw images.

---

## 6. Evaluation Metrics & Protocol

### 6.1 Metrics Suite
All binary classification metrics are computed via `src.metrics.compute_binary_metrics`:
- **Primary Discriminative Metric:** **AUROC** (Area Under Receiver Operating Characteristic).
- **Secondary Precision-Recall Metric:** **PR-AUC** (Average Precision; critical for low-prevalence NIH & PadChest).
- **Clinical Operating Point Metrics ($\tau = 0.50$):**
  - Sensitivity (Recall, True Positive Rate).
  - Specificity (True Negative Rate).
  - F1 Score (Harmonic mean of Precision and Sensitivity).
  - Balanced Accuracy ($\frac{\text{Sensitivity} + \text{Specificity}}{2}$).
  - Confusion Matrix ($\text{TP}, \text{FP}, \text{TN}, \text{FN}$).

### 6.2 Reporting Granularity
For every communication round:
1. **Per-Client Validation Metrics:** Logged independently for Client 01, Client 02, and Client 03.
2. **Cohort Summary Metrics:** Macro-average across institutions and sample-weighted average.
3. **Tracking Artifacts:** Saved to `round_metrics.json` and tabular `history.csv`.

---

## 7. Controlled Synthetic Label-Skew Experiment

To validate the theoretical behavior of Capped FedAvg before launching multi-hour medical image training, a controlled synthetic simulation is specified:

- **Setup:** 3 synthetic clients with artificially constructed size and label skew:
  - **Client A (Dominant / High Prevalence):** $n_A = 800$ samples, Prevalence = $85\%$.
  - **Client B (Minority / Low Prevalence):** $n_B = 100$ samples, Prevalence = $15\%$.
  - **Client C (Minority / Low Prevalence):** $n_C = 100$ samples, Prevalence = $15\%$.
- **Weights under FedAvg:** $p_A = 0.80, p_B = 0.10, p_C = 0.10$. (Client A dominates by $8\times$).
- **Weights under Capped FedAvg ($c = 0.50$):** Clipped weights are $\tilde{p}_A = 0.50, \tilde{p}_B = 0.10, \tilde{p}_C = 0.10$ ($\sum_j \tilde{p}_j = 0.70$), giving normalized weights $q_A = 0.50 / 0.70 \approx 0.7143$, $q_B = 0.10 / 0.70 \approx 0.1429$, $q_C = 0.10 / 0.70 \approx 0.1429$. Note that $q_A > c$ under pre-normalization clipping, while Client A's dominance is reduced relative to FedAvg.
- **Expected Empirical Observation:**
  Under FedAvg, the global model develops a strong positive prediction bias, reducing specificity on Clients B and C. Under Capped FedAvg, balanced accuracy on Clients B and C is preserved.
- **Reporting Label:** All synthetic runs are marked `"is_synthetic": true` and are reported in a separate synthetic verification appendix.

---

## 8. Statistical Reporting, Variability & Limitations

1. **Repeated Seed Runs:** Report mean and sample standard deviation over 3 random seeds ($\mu \pm \sigma$).
2. **Partial Archive Disclaimer:** If evaluated on partial local image subsets (e.g. NIH `images_001`), metrics must carry the explicit header:
   `[UNOFFICIAL: Partial archive subset (X / Y images)]`
   and must never be presented as official full-benchmark results.
3. **Image-Level vs. Patient-Level Metrics:** Note that while evaluation operates on individual radiographs, diagnostic decisions in clinical practice are patient-level.
