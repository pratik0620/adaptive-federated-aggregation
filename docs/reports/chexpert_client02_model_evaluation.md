# CheXpert Client 02: Model Interpretation & Capstone Evaluation Report

**Project Title:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
**Assigned Client Node:** CheXpert Client 02 — Stanford Hospital (Stanford, CA, USA)  
**Target Disease:** Pleural Effusion (Binary Classification)  
**Evaluated Run Artifact:** `outputs/chexpert/chexpert_resnet50_20261027_225232`  
**Best Checkpoint:** `outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt` (Epoch 3)  
**Report Date:** October 27, 2026  
**Author:** CheXpert Client 02 Team Member  

---

## 1. Executive Summary

This report documents the local training, validation audit, decision threshold analysis, and benchmark test evaluation for **CheXpert Client 02**, a participating node in the adaptive federated learning healthcare network. The assigned clinical task is the binary classification of **Pleural Effusion** from chest radiographs using an ImageNet-pretrained ResNet-50 backbone.

Following the initial 100-patient pilot, a 500-patient, 5-epoch training run was completed on an NVIDIA GeForce RTX 4050 Laptop GPU (Run ID: `chexpert_resnet50_20261027_225232`). The primary model selection metric was validation Area Under the Receiver Operating Characteristic curve (AUROC), which peaked at **Epoch 3** with a validation AUROC of **0.89735** and a Precision-Recall AUC (PR-AUC) of **0.95191** across the complete, leak-free validation split of **10,024 frontal radiographs**.

At the standard decision threshold of $0.50$, the model achieves **92.71% sensitivity** and **68.21% specificity** on validation data (Balanced Accuracy: **80.46%**). An exploratory validation-only threshold sweep across 13 candidate points ($0.46 \le \tau \le 0.70$) demonstrated that shifting the threshold to **$\tau = 0.70$** yields the highest validation balanced accuracy (**83.07%**) and Youden's J index (**0.6615**), increasing specificity to **78.36%** (+10.15 percentage points) while maintaining **87.79% sensitivity**. 

On the held-out Stanford benchmark test split ($N = 202$ frontal images, 31.68% positive prevalence), the model evaluated at the preset baseline threshold of $0.50$ achieved an AUROC of **0.83967**, PR-AUC of **0.67338**, sensitivity of **93.75%**, and specificity of **55.80%**. The observed drop in precision from 89.66% on validation to 49.59% on test is directly explained by Bayes' theorem under the severe base-rate shift from 74.83% validation prevalence to 31.68% test prevalence.

> [!IMPORTANT]
> **Scientific Scope & Non-Clinical Disclaimer:** These findings represent empirical, single-client local benchmark results within an academic capstone simulation. They do not constitute proof of clinical safety, diagnostic efficacy, generalizability across external health systems, or federated aggregation superiority. The candidate threshold ($\tau = 0.70$) was selected strictly on validation data and has not been tuned or evaluated against the test set.

---

## 2. Client Role and Binary Prediction Task

### 2.1 Clinical Background
Pleural effusion—the pathological accumulation of fluid in the pleural space between the parietal and visceral pleura—is a frequent manifestation of cardiopulmonary disease, malignancy, pneumonia (parapneumonic effusion/empyema), and volume overload. On chest radiography, physical signs depend heavily on patient position and beam orientation:
- **Upright PA radiographs:** Fluid collects in dependent sulci, producing blunting of the costophrenic angles and a curved lateral meniscus.
- **Supine/portable AP radiographs:** Fluid layers posteriorly, presenting as diffuse hemithorax haziness, apical capping, or vascular obscuration without a distinct meniscus.

### 2.2 Client Node Role in Federation
In the federated architecture, three independent hospital centers collaborate to train a disease predictor without pooling raw patient records:
1. `client_01_NIH`: NIH Clinical Center (predominantly ambulatory/outpatient adult and pediatric scans, PA-dominant).
2. `client_02_CheXpert`: Stanford Hospital (inpatient, emergency, and intensive care adult radiographs, AP-dominant).
3. `client_03_PadChest`: Hospital Universitario de San Juan de Alicante (European tertiary hospital with mixed adult/pediatric scans).

CheXpert Client 02 represents a tertiary inpatient referral center characterized by high patient acuity, older demographics, and bedbound patients requiring portable anteroposterior (AP) imaging.

### 2.3 Mathematical Task Formulation
The prediction task is framed as single-output binary classification:
$$\hat{y} = \mathbb{I}(\sigma(z) \ge \tau)$$
where $z \in \mathbb{R}$ is the unconstrained scalar logit emitted by the classification head, $\sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]$ represents the modeled posterior probability $P(\text{Effusion} = 1 \mid X)$, and $\tau \in (0, 1)$ is the decision threshold. The loss minimized during local training is binary cross-entropy with logits:
$$\mathcal{L}(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$$

---

## 3. Dataset Cohort and Split Information

### 3.1 Raw Dataset Characteristics
The CheXpert dataset (Stanford Hospital) comprises 121,817 chest radiographs across 41,009 unique patients. Under the project's standardized preprocessing protocol ([`docs/three_client_heterogeneity_analysis.md`](file:///D:/DS%20CP/adaptive-federated-aggregation/docs/three_client_heterogeneity_analysis.md)):
- Lateral radiographs (19,417 images) are excluded, restricting the modeling space to frontal views (AP and PA).
- Label consolidation maps explicit positive mentions to $y=1$, explicit negative mentions to $y=0$, and excludes uncertain (`-1`) or unmentioned blank entries.

### 3.2 Cohort Partitioning and Patient Isolation
The repository strictly enforces patient-level partitioning to eliminate data leakage across splits:

| Split Partition | Source Definition | Image Count | Unique Patients | Unique Studies | Positive Count | Negative Count | Prevalence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Eligible Training Pool** | Processed `train.csv` (Frontal AP/PA) | 92,164 | 36,726 | 76,513 | 69,393 | 22,771 | 75.29% |
| **500-Patient Training Cohort** | Seeded Stratified Sample (`seed=42`) | **1,241** | **500** | **1,038** | **915** | **326** | **73.73%** |
| **Official Validation Split** | Processed `val.csv` (Frontal AP/PA) | **10,024** | **4,081** | **8,222** | **7,501** | **2,523** | **74.83%** |
| **Held-Out Test Benchmark** | Stanford Radiologist Consensus (`valid.csv`) | **202** | **200** | **200** | **64** | **138** | **31.68%** |

#### Patient Isolation Audit
- **Training Cohort:** 500 unique patient IDs (`patient05738` through `patient64503`), as recorded in `patient_manifest.json`.
- **Validation Cohort:** 4,081 unique patient IDs.
- **Intersection:** $| \text{Train Patients} \cap \text{Val Patients} | = \mathbf{0}$. Patient leakage is formally zero.
- **Disk Accessibility:** 10,024 of 10,024 validation radiographs (100.0%) were indexed and verified on local disk without missing files.

---

## 4. Model Architecture and Training Configuration

### 4.1 Architecture Specification
The model uses a torchvision ResNet-50 backbone modified for single-logit binary classification:
- **Feature Extractor:** 50-layer convolutional residual network initialized with torchvision ImageNet-1k default weights (`ResNet50_Weights.DEFAULT`).
- **Global Pooling:** Adaptive average pooling reducing 4D spatial feature maps to a 2048-dimensional embedding vector.
- **Classification Head:** Linear projection $W \in \mathbb{R}^{1 \times 2048}, b \in \mathbb{R}$ mapped directly to a single scalar logit:
  $$z = W \cdot \text{GAP}(\mathbf{F}) + b$$
- **Dropout:** Rate = 0.0 (deterministic inference).
- **Total Parameters:** 23,510,081 (all 320 parameter tensors strictly tracked in `state_dict`).

### 4.2 Training Hyperparameters

| Hyperparameter | Configuration Value | Justification / Context |
| :--- | :---: | :--- |
| **Optimization Algorithm** | AdamW | Decoupled weight decay regularization for adaptive optimizers. |
| **Initial Learning Rate ($\eta_0$)** | $1.0 \times 10^{-4}$ | Standard fine-tuning rate for pretrained ResNet-50 backbones. |
| **Weight Decay ($\lambda$)** | $1.0 \times 10^{-4}$ | $L_2$ penalty preventing convolutional filter co-adaptation. |
| **Learning Rate Schedule** | Cosine Annealing | Monotonic decay from $1.0 \times 10^{-4}$ to $9.55 \times 10^{-6}$ across 5 epochs. |
| **Batch Size** | 16 | Hardware-constrained for 6 GB laptop GPU VRAM; gradient accumulation = 1. |
| **Input Resolution** | $224 \times 224$ px | Standard ImageNet spatial footprint; bilinear interpolation. |
| **Image Normalization** | ImageNet Mean & Std | $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$. |
| **Data Augmentation** | Deterministic Val | Training applied deterministic transforms (no random crops/rotations). |
| **Mixed Precision (AMP)** | Enabled (`fp16`) | PyTorch `torch.amp.autocast('cuda')` utilizing RTX 4050 Tensor Cores. |
| **Loss Formulation** | Unweighted BCE | Standard BCE with logits (`pos_weight=None`). |
| **Execution Device** | NVIDIA RTX 4050 Laptop GPU | 6 GB VRAM; PyTorch 2.13.0+cu130. |
| **Total Training Time** | 364.73 seconds (~6.1 min) | 1,241 train + 10,024 val images evaluated per epoch. |

---

## 5. Training History and Model Selection

The training run proceeded for 5 complete epochs. Validation evaluation was performed at the conclusion of every epoch on the entire 10,024-image validation cohort:

| Epoch | Train Loss | Train AUROC | Val Loss | Val AUROC | Val PR-AUC | Val Sensitivity | Val Specificity | Val Balanced Acc | Learning Rate | Epoch Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.4593 | 0.8108 | 0.3751 | 0.8831 | 0.9492 | 0.9227 | 0.6258 | 0.7743 | $1.00 \times 10^{-4}$ | 134.2s |
| 2 | 0.2840 | 0.9324 | 0.3882 | 0.8956 | 0.9533 | 0.8415 | 0.8086 | 0.8250 | $9.05 \times 10^{-5}$ | 56.8s |
| **3\*** | **0.1899** | **0.9719** | **0.3459** | **0.89735** | **0.95191** | **0.92708** | **0.68212** | **0.80460** | **$6.55 \times 10^{-5}$** | **57.0s** |
| 4 | 0.1196 | 0.9897 | 0.3643 | 0.8843 | 0.9435 | 0.9332 | 0.6540 | 0.7936 | $3.45 \times 10^{-5}$ | 57.8s |
| 5 | 0.0946 | 0.9940 | 0.3637 | 0.8923 | 0.9475 | 0.9181 | 0.7015 | 0.8098 | $9.55 \times 10^{-6}$ | 58.0s |

*\*Epoch 3 was identified as the optimal checkpoint based on the primary model selection criterion (`val_auroc = 0.8973476016`) and lowest validation loss (`0.345894`). This checkpoint was serialized to `best_checkpoint.pt`.*

### Convergence Analysis
- **Training Progression:** Training loss decreased smoothly from 0.4593 to 0.0946, while training AUROC advanced from 0.8108 to 0.9940.
- **Generalization Gap:** Validation AUROC plateaued between 0.8831 and 0.8973. Beyond Epoch 3, validation loss began to creep upward from 0.3459 to 0.3637 despite continued drops in training loss, signaling the onset of local overfitting on the 500-patient training set. Checkpoint selection at Epoch 3 properly halted training before over-specialization.

---

## 6. Validation Performance at Standard Threshold ($\tau = 0.50$)

Recomputed metrics on the full 10,024 validation images strictly reproduced the saved baseline (`val_metrics.json`):

| Evaluation Metric | Measured Value | Standard Formulation | Clinical / Statistical Interpretation |
| :--- | :---: | :---: | :--- |
| **AUROC** | **0.89735** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) dt$ | Excellent discriminative ranking across arbitrary thresholds. |
| **PR-AUC** | **0.95191** | $\int_0^1 \text{Precision}(r) dr$ | High area under precision-recall curve, supported by high base prevalence (74.83%). |
| **Sensitivity (Recall)** | **0.92708** | $\frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{6954}{7501}$ | Identifies 92.7% of patients with radiologically confirmed effusion. |
| **Specificity** | **0.68212** | $\frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{1721}{2523}$ | Identifies 68.2% of patients without effusion; generates 802 false alarms. |
| **Precision (PPV)** | **0.89660** | $\frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{6954}{7756}$ | 89.7% of positive flags represent true effusion cases under 74.83% prevalence. |
| **Negative Predictive Value (NPV)** | **0.75882** | $\frac{\text{TN}}{\text{TN} + \text{FN}} = \frac{1721}{2268}$ | 75.9% of negative flags represent true negative cases. |
| **F1-Score** | **0.91158** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic balance favoring positive class recall. |
| **Balanced Accuracy** | **0.80460** | $\frac{\text{Sensitivity} + \text{Specificity}}{2}$ | Unweighted mean of class accuracies, compensating for class imbalance. |
| **Log-Loss** | **0.34589** | $-\frac{1}{N} \sum y \log p + (1-y) \log(1-p)$ | Mean cross-entropy penalizing probabilistic miscalibration. |

### Score Distribution & Separation
- **Negative Class ($y=0$, $N=2,523$):** Median predicted probability = **0.2133** (IQR: $[0.0322, 0.6361]$).
- **Positive Class ($y=1$, $N=7,501$):** Median predicted probability = **0.9623** (IQR: $[0.8796, 0.9873]$).
- **Separation Delta:** $\Delta \text{Median} = \mathbf{+0.7490}$. The model demonstrates robust separation, with the upper quartile of negatives ($0.6361$) well separated from the lower quartile of positives ($0.8796$).

---

## 7. Validation-Only Threshold Sweep Analysis

To evaluate operating trade-offs without touching test data, 13 candidate thresholds from $0.46$ to $0.70$ in increments of $0.02$ were evaluated across all 10,024 validation images:

```
Threshold Optimization Trajectory (Validation Split, N=10,024):
  Threshold: 0.46 ──► 0.50 ──► 0.54 ──► 0.58 ──► 0.62 ──► 0.66 ──► 0.70
  Sens (%):  93.36   92.71   92.00   91.07   90.16   89.07   87.79
  Spec (%):  66.35   68.21   70.23   72.77   74.32   76.34   78.36
  Bal Acc:   79.86   80.46   81.12   81.92   82.24   82.70   83.07 (Peak)
  Youden J:  0.5971  0.6092  0.6223  0.6384  0.6448  0.6541  0.6615 (Peak)
```

### Detailed Threshold Operating Points

| Threshold ($\tau$) | Sensitivity | Specificity | Balanced Acc | Youden's J | Precision | F1-Score | True Neg (TN) | False Pos (FP) | False Neg (FN) | True Pos (TP) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.46 | 0.9336 | 0.6635 | 0.7986 | 0.5971 | 0.8919 | 0.9123 | 1,674 | 849 | 498 | 7,003 |
| 0.48 | 0.9299 | 0.6734 | 0.8016 | 0.6033 | 0.8943 | 0.9118 | 1,699 | 824 | 526 | 6,975 |
| **0.50** | **0.9271** | **0.6821** | **0.8046** | **0.6092** | **0.8966** | **0.9116** | **1,721** | **802** | **547** | **6,954** |
| 0.52 | 0.9239 | 0.6924 | 0.8082 | 0.6163 | 0.8993 | 0.9114 | 1,747 | 776 | 571 | 6,930 |
| 0.54 | 0.9200 | 0.7023 | 0.8112 | 0.6223 | 0.9019 | 0.9108 | 1,772 | 751 | 600 | 6,901 |
| 0.56 | 0.9148 | 0.7134 | 0.8141 | 0.6282 | 0.9047 | 0.9097 | 1,800 | 723 | 639 | 6,862 |
| 0.58 | 0.9107 | 0.7277 | 0.8192 | 0.6384 | 0.9086 | 0.9096 | 1,836 | 687 | 670 | 6,831 |
| 0.60 | 0.9056 | 0.7360 | 0.8208 | 0.6416 | 0.9107 | 0.9082 | 1,857 | 666 | 708 | 6,793 |
| 0.62 | 0.9016 | 0.7432 | 0.8224 | 0.6448 | 0.9126 | 0.9071 | 1,875 | 648 | 738 | 6,763 |
| 0.64 | 0.8960 | 0.7523 | 0.8241 | 0.6483 | 0.9149 | 0.9054 | 1,898 | 625 | 780 | 6,721 |
| 0.66 | 0.8907 | 0.7634 | 0.8270 | 0.6541 | 0.9180 | 0.9041 | 1,926 | 597 | 820 | 6,681 |
| 0.68 | 0.8851 | 0.7733 | 0.8292 | 0.6584 | 0.9207 | 0.9025 | 1,951 | 572 | 862 | 6,639 |
| **0.70\*** | **0.8779** | **0.7836** | **0.8307** | **0.6615** | **0.9234** | **0.9001** | **1,977** | **546** | **916** | **6,585** |

*\*Optimal operating point identified on validation data for both Balanced Accuracy ($0.8307$) and Youden's J index ($0.6615$).*

### Comparison: Default $\tau = 0.50$ vs. Candidate $\tau = 0.70$

| Dimension | Default Threshold ($\tau = 0.50$) | Candidate Threshold ($\tau = 0.70$) | Net Absolute Delta | Relative Shift |
| :--- | :---: | :---: | :---: | :---: |
| **Sensitivity** | 92.71% | 87.79% | $-4.92\text{ pp}$ | $-5.31\%$ |
| **Specificity** | 68.21% | 78.36% | $\mathbf{+10.15\text{ pp}}$ | $\mathbf{+14.88\%}$ |
| **Balanced Accuracy** | 80.46% | 83.07% | $+2.61\text{ pp}$ | $+3.24\%$ |
| **Youden's J Index** | 0.6092 | 0.6615 | $+0.0523$ | $+8.58\%$ |
| **Precision (PPV)** | 89.66% | 92.34% | $+2.68\text{ pp}$ | $+2.99\%$ |
| **False Positives (FP)** | 802 | 546 | $\mathbf{-256}$ | $\mathbf{-31.92\%}$ |
| **False Negatives (FN)** | 547 | 916 | $+369$ | $+67.46\%$ |

---

## 8. Confusion-Matrix Interpretation & Clinical Trade-Offs

### 8.1 Empirical Confusion Matrices

```
Confusion Matrix at Threshold 0.50:            Confusion Matrix at Threshold 0.70:
                Predicted                                      Predicted
            Negative   Positive                            Negative   Positive
Actual Neg:   1,721       802                  Actual Neg:   1,977       546
Actual Pos:     547     6,954                  Actual Pos:     916     6,585
Total: 10,024 images                           Total: 10,024 images
```

### 8.2 Clinical Ramifications of Errors
In medical imaging, classification errors carry distinct, asymmetrical costs:

1. **False Negatives ($\text{FN}$):**
   - *Clinical Scenario:* A patient with moderate pleural effusion is flagged as negative.
   - *Downstream Impact:* Delay in diagnostic thoracentesis or bedside ultrasound. In infected effusions (empyema), delays can lead to pleural loculation, sepsis, and surgical decortication. In trauma, occult hemothorax may progress to hypovolemic shock.
   - *Threshold Sensitivity:* Operating at $\tau = 0.50$ restricts FNs to 547 (5.46% of cohort). Raising the threshold to $\tau = 0.70$ increases FNs by 369 cases (to 916, or 9.14% of cohort).

2. **False Positives ($\text{FP}$):**
   - *Clinical Scenario:* A patient with bibasilar atelectasis, elevated hemidiaphragm, or pleural thickening is incorrectly flagged as effusion positive.
   - *Downstream Impact:* Triggers unnecessary secondary imaging (contrast-enhanced chest CT), unnecessary antibiotic prescriptions, invasive diagnostic taps that carry risks of iatrogenic pneumothorax (~5–10% risk without real-time ultrasound guidance), patient anxiety, and unnecessary clinician alerts.
   - *Threshold Sensitivity:* Moving to $\tau = 0.70$ successfully eliminates **256 false alarms** (a 31.9% reduction from 802 to 546).

### 8.3 Contextual Decision Threshold Selection
No single threshold is universally optimal in medicine:
- **Triage / Emergency Screening:** In acute emergency settings where missing an effusion is life-threatening, a lower threshold ($\tau \approx 0.48\text{--}0.50$) prioritizing high sensitivity ($\ge 92.7\%$) is clinically favored.
- **Outpatient Monitoring / High-Throughput Screening:** In routine checkups where confirmatory CT scan resources are constrained and procedure complications must be avoided, a conservative threshold ($\tau \approx 0.66\text{--}0.70$) maximizing specificity ($78.4\%$) and balanced accuracy ($83.1\%$) is superior.

---

## 9. Test Performance at Baseline Threshold ($\tau = 0.50$)

The held-out Stanford benchmark test split ($N = 202$ frontal radiographs across 200 patients) was evaluated at the preset training threshold of $0.50$ (from `test_metrics.json`):

```
Test Split Confusion Matrix (Threshold = 0.50, N = 202):
                Predicted
            Negative   Positive
Actual Neg:     77        61
Actual Pos:      4        60
Prevalence: 31.68% (64 Pos / 138 Neg)
```

### Test Performance Metrics

| Metric | Test Set Value ($\tau = 0.50$) | Validation Set Value ($\tau = 0.50$) | Performance Delta |
| :--- | :---: | :---: | :---: |
| **AUROC** | **0.83967** | 0.89735 | $-0.05768$ |
| **PR-AUC** | **0.67338** | 0.95191 | $-0.27853$ |
| **Sensitivity** | **0.93750** | 0.92708 | $+0.01042$ |
| **Specificity** | **0.55797** | 0.68212 | $-0.12415$ |
| **Precision (PPV)** | **0.49587** | 0.89660 | $\mathbf{-0.40073}$ |
| **Balanced Accuracy** | **0.74774** | 0.80460 | $-0.05686$ |
| **F1-Score** | **0.64865** | 0.91158 | $-0.26293$ |
| **Log-Loss** | **0.80849** | 0.34589 | $+0.46260$ |

---

## 10. Validation vs. Test Prevalence Disparity & Bayes' Theorem

The most striking empirical finding between splits is the drop in Precision from **89.66% on validation to 49.59% on test**, despite Sensitivity remaining virtually unchanged (92.71% vs 93.75%).

### 10.1 Mathematical Origin via Bayes' Theorem
Positive Predictive Value (Precision) is fundamentally a function of disease prevalence $\pi = P(Y=1)$:
$$\text{PPV} = \frac{\text{Sensitivity} \cdot \pi}{\text{Sensitivity} \cdot \pi + (1 - \text{Specificity}) \cdot (1 - \pi)}$$

Plugging in the empirical parameters:
1. **Validation Split ($\pi = 0.7483$):**
   $$\text{PPV}_{\text{val}} = \frac{0.9271 \times 0.7483}{(0.9271 \times 0.7483) + (1 - 0.6821) \times (1 - 0.7483)} = \frac{0.6937}{0.6937 + 0.0800} = \mathbf{89.66\%}$$
2. **Test Split ($\pi = 0.3168$):**
   $$\text{PPV}_{\text{test}} = \frac{0.9375 \times 0.3168}{(0.9375 \times 0.3168) + (1 - 0.5580) \times (1 - 0.3168)} = \frac{0.2970}{0.2970 + 0.3019} = \mathbf{49.59\%}$$

### 10.2 Implications for Deployment and Federated Aggregation
- **Prevalence Artifact, Not Feature Breakdown:** The precision drop is not primarily a degradation of the visual feature extractor (AUROC remains strong at 0.8397), but rather an inevitable consequence of evaluating a model calibrated on high-prevalence data ($\sim 75\%$) against a lower-prevalence benchmark ($\sim 32\%$).
- **Federated Generalization Risk:** In the federated network, NIH ($14.7\%$ train prevalence) and PadChest ($16.4\%$ train prevalence) exhibit even lower base rates than CheXpert test data. If the CheXpert local model is evaluated out-of-domain on NIH without threshold recalibration, precision will decline further unless the classifier bias term is adjusted.

---

## 11. Limitations, Risks, and Recommended Next Steps

### 11.1 Methodological Limitations
1. **Small Test Set Sample Size ($N = 202$):**
   The Stanford consensus benchmark test split contains only 202 frontal radiographs across 200 patients (64 positives, 138 negatives). A single misclassified image shifts sensitivity by $1.56\%$ ($\frac{1}{64}$) and specificity by $0.72\%$ ($\frac{1}{138}$). Empirical metrics on this test set possess wide sampling variance.
2. **Image-Level vs. Patient-Level Performance:**
   The validation split contains 10,024 images from 4,081 patients (mean $\approx 2.46$ images/patient). Standard evaluation treats each image independently. Because clinical treatment decisions occur at the patient encounter level, image-level AUROC does not directly equal patient-level diagnostic accuracy.
3. **Frontal-Only Limitation:**
   Excluding lateral radiographs (15.94% of CheXpert) ensures cross-client interoperability with NIH (which lacks lateral views), but prevents detection of shallow fluid accumulations in the posterior costophrenic angles (visible on lateral views at volumes as small as $50\text{ mL}$, whereas frontal views typically require $>175\text{ mL}$ for blunting).
4. **Lack of Calibrated Probability Outputs:**
   Unweighted BCE loss under high training prevalence ($73.7\%$) skews raw probabilities toward positive predictions. Post-hoc calibration (e.g. Platt scaling, isotonic regression) on an independent validation set would improve probability reliability.

### 11.2 Recommendations for Future Rigor
- **Cluster-Bootstrap Confidence Intervals:**
  To measure sampling uncertainty without fabricating confidence intervals, future evaluations should perform a **patient-level cluster bootstrap** ($B = 1,000$ iterations). Resampling patients with replacement preserves intra-patient image correlations and produces authentic 95% percentile confidence intervals for AUROC, sensitivity, and specificity.
- **Multi-Center Generalization Matrix:**
  Evaluate the 500-patient CheXpert checkpoint zero-shot on the validation splits of NIH Client 01 ($N = 5,836$) and PadChest Client 03 ($N = 5,977$) to quantify true domain shift before federated aggregation.

---

## 12. Reproducibility Details & Supporting Artifacts

### 12.1 Environment and Runtime
- **Repository Root:** `D:\DS CP\adaptive-federated-aggregation`
- **Python Version:** `3.13.5`
- **PyTorch Version:** `2.13.0+cu130`
- **CUDA Runtime:** CUDA 12.8 / Driver 573.48 (NVIDIA GeForce RTX 4050 Laptop GPU)
- **Random Seed:** `42` (applied to Python `random`, `numpy`, PyTorch CPU/CUDA)

### 12.2 Primary Supporting Artifacts
- **Model Checkpoint:** [`outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt`](file:///D:/DS%20CP/adaptive-federated-aggregation/outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt)
- **Validation Diagnostic Report:** [`outputs/chexpert/chexpert_resnet50_20261027_225232/validation_diagnostic/validation_diagnostic_report.json`](file:///D:/DS%20CP/adaptive-federated-aggregation/outputs/chexpert/chexpert_resnet50_20261027_225232/validation_diagnostic/validation_diagnostic_report.json)
- **Validation Threshold Sweep Table:** [`outputs/chexpert/chexpert_resnet50_20261027_225232/validation_diagnostic/validation_threshold_sweep.csv`](file:///D:/DS%20CP/adaptive-federated-aggregation/outputs/chexpert/chexpert_resnet50_20261027_225232/validation_diagnostic/validation_threshold_sweep.csv)
- **Diagnostic Execution Script:** [`scripts/chexpert_client02_validation_diagnostic.py`](file:///D:/DS%20CP/adaptive-federated-aggregation/scripts/chexpert_client02_validation_diagnostic.py)
- **Training Run Configuration:** [`outputs/chexpert/chexpert_resnet50_20261027_225232/config.json`](file:///D:/DS%20CP/adaptive-federated-aggregation/outputs/chexpert/chexpert_resnet50_20261027_225232/config.json)
- **Training Epoch History:** [`outputs/chexpert/chexpert_resnet50_20261027_225232/history.csv`](file:///D:/DS%20CP/adaptive-federated-aggregation/outputs/chexpert/chexpert_resnet50_20261027_225232/history.csv)
- **Training Run Metadata:** [`outputs/chexpert/chexpert_resnet50_20261027_225232/run_metadata.json`](file:///D:/DS%20CP/adaptive-federated-aggregation/outputs/chexpert/chexpert_resnet50_20261027_225232/run_metadata.json)
- **Patient Manifest:** [`outputs/chexpert/chexpert_resnet50_20261027_225232/patient_manifest.json`](file:///D:/DS%20CP/adaptive-federated-aggregation/outputs/chexpert/chexpert_resnet50_20261027_225232/patient_manifest.json)
- **Validation Benchmark Source:** [`data/processed/CheXpert/val.csv`](file:///D:/DS%20CP/adaptive-federated-aggregation/data/processed/CheXpert/val.csv)
- **Held-Out Test Metrics:** [`outputs/chexpert/chexpert_resnet50_20261027_225232/test_metrics.json`](file:///D:/DS%20CP/adaptive-federated-aggregation/outputs/chexpert/chexpert_resnet50_20261027_225232/test_metrics.json)

