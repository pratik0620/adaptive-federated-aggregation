# CheXpert Client 02: Federated Learning Integration Audit Report

**Project Title:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
**Target Node:** CheXpert Client 02 — Stanford Hospital (`client_02_CheXpert`)  
**Assigned Prediction Task:** Binary Pleural Effusion Classification  
**Auditor:** Antigravity AI (Independent Verification & Validation Engine)  
**Audit Date:** October 28, 2026  
**Audited Commit / Head:** `c8fd36cf692abdf1059cf6c95579f5915ba62729` (Branch: `main`)  
**Report Path:** `docs/reports/chexpert_federated_integration_audit.md`  

---

## 1. Executive Summary

This independent technical audit evaluates the code, documentation, and training run artifacts submitted for **CheXpert Client 02** to determine whether the teammate's deliverables can be integrated into the planned multi-client federated learning baseline (collaborating with NIH Client 01 and PadChest Client 03).

The teammate's handoff document ([`docs/reports/chexpert_client02_training_handoff.md`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/docs/reports/chexpert_client02_training_handoff.md)) reports successful local training of an ImageNet-pretrained ResNet-50 binary classifier on a 500-patient CheXpert subset (1,241 frontal radiographs; positive prevalence 73.73%), achieving a peak validation AUROC of **0.89735** at Epoch 3 and a test AUROC of **0.83967** on a 202-image Stanford consensus test split.

### Overall Status Assessment

| Audit Dimension | Status | Key Finding / Operational Reality |
| :--- | :---: | :--- |
| **Commit & Repository State** | **VERIFIED** | Clean git tree on `main` at commit `c8fd36c`. Documentation and validation scripts are present. Leftover prototype script `chexpert_val_diagnostic.py` at root contains machine-specific absolute paths. |
| **Model Architecture Compatibility** | **VERIFIED** | Model definition [`src/models/resnet.py:ResNet50Binary`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/models/resnet.py#L18-L80) matches the project-wide single-logit linear head (`nn.Linear(2048, 1)` with `dropout_rate = 0.0`), matching NIH and PadChest architectures across 320 parameter tensors (23,510,081 parameters). |
| **Physical Checkpoint Availability** | **BLOCKED** | Checkpoint `best_checkpoint.pt` (~282.6 MB) and run folder `outputs/chexpert/chexpert_resnet50_20261027_225232` are **missing** from the repository checkout and local filesystem. Weight tensor numerical integrity cannot be verified directly. |
| **Split Isolation & Reproducibility** | **VERIFIED** | Strict patient-level isolation (zero train/val overlap) is verified via [`src/data/patient_sampler.py:select_training_patients`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/data/patient_sampler.py#L78-L209). Deterministic seeded sampling (`seed = 42`) guarantees reproducibility. |
| **Validation / Test Separation** | **VERIFIED** | Best-checkpoint selection was governed exclusively by validation AUROC/loss. Decision threshold sweep ($0.46 \le \tau \le 0.70$) in [`scripts/chexpert_client02_validation_diagnostic.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/scripts/chexpert_client02_validation_diagnostic.py) strictly excluded test data. |
| **Dataset Scale vs. Pilot Labeling** | **CONCERN** | The trained 500-patient cohort represents only **1.36%** of eligible CheXpert training patients (1,241 / 92,164 images). While described as a 500-patient run, it is fundamentally an intermediate pilot, not a full client representation. |
| **Federated Client Readiness** | **BLOCKED** | Training script [`src/train_chexpert.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/train_chexpert.py) and trainer [`src/trainer.py:ClientTrainer`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/trainer.py#L25-L366) are monolithic local-training loops. They lack federated API hooks (`set_weights`, `fit_round`, `evaluate`), and AdamW optimizer state handling across communication rounds requires explicit reset semantics. |
| **Local Diagnostic Execution** | **BLOCKED** | Running `python scripts/chexpert_client02_validation_diagnostic.py` currently fails immediately with `[Error] Run directory does not exist`. Furthermore, `torchvision` and `pillow` are missing from the current Python virtual environment. |

---

## 2. Verified Commits and File Changes

### 2.1 Working-Tree and Branch State
- **Current Branch:** `main` (tracked to `origin/main`, synchronized).
- **Working Tree:** Clean (no uncommitted or untracked changes).
- **Current HEAD Commit:** `c8fd36cf692abdf1059cf6c95579f5915ba62729`
- **Prior Commit:** `bba34119d371fc6f576a3bf84e85b6ade67bd390`

### 2.2 Teammate Commit Breakdown

#### Commit `c8fd36c` (October 27, 2026, 23:49:44 +0530)
- **Commit Message:** `docs: add CheXpert Client 02 training handoff`
- **Files Added:**
  1. [`docs/reports/chexpert_client02_training_handoff.md`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/docs/reports/chexpert_client02_training_handoff.md) (139 lines):
     Summarizes training hardware, hyperparameter configuration, validation metric parity gate results, and test evaluation metrics. Explicitly marks federated aggregation orchestration as outside the teammate's scope.
  2. [`docs/reports/chexpert_client02_model_evaluation.md`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/docs/reports/chexpert_client02_model_evaluation.md) (317 lines):
     In-depth scientific and clinical evaluation report covering clinical background, mathematical task formulation, loss convergence across 5 epochs, validation threshold sweep ($0.46 \le \tau \le 0.70$), clinical confusion matrix trade-offs, and Bayesian explanation of precision drop between validation and test splits.
  3. [`scripts/chexpert_client02_validation_diagnostic.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/scripts/chexpert_client02_validation_diagnostic.py) (685 lines):
     Automated validation-only audit script. Contains preflight integrity checks, patient isolation assertion, validation image path resolution, metric parity verification at $\tau = 0.50$, and validation threshold sweep saving JSON and CSV artifacts. Configured with dynamic repo root resolution `REPO_ROOT = Path(__file__).resolve().parent.parent`.

#### Commit `bba3411` (October 27, 2026, 22:07:45 +0530)
- **Commit Message:** `Add CheXpert Client 02 validation diagnostic and update .gitignore`
- **Files Added / Modified:**
  1. [`.gitignore`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/.gitignore):
     Added `*.onnx` and `*.safetensors` patterns alongside existing weight exclusions (`checkpoints/`, `outputs/`, `*.pt`, `*.pth`, `*.ckpt`).
  2. [`chexpert_val_diagnostic.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/chexpert_val_diagnostic.py) (410 lines at repository root):
     *Audit Finding / Code Hygiene Concern:* This is an earlier prototype of the validation diagnostic. Crucially, line 30 contains a hardcoded absolute path to the teammate's local machine: `REPO_ROOT = Path(r"D:\DS CP\adaptive-federated-aggregation").resolve()`. It also targets an older pilot run (`outputs/chexpert/chexpert_pilot_resnet50_20261015_231055`). This file is obsolete and superseded by `scripts/chexpert_client02_validation_diagnostic.py`.

### 2.3 Core Pipeline Files Checked
- **Shared Model Architecture:** [`src/models/resnet.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/models/resnet.py) (`ResNet50Binary`, `build_resnet50_binary`, `save_checkpoint`, `load_checkpoint`)
- **Training Orchestration:** [`src/train_chexpert.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/train_chexpert.py), [`src/trainer.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/trainer.py) (`ClientTrainer`)
- **Dataset & Preprocessing:** [`src/data/chexpert_dataset.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/data/chexpert_dataset.py) (`CheXpertDataset`, `CheXpertPathResolver`, `filter_chexpert_frontal_views`, `get_chexpert_transforms`)
- **Patient Cohort Sampler:** [`src/data/patient_sampler.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/data/patient_sampler.py) (`select_training_patients`, `create_stratified_patient_sequence`)
- **Evaluation Metrics:** [`src/metrics.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/metrics.py) (`compute_binary_metrics`)

---

## 3. Shared-Model Compatibility

### 3.1 Model Architecture & Classifier Head Definition
- **Status:** **VERIFIED**
- **Backbone Extractor:** Standard ResNet-50 (`torchvision.models.resnet50`). The final fully connected layer (`backbone.fc`) is replaced with `nn.Identity()` in [`src/models/resnet.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/models/resnet.py#L46) to emit a 2048-dimensional feature embedding.
- **Classification Head:** When `dropout_rate == 0.0`, [`ResNet50Binary`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/models/resnet.py#L54-L55) creates:
  ```python
  self.classifier = nn.Linear(2048, 1)
  ```
  This projects the 2048 features directly to a single unconstrained scalar logit $z \in \mathbb{R}$.
- **Parameter Breakdown:**
  - Total parameter tensors: **320 tensors** (161 weights, 106 biases/running stats, 53 `num_batches_tracked` integer buffers).
  - Total trainable parameters: **23,510,081** (Backbone: 23,508,032; Classifier: 2,048 weights + 1 bias = 2,049).

### 3.2 Preprocessing and Input Geometry
- **Status:** **VERIFIED**
- **Input Dimensions:** $(N, 3, 224, 224)$, where $N$ is mini-batch size.
- **Channel Expansion:** Grayscale single-channel chest radiographs are replicated into 3-channel RGB representation via `load_image_rgb` ([`src/data/nih_dataset.py:load_image_rgb`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/data/nih_dataset.py#L32-L55)).
- **Spatial Interpolation:** Bilinear resizing to $224 \times 224$ pixels.
- **Normalization Standard:** Standard ImageNet constants:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
  Applied across all three client pipelines (`train_chexpert.py`, `train_nih.py`, `train_padchest.py`).

### 3.3 State Dictionary Structure and Strict Loading
- **Status (Code Definition):** **VERIFIED**
- **Status (Physical Checkpoint Weights):** **NOT VERIFIED** (due to missing weight file)
- **Serialized Keys Format:** In [`src/models/resnet.py:save_checkpoint`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/models/resnet.py#L92-L132), the checkpoint dictionary format is:
  ```python
  {
      "model_architecture": "resnet50",
      "num_classes": 1,
      "classification_head": "single_logit_binary",
      "pretrained": True,
      "dropout_rate": 0.0,
      "model_state_dict": {
          "backbone.conv1.weight": torch.Tensor(64, 3, 7, 7),
          "backbone.bn1.weight": torch.Tensor(64),
          "backbone.bn1.bias": torch.Tensor(64),
          "backbone.bn1.running_mean": torch.Tensor(64),
          "backbone.bn1.running_var": torch.Tensor(64),
          "backbone.bn1.num_batches_tracked": torch.Tensor(()),
          # ... residual stages layer1 through layer4 ...
          "classifier.weight": torch.Tensor(1, 2048),
          "classifier.bias": torch.Tensor(1)
      },
      "epoch": 3,
      "best_val_metric": 0.8973476016,
      "metric_name": "val_auroc",
      "config": { ... },
      "optimizer_state_dict": { ... },
      "scheduler_state_dict": { ... }
  }
  ```
- **Dropout Rate Caveat:** Notice that in [`src/models/resnet.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/models/resnet.py#L49-L55), if `dropout_rate > 0.0`, `self.classifier` is instantiated as `nn.Sequential(nn.Dropout, nn.Linear)`, resulting in parameter keys `classifier.1.weight` and `classifier.1.bias`. Because all clients default to `dropout_rate = 0.0`, the keys align as `classifier.weight` and `classifier.bias`. If any client enables dropout, strict weight loading (`strict=True`) will fail.

### 3.4 Compatibility for Federated Averaging (FedAvg)
- **Cross-Client Alignment:** All three clients (`client_01_NIH`, `client_02_CheXpert`, `client_03_PadChest`) import the exact same [`build_resnet50_binary`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/models/resnet.py#L82-L89) factory. Parameter shapes and layer names are 100% congruent.
- **Unit Test Coverage:** [`tests/test_client_pipelines.py:test_federated_checkpoint_compatibility_across_clients`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/tests/test_client_pipelines.py#L238-L301) verifies mathematical element-wise parameter averaging:
  $$w_{\text{global}} = \sum_{k=1}^K \frac{n_k}{N} w_k$$
  across simulated client state dictionaries and confirms forward-pass execution.
- **BatchNorm Running Statistics:** ResNet-50 includes 53 `BatchNorm2d` modules. Standard FedAvg averages both convolutional weights and affine batchnorm parameters (`weight`, `bias`). However, federated coordinators must decide whether to aggregate non-trainable running buffers (`running_mean`, `running_var`, `num_batches_tracked`) or maintain client-local statistics to mitigate non-IID feature distribution shift.

---

## 4. Checkpoint Availability and Verification Status

### 4.1 Checkpoint Availability Audit
- **Status:** **BLOCKED**
- **Reported Location:** `outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt`
- **Verification on Disk:**
  - Directory `outputs/chexpert/chexpert_resnet50_20261027_225232` **does not exist** in the workspace.
  - Zero `.pt`, `.pth`, `.ckpt`, or `.safetensors` files exist anywhere in the repository.
  - Existing folders in `outputs/chexpert/` are older dry-runs from October 11, 2026 (`chexpert_pilot_resnet50_20261011_162317`, `chexpert_pilot_resnet50_20261011_164650`, and an empty `chexpert_resnet50_20261011_160552`).
- **Explanation:** In compliance with repository `.gitignore` policies and instructions in [`docs/traning_guide/CHEXPERT_TRAINING_GUIDE.md`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/docs/traning_guide/CHEXPERT_TRAINING_GUIDE.md) Section 6 ("Do NOT commit or upload images, dataset archives, or model checkpoints to GitHub"), the ~282.6 MB binary checkpoint was excluded from the Git commit. The teammate documented that artifacts reside on their local machine (drive `D:`) and must be retrieved via team shared cloud storage.

### 4.2 Diagnostic Execution Failure
Attempting to run the teammate's verification diagnostic script:
```powershell
python scripts/chexpert_client02_validation_diagnostic.py
```
exits with code 1:
```text
[Error] Run directory does not exist: E:\College\Third Year\Data Science\Course Project\Project Repo\outputs\chexpert\chexpert_resnet50_20261027_225232
```

### 4.3 Weight Verification Status
- **Status:** **NOT VERIFIED**
- Because the physical weight file is unavailable in the checkout, this audit **cannot verify**:
  1. Whether the saved tensor dictionary loads strictly (`strict=True`) without missing or unexpected keys.
  2. Whether any weight tensors contain `NaN`, `Inf`, or uninitialized values.
  3. Whether the serialized validation metric (`0.89735`) matches the internal checkpoint metadata.
  4. Whether recomputed validation inference matches the reported confusion matrix ($\text{TN}=1721, \text{FP}=802, \text{FN}=547, \text{TP}=6954$).

> [!WARNING]
> Compatibility of actual trained weights cannot be certified solely based on Python code inspection. Formal sign-off requires physical acquisition of `best_checkpoint.pt` and execution of the validation diagnostic.

---

## 5. Training and Evaluation Integrity

### 5.1 Patient-Level Split Isolation
- **Status:** **VERIFIED**
- **Isolation Enforcement:**
  In [`src/data/patient_sampler.py:select_training_patients`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/data/patient_sampler.py#L151-L167), explicit set intersections are computed before training:
  ```python
  leakage_with_val = selected_set & val_patients
  if leakage_with_val:
      raise AssertionError(f"Patient leakage detected! {len(leakage_with_val)} training patients overlap with validation set...")
  ```
- **Empirical Split Isolation:**
  - Training cohort: 500 unique patient IDs.
  - Validation split: 4,081 unique patient IDs (from 10,024 frontal images).
  - Test split: 200 unique patient IDs (from 202 frontal images).
  - Overlap: Exactly **0 overlapping patients** between any splits. Data leakage at the patient level is strictly zero.

### 5.2 Patient Selection & Reproducibility
- **Status:** **VERIFIED**
- **Algorithm:** [`src/data/patient_sampler.py:create_stratified_patient_sequence`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/data/patient_sampler.py#L23-L75) partitions eligible training patients by disease status, deterministically sorts their IDs, initializes seeded pseudo-random number generators (`random.Random(seed)` and `random.Random(seed + 1000)`), shuffles each partition, and proportionally interleaves positive and negative cases.
- **Reproducibility:** Under `--seed 42` and `--patient_limit 500`, this procedure deterministically yields the exact same 500 patient IDs across repeated executions.

### 5.3 Validation-Based Model Selection
- **Status:** **VERIFIED**
- **Selection Metric:** Monitored on the official 10,024-image frontal validation set after every epoch in [`src/trainer.py:ClientTrainer.train`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/trainer.py#L227-L246).
- **Epoch Trajectory:**
  - Epoch 1: Val Loss = 0.3751, Val AUROC = 0.8831
  - Epoch 2: Val Loss = 0.3882, Val AUROC = 0.8956
  - **Epoch 3 (Optimal Checkpoint):** Val Loss = **0.3459** (minimum), Val AUROC = **0.89735** (maximum)
  - Epoch 4: Val Loss = 0.3643, Val AUROC = 0.8843 (overfitting onset)
  - Epoch 5: Val Loss = 0.3637, Val AUROC = 0.8923
- Checkpoint serialization properly selected Epoch 3 before local over-specialization occurred.

### 5.4 Separation of Validation Threshold Selection and Test Evaluation
- **Status:** **VERIFIED**
- **Validation-Only Sweep:** In [`scripts/chexpert_client02_validation_diagnostic.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/scripts/chexpert_client02_validation_diagnostic.py#L125-L126), `test.csv` is explicitly excluded from loading, path resolution, and inference. The candidate threshold $\tau = 0.70$ was selected strictly from exploratory validation data (maximizing Balanced Accuracy at 83.07% and Youden's J at 0.6615).
- **Unbiased Test Evaluation:** Test split evaluation ([`docs/reports/chexpert_client02_model_evaluation.md`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/docs/reports/chexpert_client02_model_evaluation.md) Section 9) was evaluated exclusively at the default preset baseline threshold $\tau = 0.50$. The candidate threshold $\tau = 0.70$ was not tuned or applied to the test split, preventing data snooping and threshold overfitting.

### 5.5 Pilot Subset Status vs. Full Client Representation
- **Status:** **CONCERN / FINDING**
- In [`src/train_chexpert.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/train_chexpert.py#L133-L136), the `--pilot` flag is defined for 100 patients. The teammate ran `--patient_limit 500 --epochs 5` without the `--pilot` flag, resulting in the run name `chexpert_resnet50_20261027_225232`.
- **Cohort Scale Reality:**
  - CheXpert eligible frontal training pool: **36,726 unique patients** / **92,164 frontal radiographs**.
  - Selected training cohort: **500 unique patients** / **1,241 frontal radiographs**.
  - Cohort coverage: **1.36% of eligible patients** and **1.35% of eligible images**.
- **Assessment:** While the handoff document describes this as a "500-patient run following the initial 100-patient pilot," from a federated learning standpoint this remains an intermediate pilot subset. The project lead must not treat this checkpoint as representing the full CheXpert institutional distribution.

### 5.6 Image-Level Metrics vs. Patient-Level Diagnostic Performance
- **Status:** **VERIFIED (METHODOLOGY ACKNOWLEDGED)**
- All evaluations in [`src/metrics.py:compute_binary_metrics`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/metrics.py#L15-L146) operate strictly at the individual radiograph image level.
- In CheXpert, the 10,024 validation images originate from 4,081 unique patients (averaging ~2.46 radiographs per patient).
- The teammate's evaluation report properly notes this limitation: image-level AUROC does not prove patient-level diagnostic accuracy. If multiple scans exist per patient (e.g., serial ICU monitoring), correlated image errors are treated as independent events.

---

## 6. Federated Client Integration Readiness

### 6.1 Reusability of Training Code for Federated Learning
- **Status:** **BLOCKED / REQUIRES REFACTORING**
- **Current Architecture:**
  [`src/train_chexpert.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/train_chexpert.py) is a standalone command-line script. [`src/trainer.py:ClientTrainer`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/trainer.py#L25-L366) binds data loading, multi-epoch training, validation model selection, final checkpoint serialization, and test evaluation into a single execution method `trainer.train()`.
- **Missing Federated Abstraction:**
  To participate in iterative federated rounds (e.g. FedAvg, FedProx, or Adaptive Federated Aggregation), Client 02 requires a modular client class (e.g., `CheXpertFederatedClient`) exposing explicit communication hooks:
  1. `set_parameters(global_state_dict: dict[str, torch.Tensor]) -> None`: Overwrites local model weights with aggregated global parameters.
  2. `fit_round(local_epochs: int = 1, round_idx: int = 0) -> tuple[dict[str, torch.Tensor], int, dict[str, float]]`: Executes local gradient updates on private client data for $E$ epochs and returns updated state dictionary (or weight delta $\Delta w$), local training sample count $n_k$, and round metrics.
  3. `evaluate(split: str = "val") -> dict[str, float]`: Evaluates the current model on local client validation or test data.

### 6.2 Optimizer State Handling Across Communication Rounds
- **Status:** **CRITICAL ARCHITECTURAL CONCERN**
- **Current Implementation:** In [`src/trainer.py:ClientTrainer.__init__`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/trainer.py#L65-L80), the AdamW optimizer is instantiated once with initial model parameters and maintains running exponential averages of gradients ($m_t$) and squared gradients ($v_t$):
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
- **Federated Behavior Analysis:**
  - If global weights $w^{(t)}$ are injected into the client at round $t$ while old AdamW moment buffers ($m_{t-1}, v_{t-1}$) are retained, the optimizer applies momentum updates based on stale gradients from an outdated parameter trajectory. This causes severe gradient trajectory divergence (momentum drift).
  - In federated learning, adaptive optimizers must either:
    1. **Reset optimizer state per round:** Re-initialize the client optimizer at the start of each communication round, or
    2. **Use Server-Side Adaptive Optimization (e.g., FedAdam):** Clients perform standard local SGD (no local momentum), and the central coordinator maintains adaptive momentum on aggregated pseudo-gradients $\Delta w = \sum_k \frac{n_k}{N} \Delta w_k$.
- The current codebase contains no mechanism to reset or swap optimizer state between rounds.

### 6.3 Reusable Modules vs. Refactoring Requirements

| Repository Component | Current Status | Integration Action Required |
| :--- | :---: | :--- |
| [`src/models/resnet.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/models/resnet.py) | **Ready** | Fully reusable. Exposes `build_resnet50_binary`, `save_checkpoint`, `load_checkpoint`. |
| [`src/data/chexpert_dataset.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/data/chexpert_dataset.py) | **Ready** | Fully reusable. Exposes `CheXpertPathResolver`, `filter_chexpert_frontal_views`, and `create_chexpert_dataloaders`. |
| [`src/data/patient_sampler.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/data/patient_sampler.py) | **Ready** | Fully reusable for reproducible client cohort slicing. |
| [`src/metrics.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/metrics.py) | **Ready** | Reusable for computing standard binary classification metrics. |
| [`src/trainer.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/src/trainer.py) | **Refactor Required** | Refactor or subclass `ClientTrainer` to separate multi-epoch training loops from single-round federated steps. |
| [`chexpert_val_diagnostic.py`](file:///E:/College/Third%20Year/Data%20Science/Course%20Project/Project%20Repo/chexpert_val_diagnostic.py) | **Deprecate / Remove** | Obsolete root script with hardcoded `D:\` paths. Delete to avoid confusion. |

### 6.4 Metadata Accompanying Checkpoints for Reproducibility
For verifiable federated aggregation, every client checkpoint transmitted to the coordinator must contain:
1. `client_id`: Institutional identifier (`client_02_CheXpert`).
2. `global_round`: Current federated communication round index ($t$).
3. `local_epochs`: Number of local epochs completed in this round ($E$).
4. `sample_count` ($n_k$): Number of private training images used to compute FedAvg weights.
5. `patient_count`: Number of unique patients in the local training split.
6. `class_distribution`: Number of positive and negative samples (prevalence $\pi_k$).
7. `git_commit`: Active repository commit hash (`c8fd36c`).
8. `random_seed`: Seed used for DataLoader shuffling and initialization.
9. `model_state_dict`: Unwrapped PyTorch state dictionary (CPU tensors).

### 6.5 Empirical Risks from Pilot Cohort and Test Set

```
Prevalence & Precision Shift Across CheXpert Splits:
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│     Training Cohort     │     Validation Split    │      Test Benchmark     │
│   500 Patients (1,241)  │  4,081 Patients (10,024)│    200 Patients (202)   │
│   Prevalence = 73.73%   │   Prevalence = 74.83%   │   Prevalence = 31.68%   │
│   Optimized for Recall  │   PPV = 89.66% (τ=0.50) │   PPV = 49.59% (τ=0.50) │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

1. **High Positive Base Rate in Training Cohort (73.73%):**
   Unweighted BCE loss on high-prevalence training data calibrates the model to output high posterior probabilities (median positive prediction $0.9623$). When transferred to other federated nodes with lower prevalence (NIH Client 01 has ~14.7% prevalence; PadChest Client 03 has ~16.4% prevalence), the CheXpert local model will produce an elevated rate of false alarms unless the classification bias term is recalibrated.
2. **Small Test Benchmark ($N = 202$ images across 200 patients):**
   The Stanford consensus test set has only 64 positive and 138 negative cases. A single misclassified radiograph changes sensitivity by **$1.56\%$** ($\frac{1}{64}$) and specificity by **$0.72\%$** ($\frac{1}{138}$). Empirical metrics on this test set carry substantial sampling variance; multi-round federated convergence cannot be evaluated with statistical power on this split alone.

---

## 7. Blockers and Required Actions

The table below outlines all blocking issues preventing immediate federated integration, ranked by severity:

| Severity | Issue / Blocker | Repository Impact | Required Remediation Action | Responsible Party |
| :---: | :--- | :--- | :--- | :--- |
| **CRITICAL** | **Physical checkpoint file missing** | Checkpoint `outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt` does not exist in repo or on disk. | Retrieve the zipped run folder `chexpert_resnet50_20261027_225232` from team cloud storage (Google Drive / OneDrive) and place it under `outputs/chexpert/`. | Project Lead / Teammate |
| **HIGH** | **Missing virtual environment dependencies** | `torchvision` and `pillow` are missing from the current Python environment; running any model instantiation or image transforms fails. | Run in terminal:<br>`pip install torchvision pillow` | Workspace Admin |
| **HIGH** | **Lack of modular federated client API** | `src/train_chexpert.py` and `ClientTrainer` cannot receive global weights, train for 1 round, and emit state dictionaries without refactoring. | Implement a `CheXpertClient` class conforming to the planned federated aggregation interface with explicit `set_weights` and `fit_round` methods. | Federated Lead |
| **HIGH** | **Optimizer state retention across rounds** | Retaining AdamW momentum buffers across federated rounds creates momentum drift against newly aggregated global weights. | Specify federated optimizer strategy: either reset client AdamW buffers per round, or switch client to SGD and use server-side FedAdam. | Federated Lead |
| **MEDIUM** | **Leftover uncleaned prototype script** | `chexpert_val_diagnostic.py` at repository root contains hardcoded paths to teammate's private drive `D:\`. | Remove `chexpert_val_diagnostic.py` from the root directory to eliminate confusion with `scripts/chexpert_client02_validation_diagnostic.py`. | Repository Lead |
| **MEDIUM** | **Pilot cohort scale limitation** | 500-patient training run covers only 1.35% of eligible CheXpert training data. | Treat current checkpoint strictly as a pilot integration milestone; schedule full CheXpert dataset training for final experimental benchmark. | Project Lead |

---

## 8. Recommended Next Steps

Follow this exact sequential action plan to complete verification and integrate CheXpert Client 02 into the federated pipeline:

### Step 1: Install Required Environment Dependencies
Verify and install `torchvision` and `pillow` into the active virtual environment:
```powershell
python -m pip install torchvision pillow
```
Verify installation:
```powershell
python -c "import torchvision, PIL; print('TorchVision:', torchvision.__version__, '| Pillow:', PIL.__version__)"
```

### Step 2: Ingest the CheXpert Client 02 Run Folder
Download the training run folder `chexpert_resnet50_20261027_225232` from team cloud storage and place it at:
```text
outputs/chexpert/chexpert_resnet50_20261027_225232/
├── best_checkpoint.pt
├── config.json
├── history.csv
├── path_resolution_report.json
├── patient_manifest.json
├── run_metadata.json
├── test_metrics.json
└── val_metrics.json
```
Verify file presence:
```powershell
Test-Path outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt
```

### Step 3: Run the Non-Destructive Weight Verification Check
Once `best_checkpoint.pt` is in place, run this read-only Python inspection command to verify tensor shapes, parameter counts, and metadata without modifying the file:
```powershell
python -c "import torch; ckpt = torch.load('outputs/chexpert/chexpert_resnet50_20261027_225232/best_checkpoint.pt', map_location='cpu'); state = ckpt['model_state_dict']; print('Tensors:', len(state)); print('Epoch:', ckpt.get('epoch')); print('Best Metric:', ckpt.get('best_val_metric')); print('Classifier weight shape:', state['classifier.weight'].shape); print('Has NaNs:', any(torch.isnan(t).any().item() for t in state.values() if t.is_floating_point()))"
```

### Step 4: Execute the Full Validation Diagnostic and Parity Gate
Run the teammate's automated validation diagnostic:
```powershell
python scripts/chexpert_client02_validation_diagnostic.py --run_dir outputs/chexpert/chexpert_resnet50_20261027_225232
```
*Expected Result:*
- Patient isolation passes (0 overlap).
- Recomputed validation AUROC at $\tau = 0.50$ matches $0.89735$ within tolerance $10^{-4}$.
- Recomputed confusion matrix matches $\text{TN} = 1721, \text{FP} = 802, \text{FN} = 547, \text{TP} = 6954$.
- Diagnostic report is saved to `outputs/chexpert/chexpert_resnet50_20261027_225232/validation_diagnostic/validation_diagnostic_report.json`.

### Step 5: Clean Up Repository Root
Remove the obsolete prototype script containing hardcoded `D:\` paths:
```powershell
git rm chexpert_val_diagnostic.py
```

### Step 6: Define Federated Client Adapter
Create `src/federated/chexpert_client.py` wrapping `CheXpertDataset` and `ResNet50Binary` with modular `set_parameters(state_dict)` and `train_round(epochs, lr)` interfaces, resolving the AdamW optimizer state reset policy before launching multi-client federation rounds.
