# Federated Learning Baseline Guide: Standard FedAvg

**Project Title:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
**Task:** Simulated Cross-Silo Federated Learning for Binary Pleural Effusion Classification  
**Target Nodes:**
1. **Client 01:** NIH ChestX-ray14 (`client_01_NIH`)
2. **Client 02:** CheXpert (`client_02_CheXpert`)
3. **Client 03:** PadChest (`client_03_PadChest`)

---

## 1. System Overview & Modular Architecture

The federated architecture simulates a cross-silo healthcare network where raw patient radiographs and electronic health records remain strictly on-premise at each clinical institution. A central server orchestrates communication rounds, broadcasts model parameters, and aggregates parameter updates using standard Federated Averaging (FedAvg).

The code is organized under `src/federated/`:

```
src/federated/
├── __init__.py           # Package exports
├── aggregation.py        # Decoupled FedAvg weighting and state validation
├── client.py             # FederatedClient base interface & SyntheticFederatedClient
├── adapters.py           # NIH, CheXpert, and PadChest pipeline adapters
├── server.py             # FederatedServer orchestration and logging
├── config.py             # FederatedConfig dataclass and CLI argument parsing
└── run_federated.py      # CLI execution entry point
```

### Architectural Decoupling
The parameter aggregation logic (`src/federated/aggregation.py`) is decoupled from the server orchestration (`src/federated/server.py`). This ensures that future aggregation strategies (such as **Capped FedAvg** or adaptive robust weighting) can be plugged directly into `FederatedServer(aggregation_fn=...)` without altering client training loops or dataset adapters.

---

## 2. Server-Client Communication Workflow

Each federated communication round $t \in \{1, \dots, T\}$ follows a synchronous communication protocol:

```
Central Server                          Client Nodes (NIH, CheXpert, PadChest)
      │                                                   │
      │ 1. Broadcast global weights theta_t               │
      ├──────────────────────────────────────────────────►│ (set_parameters)
      │                                                   │
      │                                                   │ 2. Local Training (fit_round)
      │                                                   │    - Reset local optimizer state
      │                                                   │    - Train for E local epochs
      │                                                   │    - Count exact samples used n_k
      │ 3. Collect updates (theta_k, n_k, train_metrics)   │
      │◄──────────────────────────────────────────────────┤ (Zero raw images/records sent)
      │                                                   │
      │ 4. FedAvg Aggregation                             │
      │    p_k = n_k / sum(n_k)                           │
      │    theta_{t+1} = sum(p_k * theta_k)               │
      │    Update global model                            │
      │                                                   │
      │ 5. Evaluate Updated Model on Validation Splits   │
      ├──────────────────────────────────────────────────►│ (evaluate on val_loader)
      │◄──────────────────────────────────────────────────┤ (AUROC, PR-AUC, F1, Loss)
      │                                                   │
      │ 6. Checkpoint & Log                               │
      │    - Save round checkpoint                        │
      │    - Export history.csv & round_metrics.json      │
```

### Privacy Guarantee
Clients never transmit raw chest radiographs, image file paths, patient identifiers, or tabular patient records. Only updated PyTorch parameter dictionaries, scalar sample counts $n_k$, and aggregated evaluation metrics are exchanged across the network boundary.

---

## 3. Mathematical Formulation of FedAvg

For $K$ participating clients in communication round $t$:
Let $n_k$ denote the number of training examples actually used by client $k$ during round $t$, and let the total cohort sample size be:

$$N = \sum_{k=1}^K n_k$$

The aggregation weight for client $k$ is defined as:

$$p_k = \frac{n_k}{N}$$

The updated global model parameters $\theta_{t+1}$ are computed by taking the sample-weighted average of client parameter updates:

$$\theta_{t+1} = \sum_{k=1}^K p_k \theta_{t+1}^{(k)}$$

### Parameter and Buffer Handling
1. **Weight Constraints:** Weights are strictly validated: $p_k > 0$ and $\sum_{k=1}^K p_k = 1.0$ (within numerical tolerance $\pm 10^{-6}$).
2. **Floating-point Parameters:** All weight and bias tensors (`torch.float32`, `torch.float16`) are accumulated in `torch.float64` on CPU to prevent numerical underflow and cast back to their original dtype.
3. **Integer Model Buffers (e.g., BatchNorm `num_batches_tracked`):** Rather than performing truncated integer division or raising casting errors, integer buffers are weighted in float, rounded to the nearest integer (`torch.round`), and cast back to their original integer dtype (`torch.int64`).
4. **Strict Model Integrity:** Parameter keys, shapes, and dtypes must match across all clients. Any occurrence of `NaN`, `Inf`, or mismatched keys raises an immediate exception. Optimizer states and non-model checkpoint objects are strictly rejected from parameter aggregation.

---

## 4. Initialization and Optimizer Reset Policy

### Fair Baseline Initialization
- All clients start from the **same global model initialization** distributed by the server at Round 1.
- The teammate's pre-trained CheXpert checkpoint is retained as an independent single-client audit artifact, but is **not** injected as the starting point for only one client, preventing unfair initialization advantages.

### Optimizer Reset Policy
- Adaptive optimizers (such as AdamW) maintain first-moment ($m_t$) and second-moment ($v_t$) gradient histories. Retaining momentum buffers across rounds creates **momentum drift** against newly aggregated global parameters.
- At the start of each round, `FederatedClient.fit_round()` instantiates a fresh optimizer instance (or clears optimizer state buffers).
- Local optimizer states remain local to the client and are never sent to or averaged by the server.

---

## 5. Verification: Running Tests & Smoke Tests

### 5.1 Running the Federated Unit Test Suite
The repository includes 11 dedicated unit and integration tests verifying every requirement of the federated architecture:

```powershell
python -m pytest tests/test_federated_baseline.py -v
```

This tests:
1. `test_sample_weighted_fedavg_known_tensors`: Exact sample-weighted FedAvg against ground-truth analytical tensors.
2. `test_aggregation_weights_sum_to_one`: Weights positivity and unity constraint ($\sum p_k = 1.0$).
3. `test_rejection_of_mismatched_parameters_and_nans`: Detection and rejection of missing keys, shape mismatches, dtype mismatches, and NaNs.
4. `test_handling_floating_point_parameters_and_integer_buffers`: Explicit float accumulation and integer buffer rounding (`torch.int64`).
5. `test_identical_global_parameters_distributed_to_clients`: Strict broadcast synchronization across clients.
6. `test_client_updates_aggregated_and_broadcast`: End-to-end parameter movement and aggregation.
7. `test_optimizer_reset_between_rounds`: Verification that AdamW momentum states do not carry over between rounds.
8. `test_reproducibility_with_fixed_seed`: Bit-for-bit trajectory equivalence under fixed seeds.
9. `test_non_participating_client_handling`: Correct weight normalization when a subset of clients participate.
10. `test_privacy_no_raw_data_exchange`: Verification that client returns only parameters, sample counts, and metrics.
11. `test_synthetic_end_to_end_smoke_test`: Full 2-client, 2-round synthetic smoke test with model checkpoint save/reload.

### 5.2 Running the Synthetic Smoke Test via CLI
To exercise the complete server-client lifecycle in synthetic mode (requiring no images, external downloads, or GPU):

```powershell
python src/federated/run_federated.py --synthetic --rounds 2 --local_epochs 2 --clients client_01 client_02 client_03
```

Expected output:
- Prints round progress and participating client sample counts.
- Displays validation AUROC and validation loss per round for each client.
- Saves checkpoints (`best_global_checkpoint.pt`, `final_global_checkpoint.pt`) and logs (`history.csv`, `round_metrics.json`) to `outputs/federated/`.

---

## 6. Configuring a Small Pilot Run

To run a small pilot experiment with real datasets and restricted patient counts:

```powershell
python src/federated/run_federated.py `
    --patient_limit 50 `
    --rounds 3 `
    --local_epochs 1 `
    --batch_size 16 `
    --lr 1e-4 `
    --optimizer adamw `
    --seed 42 `
    --device cpu `
    --output_dir outputs/federated/pilot_experiment
```

### Key Arguments:
- `--patient_limit 50`: Caps unique patients per client at 50 (counting unique patients, not images).
- `--rounds 3`: Runs 3 federated communication rounds.
- `--local_epochs 1`: 1 local epoch per client per round.
- `--optimizer adamw`: Configures local optimizer (`adamw`, `sgd`, or `adam`).
- `--seed 42`: Sets deterministic patient sampling and DataLoader shuffling.

---

## 7. Storage and Artifact Locations

Each federated training run generates a timestamped directory under `outputs/federated/fedavg_<timestamp>/`:

| Artifact | Format | Description |
| :--- | :---: | :--- |
| `best_global_checkpoint.pt` | PyTorch | Model checkpoint achieving the highest mean validation AUROC across clients. |
| `final_global_checkpoint.pt` | PyTorch | Model checkpoint at the end of the final communication round. |
| `global_checkpoint_round_XX.pt` | PyTorch | Checkpoints saved at the end of every individual communication round. |
| `history.csv` | CSV | Tabular metrics per round (loss, AUROC, PR-AUC, F1, balanced accuracy, client weights, sample counts). |
| `round_metrics.json` | JSON | Complete machine-readable log of training and validation metrics for every client. |
| `federated_config.json` | JSON | Serialized run configuration (hyperparameters, participating clients, seed, device). |

---

## 8. Prerequisites Before Running Real Three-Client Training

Before launching full multi-client federated training on the physical medical datasets, resolve these environment and data prerequisites:

1. **Environment Dependencies (`torchvision` & `pillow`):**
   - The active Python 3.13 environment currently lacks `torchvision` and `pillow`.
   - Install compatible packages:
     ```powershell
     pip install torchvision pillow
     ```
   - Verify installation:
     ```powershell
     python -c "import torchvision, PIL; print('TorchVision:', torchvision.__version__, '| Pillow:', PIL.__version__)"
     ```

2. **CheXpert Client 02 Raw Images:**
   - Raw frontal radiographs must be accessible in `data/raw/CheXpert/images` (or indexed via `--image_dir`).

3. **PadChest Client 03 Raw Images:**
   - PadChest metadata CSVs exist in `data/raw/PadChest/metadata/`, but raw radiograph archives (`images/0/` through `images/49/`) must be downloaded and placed under `data/raw/PadChest/images`.

---

## 9. Distinguishing Synthetic Validation from Real Dataset Results

Scientific and clinical integrity requires unambiguous distinction between test artifacts:

- **Synthetic Smoke Tests:**
  - Evaluated on randomly generated synthetic feature vectors.
  - Used strictly for software verification (verifying pipeline plumbing, tensor shapes, and state synchronization).
  - Explicitly tagged with `"is_synthetic": true` in metadata.
  - **Never** cite synthetic AUROC or loss as empirical clinical evidence.

- **Partial Archive Subsets:**
  - When local archives contain only a subset of official split images (e.g. pilot cohorts), evaluation metrics are tagged with:
    `"evaluation_status": "PARTIAL_ARCHIVE_SUBSET_UNOFFICIAL"`.
  - Reports must explicitly include the disclaimer:
    `UNOFFICIAL: Validation evaluated on partial accessible subset (X / Y images)`.
  - Must not be reported as official full-test benchmark metrics.

- **Official Full-Benchmark Evaluation:**
  - Applicable only when 100% of official split images are resolved and evaluated.
  - Held-out test splits (`test.csv`) must **never** be used for model selection, threshold sweeps, or aggregation weight calculation during federated training. Test evaluation is performed once on the final selected global model.
