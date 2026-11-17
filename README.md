# Adaptive Federated Aggregation for Chest X-ray Classification

This project trains a shared binary classifier for **Pleural Effusion** across three heterogeneous chest X-ray clients: NIH ChestX-ray14, CheXpert, and PadChest. It includes patient-level data preparation, client-local ResNet-50 training, conflict-aware federated aggregation, and cross-client test evaluation.

The federated coordinator currently runs as a **single-machine simulation**: each client uses its own dataset directory, but all client processes and model updates are orchestrated locally. It does not provide secure transport, cryptographic aggregation, or a production multi-hospital deployment.

## What is implemented

- Dataset preprocessing notebooks and leakage-safe patient splits.
- Separate client training entry points: `src/train_nih.py`, `src/train_chexpert.py`, and `src/train_padchest.py`.
- A common ResNet-50 binary classifier and training loop with validation-based checkpoint selection.
- Federated rounds in `src/train_federated.py`: broadcast the current global checkpoint, train each client locally, and aggregate the returned model updates.
- A `--aggregation fedavg` option for a sample-proportional FedAvg baseline, so results can be compared under the same client splits and evaluation protocol.
- Adaptive conflict-aware aggregation in `src/federated.py`:
  - Detects conflicting client update directions by pairwise cosine similarity.
  - Projects conflicting components out of each update (PCGrad-style).
  - Weights updates by `n_k ** 0.5` by default, reducing large-client dominance compared with sample-proportional FedAvg.
  - Reports the client weights, pairwise similarities, and projected conflicts for every round.
- Held-out cross-client evaluation by default: final local client models and the global model are each evaluated on every participating client's test split. Results are saved as `cross_client_generalization.json`.

The adaptive rule is a project baseline, not a claim that the method has been clinically validated or proven superior. Compare it against FedAvg and other baselines on complete, permitted datasets before drawing performance conclusions.

## Data setup

Obtain each dataset and its metadata through its official source and follow its access and use terms. This repository does not download or redistribute the medical images. Keep image archives, processed records, and model checkpoints out of Git.

1. Prepare each dataset using its notebook in `notebooks/` (or use already prepared splits).
2. Ensure each processed directory contains `train.csv`, `val.csv`, and `test.csv`:
   - `data/processed/NIH/`
   - `data/processed/CheXpert/`
   - `data/processed/PadChest/`
3. Place the corresponding local image files under `data/raw/` or provide explicit image roots to the training command.

For details on supported layouts and data-specific constraints, see:

- `docs/traning_guide/NIH_TRAINING_GUIDE.md`
- `docs/traning_guide/CHEXPERT_TRAINING_GUIDE.md`
- `docs/traning_guide/PADCHEST_TRAINING_GUIDE.md`

The client loaders use frontal views by default for CheXpert and PadChest so all clients share a compatible view domain. Patient-level selection and missing-image reports are retained. If the locally available images are only a subset, pass `--sample_from_accessible` and treat resulting metrics as partial-subset results, not full-dataset benchmarks.

## Install

From the repository root:

```powershell
python -m pip install -r requirements.txt
```

PyTorch may need a platform-specific installation if the version selected by `requirements.txt` is not compatible with your operating system or CUDA setup.

## Run a federated pilot

Start with a small run to verify local data paths and the full round workflow:

```powershell
python src/train_federated.py --rounds 2 --local_epochs 1 --patient_limit 100 --batch_size 8 --device cpu --sample_from_accessible
```

Use `--device cuda` on a machine with a working CUDA-enabled PyTorch installation. Use `--aggregation fedavg` to run the FedAvg baseline with the same setup. The coordinator accepts per-client path overrides, for example:

```powershell
python src/train_federated.py `
  --rounds 5 `
  --local_epochs 1 `
  --device cuda `
  --nih_image_dir "D:\datasets\NIH\images" `
  --chexpert_image_dir "D:\datasets\CheXpert" `
  --padchest_image_dir "D:\datasets\PadChest\images"
```

Processed split directories can be overridden with `--nih_processed_dir`, `--chexpert_processed_dir`, and `--padchest_processed_dir`. Add `--skip_cross_client_eval` to omit the final cross-client test matrix when doing a quick training-only run. Use `--no-pretrained` to initialize without ImageNet weights (otherwise the initial ResNet-50 weights may need to be downloaded by torchvision).

Individual clients can also be trained outside the coordinator; for example:

```powershell
python src/train_nih.py --pilot --sample_from_accessible --device cpu
```

For a federated round, each client trainer accepts `--init_checkpoint PATH` to start from the coordinator's current global model. `--evaluate_only` evaluates that checkpoint on the client's held-out test split without training.

## Output artifacts

Runs are written under `outputs/federated/<run_name>/`, including:

- `global_checkpoint.pt` and per-round global checkpoints.
- `client_rounds/` with each client's local checkpoint, training history, and metrics.
- `round_history.json` with client sample counts, adaptive weights, update similarities, and local validation metrics.
- `cross_client_generalization.json` with each final local model and the global model evaluated on each client's held-out test split (unless disabled).
- `config.json` with the run configuration.

Evaluation reports flag incomplete test coverage. Do not describe partial archive metrics as official full-split results. Do not commit generated model checkpoints or medical data.

## Tests

Run the test suite from the repository root:

```powershell
python -m pytest -q .\tests
```
