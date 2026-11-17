"""Run adaptive federated rounds across the NIH, CheXpert, and PadChest clients."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.federated import AdaptiveFederatedAggregator, ClientUpdate, FedAvgAggregator
from src.models.resnet import build_resnet50_binary, load_checkpoint, save_checkpoint


ROOT = PROJECT_ROOT
CLIENTS = {
    "NIH": {
        "script": ROOT / "src" / "train_nih.py",
        "processed_default": "data/processed/NIH",
        "processed_arg": "nih_processed_dir",
        "image_arg": "nih_image_dir",
    },
    "CheXpert": {
        "script": ROOT / "src" / "train_chexpert.py",
        "processed_default": "data/processed/CheXpert",
        "processed_arg": "chexpert_processed_dir",
        "image_arg": "chexpert_image_dir",
    },
    "PadChest": {
        "script": ROOT / "src" / "train_padchest.py",
        "processed_default": "data/processed/PadChest",
        "processed_arg": "padchest_processed_dir",
        "image_arg": "padchest_image_dir",
    },
}


@dataclass(frozen=True)
class RunContext:
    args: argparse.Namespace
    root: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and adaptively aggregate a shared ResNet-50 across three medical-image clients."
    )
    parser.add_argument("--data_root", type=Path, default=ROOT)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/federated"))
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--clients", nargs="+", choices=CLIENTS, default=list(CLIENTS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sample_weight_power", type=float, default=0.5)
    parser.add_argument(
        "--aggregation",
        choices=["adaptive", "fedavg"],
        default="adaptive",
        help="Adaptive conflict projection or sample-proportional FedAvg baseline.",
    )
    parser.add_argument("--patient_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--sample_from_accessible", action="store_true")
    parser.add_argument("--all_views", action="store_true")
    parser.add_argument("--skip_cross_client_eval", action="store_true")
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    for client, details in CLIENTS.items():
        parser.add_argument(
            f"--{details['processed_arg']}",
            default=details["processed_default"],
            help=f"Processed split directory for {client}.",
        )
        parser.add_argument(
            f"--{details['image_arg']}",
            type=Path,
            default=None,
            help=f"Optional image root for {client}.",
        )

    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1.")
    if args.local_epochs < 1:
        parser.error("--local_epochs must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")
    if args.patient_limit is not None and args.patient_limit < 1:
        parser.error("--patient_limit must be positive.")
    if len(set(args.clients)) != len(args.clients):
        parser.error("--clients must not contain duplicates.")
    return args


def _resolve_path(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _client_command(
    context: RunContext,
    client: str,
    checkpoint: Path,
    output_dir: Path,
    evaluate_only: bool = False,
) -> list[str]:
    args = context.args
    root = context.root
    details = CLIENTS[client]
    processed_dir = _resolve_path(root, getattr(args, details["processed_arg"]))
    command = [
        sys.executable,
        str(details["script"]),
        "--data_root",
        str(root),
        "--processed_dir",
        str(processed_dir),
        "--output_dir",
        str(output_dir),
        "--run_name",
        "evaluation" if evaluate_only else "local",
        "--init_checkpoint",
        str(checkpoint),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.local_epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--img_size",
        str(args.img_size),
        "--num_workers",
        str(args.num_workers),
        "--dropout",
        str(args.dropout),
        "--no-pretrained",
    ]
    image_dir = getattr(args, details["image_arg"])
    if image_dir is not None:
        command.extend(["--image_dir", str(_resolve_path(root, image_dir))])
    if args.device:
        command.extend(["--device", args.device])
    if args.patient_limit is not None:
        command.extend(["--patient_limit", str(args.patient_limit)])
    if args.sample_from_accessible:
        command.append("--sample_from_accessible")
    if args.all_views and client != "NIH":
        command.append("--no-frontal-only")
    if evaluate_only:
        command.append("--evaluate_only")
    return command


def _run_client(
    context: RunContext,
    client: str,
    checkpoint: Path,
    output_dir: Path,
    evaluate_only: bool = False,
) -> Path:
    command = _client_command(
        context, client, checkpoint, output_dir, evaluate_only=evaluate_only
    )
    subprocess.run(command, cwd=context.root, check=True)
    run_dir = output_dir / ("evaluation" if evaluate_only else "local")
    if evaluate_only:
        metrics_path = run_dir / "test_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Client {client} did not write evaluation metrics: {metrics_path}")
        return metrics_path

    best_checkpoint = run_dir / "best_checkpoint.pt"
    final_checkpoint = run_dir / "final_checkpoint.pt"
    checkpoint_path = best_checkpoint if best_checkpoint.is_file() else final_checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Client {client} completed without writing a usable checkpoint in {run_dir}."
        )
    return checkpoint_path


def _evaluate_cross_client(
    context: RunContext,
    checkpoints: dict[str, Path],
    global_checkpoint: Path,
) -> dict[str, Any]:
    args = context.args
    sources = {f"local:{name}": path for name, path in checkpoints.items()}
    sources["global"] = global_checkpoint
    results: dict[str, Any] = {}
    for source_name, source_checkpoint in sources.items():
        results[source_name] = {}
        for target in args.clients:
            eval_dir = (
                context.output_dir
                / "cross_client_evaluation"
                / source_name.replace(":", "_")
                / target
            )
            metrics_path = _run_client(
                context,
                target,
                source_checkpoint,
                eval_dir,
                evaluate_only=True,
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            results[source_name][target] = {
                key: metrics.get(key)
                for key in (
                    "total_samples",
                    "positive_fraction",
                    "auroc",
                    "pr_auc",
                    "sensitivity",
                    "specificity",
                    "evaluation_status",
                    "coverage_percentage",
                    "warnings",
                )
            }
    return results


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project data root does not exist: {root}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"adaptive_federated_{timestamp}"
    output_base = _resolve_path(root, args.output_dir)
    output_dir = output_base / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    context = RunContext(args=args, root=root, output_dir=output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet50_binary(
        pretrained=args.pretrained and args.init_checkpoint is None,
        dropout_rate=args.dropout,
        device="cpu",
    )
    if args.init_checkpoint is not None:
        initial_checkpoint = _resolve_path(root, args.init_checkpoint)
        load_checkpoint(initial_checkpoint, model=model, device="cpu")

    if args.aggregation == "adaptive":
        aggregator = AdaptiveFederatedAggregator(
            sample_weight_power=args.sample_weight_power
        )
        algorithm = "adaptive_conflict_aware_fedavg"
        effective_sample_weight_power = args.sample_weight_power
    else:
        aggregator = FedAvgAggregator()
        algorithm = "fedavg"
        effective_sample_weight_power = 1.0

    run_config = {
        "algorithm": algorithm,
        "target_disease": "Pleural Effusion",
        "model_architecture": "resnet50",
        "clients": args.clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "sample_weight_power": effective_sample_weight_power,
        "device": device,
        "seed": args.seed,
        "pretrained_backbone": args.pretrained and args.init_checkpoint is None,
        "cross_client_evaluation": not args.skip_cross_client_eval,
        "note": "Single-machine coordinator simulation; client data are not kept on separate hosts.",
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    global_checkpoint = output_dir / "global_round_000.pt"
    save_checkpoint(
        global_checkpoint,
        model=model,
        config={
            **run_config,
            "round": 0,
            "initialization": (
                "provided" if args.init_checkpoint else "pretrained_or_random"
            ),
        },
    )
    round_history: list[dict[str, Any]] = []
    final_client_checkpoints: dict[str, Path] = {}

    for round_index in range(1, args.rounds + 1):
        print(f"\n[Round {round_index}/{args.rounds}] Training local client models...")
        updates: list[ClientUpdate] = []
        client_metrics: dict[str, Any] = {}
        round_dir = output_dir / "client_rounds" / f"round_{round_index:03d}"
        for client in args.clients:
            client_dir = round_dir / client
            client_checkpoint = _run_client(
                context, client, global_checkpoint, client_dir
            )
            checkpoint_data = load_checkpoint(client_checkpoint, device="cpu")
            model_config = checkpoint_data.get("config", {})
            num_examples = int(model_config.get("resolved_train_images", 0))
            if num_examples <= 0:
                raise ValueError(
                    f"Client {client} checkpoint is missing a positive resolved_train_images count."
                )
            updates.append(
                ClientUpdate(
                    client_id=client,
                    state_dict=checkpoint_data["model_state_dict"],
                    num_examples=num_examples,
                )
            )
            final_client_checkpoints[client] = client_checkpoint
            val_metrics_path = client_dir / "local" / "val_metrics.json"
            if val_metrics_path.is_file():
                client_metrics[client] = json.loads(
                    val_metrics_path.read_text(encoding="utf-8")
                )

        global_state = load_checkpoint(global_checkpoint, device="cpu")["model_state_dict"]
        aggregation = aggregator.aggregate(global_state, updates)
        model.load_state_dict(aggregation.state_dict, strict=True)
        global_checkpoint = output_dir / f"global_round_{round_index:03d}.pt"
        save_checkpoint(
            global_checkpoint,
            model=model,
            epoch=round_index,
            config={**run_config, "round": round_index},
        )
        round_report = {
            "round": round_index,
            "client_examples": {
                update.client_id: update.num_examples for update in updates
            },
            "client_weights": aggregation.client_weights,
            "pairwise_update_cosines": aggregation.pairwise_cosines,
            "projected_conflicts": aggregation.projected_conflicts,
            "local_validation_metrics": client_metrics,
            "global_checkpoint": str(global_checkpoint),
        }
        round_history.append(round_report)
        (output_dir / "round_history.json").write_text(
            json.dumps(round_history, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"[Round {round_index}] Aggregated {len(updates)} client updates; "
            f"projected {aggregation.projected_conflicts} conflicting directions."
        )

    final_checkpoint = output_dir / "global_checkpoint.pt"
    save_checkpoint(
        final_checkpoint,
        model=model,
        epoch=args.rounds,
        config={**run_config, "round": args.rounds},
    )
    if not args.skip_cross_client_eval:
        matrix = _evaluate_cross_client(
            context,
            final_client_checkpoints,
            final_checkpoint,
        )
        (output_dir / "cross_client_generalization.json").write_text(
            json.dumps(matrix, indent=2, default=str), encoding="utf-8"
        )
    print(f"\n[Complete] Global checkpoint and reports saved to {output_dir}")


if __name__ == "__main__":
    main()
