import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from premodel import arilsetting, harsetting1, harsetting3, signfisetting, stanfisetting
from tools.mixup import mixup
from train import set_train_model


DATASETS = {
    "aril": {"loader": arilsetting, "input_size": 52, "num_classes": 6},
    "har-1": {"loader": harsetting1, "input_size": 104, "num_classes": 4},
    "har-3": {"loader": harsetting3, "input_size": 256, "num_classes": 5},
    "signfi": {"loader": signfisetting, "input_size": 90, "num_classes": 276},
    "stanfi": {"loader": stanfisetting, "input_size": 90, "num_classes": 6},
}

CSV_FIELDS = [
    "timestamp_utc",
    "dataset",
    "model",
    "seed",
    "device",
    "epochs",
    "final_train_accuracy",
    "final_train_loss",
    "validation_loss",
    "validation_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "balanced_accuracy",
    "total_parameters",
    "trainable_parameters",
    "non_trainable_parameters",
    "inference_forward_time_s",
    "inference_latency_ms_per_sample",
    "inference_latency_ms_per_batch",
    "inference_throughput_samples_per_s",
    "pre_training_time_s",
    "fine_tuning_time_s",
    "pruning_s",
    "pruning_k",
    "finetune_epochs",
    "finetune_lr",
    "training_time_s",
    "evaluation_time_s",
    "total_runtime_s",
    "checkpoint_path",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_nonzero_weights(model: torch.nn.Module) -> tuple[int, int, int]:
    """Count non-zero weights excluding mask parameters (matches paper's parameter reporting)."""
    total_nz = 0
    trainable_nz = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        nz = int((param.data != 0).sum().item())
        total_nz += nz
        if param.requires_grad:
            trainable_nz += nz
    return total_nz, trainable_nz, total_nz - trainable_nz


def save_history_plot(loss_hist: dict, acc_hist: dict, destination: Path) -> None:
    epochs = range(1, len(loss_hist["train"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_hist["train"], label="Train Loss")
    plt.plot(epochs, loss_hist["validation"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_hist["train"], label="Train Accuracy")
    plt.plot(epochs, acc_hist["validation"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History")
    plt.legend()

    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def evaluate_model(model: torch.nn.Module, dataloader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            targets = torch.argmax(labels, dim=1).long()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predictions = torch.argmax(outputs, dim=1)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += (predictions == targets).sum().item()
            total_samples += batch_size

            all_targets.extend(targets.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    avg_loss = running_loss / total_samples
    avg_accuracy = running_corrects / total_samples

    return {
        "loss": avg_loss,
        "accuracy": avg_accuracy,
        "targets": np.asarray(all_targets, dtype=np.int64),
        "predictions": np.asarray(all_predictions, dtype=np.int64),
        "num_samples": total_samples,
    }


def benchmark_inference(model: torch.nn.Module, dataloader, device: torch.device):
    model.eval()
    total_samples = 0
    total_batches = 0

    sync_device(device)
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            _ = model(inputs)
            total_samples += inputs.size(0)
            total_batches += 1
    sync_device(device)
    elapsed = time.perf_counter() - start

    return {
        "forward_time_s": elapsed,
        "latency_ms_per_sample": (elapsed / total_samples) * 1000.0,
        "latency_ms_per_batch": (elapsed / total_batches) * 1000.0,
        "throughput_samples_per_s": total_samples / elapsed,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, mixup_probability: float):
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        targets = torch.argmax(labels, dim=1).long()

        optimizer.zero_grad(set_to_none=True)
        use_mixup = np.random.random() < mixup_probability

        if use_mixup:
            mixed_inputs, label1, label2, lam = mixup(inputs, labels, 1.0)
            mixed_inputs = mixed_inputs.to(device=device, dtype=torch.float32)
            target1 = torch.argmax(label1.to(device=device, dtype=torch.float32), dim=1).long()
            target2 = torch.argmax(label2.to(device=device, dtype=torch.float32), dim=1).long()
            lam_value = float(lam.item())

            outputs = model(mixed_inputs)
            loss = criterion(outputs, target1) * lam_value + criterion(outputs, target2) * (1.0 - lam_value)
            predictions = torch.argmax(outputs, dim=1)
            correct = (
                lam_value * (predictions == target1).sum().item()
                + (1.0 - lam_value) * (predictions == target2).sum().item()
            )
            batch_size = mixed_inputs.size(0)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == targets).sum().item()
            batch_size = inputs.size(0)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_corrects += correct
        total_samples += batch_size

    return running_loss / total_samples, running_corrects / total_samples


def train_dataset(dataset_name: str, args, results_root: Path, device: torch.device, device_name: str):
    dataset_cfg = DATASETS[dataset_name]
    run_started_at = time.perf_counter()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / f"{timestamp}_{dataset_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    train_loader, validation_loader = dataset_cfg["loader"](args.batchsize)
    model, criterion, optimizer, scheduler = set_train_model(
        device=device,
        input_size=dataset_cfg["input_size"],
        hidden_size=args.hidden_size,
        attention_dim=args.attention_dim,
        num_classes=dataset_cfg["num_classes"],
        learningrate=args.learningrate,
        epochs=args.epochs,
    )

    best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_validation_accuracy = float("-inf")
    loss_hist = {"train": [], "validation": []}
    acc_hist = {"train": [], "validation": []}

    print(f"\n===== Training dataset: {dataset_name} =====", flush=True)
    training_started_at = time.perf_counter()
    for epoch in range(args.epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            mixup_probability=args.mixup_probability,
        )
        scheduler.step()

        validation_metrics = evaluate_model(model, validation_loader, criterion, device)
        loss_hist["train"].append(train_loss)
        loss_hist["validation"].append(validation_metrics["loss"])
        acc_hist["train"].append(train_accuracy)
        acc_hist["validation"].append(validation_metrics["accuracy"])

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} val_acc={validation_metrics['accuracy']:.4f}",
            flush=True,
        )

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    pre_training_time = time.perf_counter() - training_started_at

    # Load best pre-trained model
    model.load_state_dict(best_state_dict)

    # --- Pruning + Fine-tuning phase ---
    fine_tuning_time = 0.0
    if args.finetune_epochs > 0:
        print(f"\n=== Pruning model (s={args.pruning_s}, k={args.pruning_k}) ===", flush=True)
        model.prune_by_std(args.pruning_s, args.pruning_k)

        print(f"=== Fine-tuning for {args.finetune_epochs} epochs (lr={args.finetune_lr}) ===", flush=True)
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)

        best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_ft_accuracy = float("-inf")

        ft_started_at = time.perf_counter()
        for epoch in range(args.finetune_epochs):
            ft_train_loss, ft_train_accuracy = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer_ft,
                device=device,
                mixup_probability=args.mixup_probability,
            )
            scheduler_ft.step()

            ft_val_metrics = evaluate_model(model, validation_loader, criterion, device)
            print(
                f"[Fine-tune] Epoch {epoch + 1}/{args.finetune_epochs} "
                f"train_loss={ft_train_loss:.4f} train_acc={ft_train_accuracy:.4f} "
                f"val_loss={ft_val_metrics['loss']:.4f} val_acc={ft_val_metrics['accuracy']:.4f}",
                flush=True,
            )
            if ft_val_metrics["accuracy"] > best_ft_accuracy:
                best_ft_accuracy = ft_val_metrics["accuracy"]
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        fine_tuning_time = time.perf_counter() - ft_started_at
        model.load_state_dict(best_state_dict)

    training_time = pre_training_time + fine_tuning_time

    evaluation_started_at = time.perf_counter()
    train_metrics = evaluate_model(model, train_loader, criterion, device)
    validation_metrics = evaluate_model(model, validation_loader, criterion, device)
    inference_metrics = benchmark_inference(model, validation_loader, device)
    evaluation_time = time.perf_counter() - evaluation_started_at

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        validation_metrics["targets"],
        validation_metrics["predictions"],
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        validation_metrics["targets"],
        validation_metrics["predictions"],
        average="weighted",
        zero_division=0,
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        validation_metrics["targets"],
        validation_metrics["predictions"],
        average="micro",
        zero_division=0,
    )
    balanced_accuracy = balanced_accuracy_score(
        validation_metrics["targets"],
        validation_metrics["predictions"],
    )

    total_parameters, trainable_parameters, non_trainable_parameters = count_nonzero_weights(model)

    checkpoint_path = run_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": dataset_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "batchsize": args.batchsize,
            "learningrate": args.learningrate,
            "hidden_size": args.hidden_size,
            "attention_dim": args.attention_dim,
            "validation_accuracy": validation_metrics["accuracy"],
        },
        checkpoint_path,
    )

    save_history_plot(loss_hist, acc_hist, run_dir / "training_history.png")
    with (run_dir / "history.json").open("w", encoding="utf-8") as history_file:
        json.dump({"loss": loss_hist, "accuracy": acc_hist}, history_file, indent=2)

    row = {
        "timestamp_utc": timestamp,
        "dataset": dataset_name,
        "model": "prunedAttentionGRU",
        "seed": args.seed,
        "device": device_name,
        "epochs": args.epochs,
        "final_train_accuracy": train_metrics["accuracy"],
        "final_train_loss": train_metrics["loss"],
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_precision": precision_weighted,
        "weighted_recall": recall_weighted,
        "weighted_f1": f1_weighted,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
        "micro_f1": f1_micro,
        "balanced_accuracy": balanced_accuracy,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "non_trainable_parameters": non_trainable_parameters,
        "inference_forward_time_s": inference_metrics["forward_time_s"],
        "inference_latency_ms_per_sample": inference_metrics["latency_ms_per_sample"],
        "inference_latency_ms_per_batch": inference_metrics["latency_ms_per_batch"],
        "inference_throughput_samples_per_s": inference_metrics["throughput_samples_per_s"],
        "pre_training_time_s": pre_training_time,
        "fine_tuning_time_s": fine_tuning_time,
        "pruning_s": args.pruning_s,
        "pruning_k": args.pruning_k,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "training_time_s": training_time,
        "evaluation_time_s": evaluation_time,
        "total_runtime_s": time.perf_counter() - run_started_at,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }

    summary_csv = results_root / "summary.csv"
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    per_run_csv = run_dir / "metrics.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(row, metrics_file, indent=2)

    print(f"Saved results to {run_dir}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description="Train prunedAttentionGRU and export per-dataset metrics.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--learningrate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=32)
    parser.add_argument("--mixup-probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results/training_jobs")
    parser.add_argument("--pruning-s", type=float, default=0.9)
    parser.add_argument("--pruning-k", type=float, default=0.7)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(args.seed)

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = "cpu"

    print(f"Using device: {device_name}", flush=True)
    print(f"Results root: {results_root.resolve()}", flush=True)

    rows = []
    for dataset_name in args.datasets:
        rows.append(train_dataset(dataset_name, args, results_root, device, device_name))

    print("\n===== Training summary =====", flush=True)
    for row in rows:
        print(
            f"{row['dataset']}: val_acc={row['validation_accuracy']:.4f}, "
            f"val_loss={row['validation_loss']:.4f}, checkpoint={row['checkpoint_path']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
