import argparse
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from augmentation import augmentation
from premodel import aril, har1, har3, signfi, stanfi
from train import set_train_model


DATASETS = {
    "aril": {"loader": aril, "input_size": 52, "num_classes": 6},
    "har-1": {"loader": har1, "input_size": 104, "num_classes": 4},
    "har-3": {"loader": har3, "input_size": 256, "num_classes": 5},
    "signfi": {"loader": signfi, "input_size": 90, "num_classes": 276},
    "stanfi": {"loader": stanfi, "input_size": 90, "num_classes": 6},
}


def build_smoke_loaders(dataset_name: str, batch_size: int, max_train_samples: int, max_test_samples: int):
    dataset_cfg = DATASETS[dataset_name]
    X_train, y_train, X_test, y_test = dataset_cfg["loader"]()

    X_train = X_train[:max_train_samples]
    y_train = y_train[:max_train_samples]
    X_test = X_test[:max_test_samples]
    y_test = y_test[:max_test_samples]

    _, _, X_train_aug, _, _, y_train_aug = augmentation(X_train, y_train)

    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_train_aug, dtype=torch.float32),
        torch.as_tensor(y_train_aug, dtype=torch.float32),
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_test, dtype=torch.float32),
        torch.as_tensor(y_test, dtype=torch.float32),
    )

    train_subset = Subset(train_dataset, range(min(batch_size, len(train_dataset))))
    test_subset = Subset(test_dataset, range(min(batch_size, len(test_dataset))))

    train_loader = DataLoader(train_subset, batch_size=min(batch_size, len(train_subset)), shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=min(batch_size, len(test_subset)), shuffle=False)
    return train_loader, test_loader


def smoke_one_dataset(dataset_name: str, device: torch.device, batch_size: int, learning_rate: float, hidden_size: int,
                      attention_dim: int, max_train_samples: int, max_test_samples: int):
    cfg = DATASETS[dataset_name]
    print(f"\n===== Smoke test: {dataset_name} =====", flush=True)

    train_loader, test_loader = build_smoke_loaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )

    model, criterion, optimizer, scheduler = set_train_model(
        device=device,
        input_size=cfg["input_size"],
        hidden_size=hidden_size,
        attention_dim=attention_dim,
        num_classes=cfg["num_classes"],
        learningrate=learning_rate,
        epochs=1,
    )

    model.train()
    train_inputs, train_labels = next(iter(train_loader))
    train_inputs = train_inputs.to(device=device, dtype=torch.float32)
    train_labels = train_labels.to(device=device, dtype=torch.float32)
    train_targets = torch.argmax(train_labels, dim=1)

    optimizer.zero_grad(set_to_none=True)
    train_outputs = model(train_inputs)
    train_loss = criterion(train_outputs, train_targets)
    train_loss.backward()
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        test_inputs, test_labels = next(iter(test_loader))
        test_inputs = test_inputs.to(device=device, dtype=torch.float32)
        test_labels = test_labels.to(device=device, dtype=torch.float32)
        test_targets = torch.argmax(test_labels, dim=1)
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        test_preds = torch.argmax(test_outputs, dim=1)
        test_acc = (test_preds == test_targets).float().mean().item()

    print(
        f"{dataset_name}: train batch {tuple(train_inputs.shape)}, "
        f"test batch {tuple(test_inputs.shape)}, "
        f"train loss {train_loss.item():.4f}, test loss {test_loss.item():.4f}, "
        f"test acc {test_acc:.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Quick GPU smoke test for all datasets.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--learningrate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=32)
    parser.add_argument("--max-train-samples", type=int, default=8)
    parser.add_argument("--max-test-samples", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This smoke test is configured to require a GPU.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Using GPU: {gpu_name}", flush=True)
    print(f"Working directory: {Path.cwd()}", flush=True)

    passed = []
    failed = []

    for dataset_name in args.datasets:
        try:
            smoke_one_dataset(
                dataset_name=dataset_name,
                device=device,
                batch_size=args.batchsize,
                learning_rate=args.learningrate,
                hidden_size=args.hidden_size,
                attention_dim=args.attention_dim,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
            )
            passed.append(dataset_name)
        except Exception as exc:
            failed.append(dataset_name)
            print(f"{dataset_name}: FAILED with {exc.__class__.__name__}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)

    print("\n===== Smoke test summary =====", flush=True)
    print(f"Passed: {passed}", flush=True)
    print(f"Failed: {failed}", flush=True)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
