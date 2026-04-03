"""
Benchmark single-sample inference latency for each dataset's latest pruned checkpoint.

Run from the repo root:
    python benchmark_single_sample.py
"""

import csv
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# ── repo root on the path ────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from ARIL.aril import aril
from HAR.har import har1, har3
from SignFi.signfi import signfi
from StanFi.stanfi import stanfi
from PrunedAttentionGRU import prunedAttentionGRU

warnings.filterwarnings("ignore")

# ── latest pruned checkpoint for each dataset ─────────────────────────────────
CHECKPOINTS = {
    "aril":    REPO / "results/training_jobs/20260327T142902Z_aril/model.pt",
    "har-1":   REPO / "results/training_jobs/20260327T154618Z_har-1/model.pt",
    "har-3":   REPO / "results/training_jobs/20260327T210654Z_har-3/model.pt",
    "signfi":  REPO / "results/training_jobs/20260328T000619Z_signfi/model.pt",
    "stanfi":  REPO / "results/training_jobs_stanfi/20260331T102238Z_stanfi/model.pt",
}

# ── dataset loaders (return X_train, y_train, X_test, y_test) ────────────────
DATA_LOADERS = {
    "aril":   aril,
    "har-1":  har1,
    "har-3":  har3,
    "signfi": signfi,
    "stanfi": stanfi,
}

N_WARMUP  = 20   # forward passes to warm up GPU/CPU caches before timing
N_REPEATS = 100  # forward passes per sample used for the timing average


def load_model(checkpoint_path: Path, device: torch.device) -> prunedAttentionGRU:
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd   = ckpt["model_state_dict"]

    hidden_size   = sd["gru.update_gate.weight"].shape[0]
    input_size    = sd["gru.update_gate.weight"].shape[1] - hidden_size
    attention_dim = sd["attention.query.weight"].shape[0]
    num_classes   = sd["fc.weight"].shape[0]

    print(f"  arch  : input={input_size}  hidden={hidden_size}"
          f"  attn={attention_dim}  classes={num_classes}")

    model = prunedAttentionGRU(input_size, hidden_size, attention_dim, num_classes)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def benchmark(model: prunedAttentionGRU, X_test: torch.Tensor,
              device: torch.device, n_warmup: int, n_repeats: int) -> dict:
    """Run inference one sample at a time; return latency stats in ms."""
    latencies = []

    with torch.no_grad():
        for idx in range(len(X_test)):
            sample = X_test[idx].unsqueeze(0).to(device=device, dtype=torch.float32)

            # warm up on this sample
            for _ in range(n_warmup):
                _ = model(sample)

            if device.type == "cuda":
                torch.cuda.synchronize()

            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                _ = model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)

            latencies.append(float(np.mean(times)))

    return {
        "mean_ms":   float(np.mean(latencies)),
        "std_ms":    float(np.std(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms":    float(np.min(latencies)),
        "max_ms":    float(np.max(latencies)),
        "n_samples": len(latencies),
    }


def save_results(results: dict, device_name: str, out_dir: Path, timestamp: str) -> None:
    """Save results as both JSON and CSV under out_dir; never overwrites."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{timestamp}_single_sample_latency.json"
    csv_path  = out_dir / f"{timestamp}_single_sample_latency.csv"

    # guard: refuse to overwrite if somehow the timestamp collides
    for p in (json_path, csv_path):
        if p.exists():
            raise FileExistsError(f"Output file already exists: {p}")

    payload = {
        "timestamp_utc": timestamp,
        "device": device_name,
        "n_warmup": N_WARMUP,
        "n_repeats_per_sample": N_REPEATS,
        "checkpoints": {k: str(v) for k, v in CHECKPOINTS.items()},
        "results": results,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON saved : {json_path}")

    csv_fields = ["dataset", "checkpoint", "n_samples",
                  "mean_ms", "std_ms", "median_ms", "min_ms", "max_ms"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for name, s in results.items():
            writer.writerow({
                "dataset":    name,
                "checkpoint": str(CHECKPOINTS[name]),
                "n_samples":  s["n_samples"],
                "mean_ms":    s["mean_ms"],
                "std_ms":     s["std_ms"],
                "median_ms":  s["median_ms"],
                "min_ms":     s["min_ms"],
                "max_ms":     s["max_ms"],
            })
    print(f"CSV saved  : {csv_path}")


def main():
    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    out_dir     = REPO / "results" / "benchmark_latency"

    print(f"Timestamp : {timestamp}")
    print(f"Device    : {device_name}")
    print(f"Output    : {out_dir}")
    print("=" * 60)

    results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n[{name}]")
        print(f"  checkpoint : {ckpt_path.relative_to(REPO)}")

        # load test data
        print("  loading data …")
        _, _, X_test, _ = DATA_LOADERS[name]()
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)

        print(f"  test shape : {tuple(X_test.shape)}")

        # load model
        model = load_model(ckpt_path, device)

        # run benchmark
        print(f"  benchmarking ({N_WARMUP} warmup + {N_REPEATS} timed per sample) …")
        stats = benchmark(model, X_test, device, N_WARMUP, N_REPEATS)
        results[name] = stats

        print(f"  mean   : {stats['mean_ms']:.4f} ms")
        print(f"  std    : {stats['std_ms']:.4f} ms")
        print(f"  median : {stats['median_ms']:.4f} ms")
        print(f"  min    : {stats['min_ms']:.4f} ms")
        print(f"  max    : {stats['max_ms']:.4f} ms")
        print(f"  n_samples measured : {stats['n_samples']}")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Dataset':<10} {'Mean (ms)':>12} {'Std (ms)':>10} {'Median (ms)':>13} {'N':>6}")
    print(f"{'-'*10} {'-'*12} {'-'*10} {'-'*13} {'-'*6}")
    for name, s in results.items():
        print(f"{name:<10} {s['mean_ms']:>12.4f} {s['std_ms']:>10.4f}"
              f" {s['median_ms']:>13.4f} {s['n_samples']:>6}")

    # ── persist results ───────────────────────────────────────────────────────
    save_results(results, device_name, out_dir, timestamp)


if __name__ == "__main__":
    main()
