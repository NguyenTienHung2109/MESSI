#!/usr/bin/env python3
"""
Measure inference latency for DomainBed algorithms from saved checkpoints.

Usage:
    python scripts/measure_latency.py --checkpoint multi_dataset/GMOE_Full/PACS/env0/seed0/train_output/model.pkl
    python scripts/measure_latency.py --checkpoint multi_dataset/GMOE/PACS/env0/seed0/train_output/model.pkl
    python scripts/measure_latency.py --checkpoint multi_dataset/GMOE_Full/DomainNet/env1/seed0/train_output/model.pkl --batch_sizes 1 16 32
"""

import argparse
import time
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from domainbed.algorithms import get_algorithm_class


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    algorithm_name = ckpt['args']['algorithm']
    input_shape = ckpt['model_input_shape']
    num_classes = ckpt['model_num_classes']
    num_domains = ckpt['model_num_domains']
    hparams = ckpt['model_hparams']

    # Construct model
    algorithm_class = get_algorithm_class(algorithm_name)
    algorithm = algorithm_class(input_shape, num_classes, num_domains, hparams)
    algorithm.load_state_dict(ckpt['model_dict'], strict=False)
    algorithm.to(device)
    algorithm.eval()

    return algorithm, algorithm_name, input_shape, num_classes, hparams


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_latency(model, input_shape, batch_size, device, warmup=50, repeats=200):
    """Measure forward-pass latency."""
    x = torch.randn(batch_size, *input_shape, device=device)

    # Warmup
    for _ in range(warmup):
        model.predict(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    return {
        'batch_size': batch_size,
        'total_ms': float(np.mean(times) * 1000),
        'total_std_ms': float(np.std(times) * 1000),
        'per_image_ms': float(np.mean(times) / batch_size * 1000),
        'throughput': float(batch_size / np.mean(times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure inference latency from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, algo_name, input_shape, num_classes, hparams = load_model(args.checkpoint, device)

    total_params, trainable_params = count_parameters(model)
    print(f"Algorithm: {algo_name}")
    print(f"Input shape: {input_shape}")
    print(f"Num classes: {num_classes}")
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M")

    print(f"\nMeasuring latency (warmup={args.warmup}, repeats={args.repeats})")
    print(f"{'Batch':>6} {'Total (ms)':>12} {'Per-image (ms)':>15} {'Throughput':>14}")
    print("-" * 52)

    for bs in args.batch_sizes:
        result = measure_latency(model, input_shape, bs, device, args.warmup, args.repeats)
        print(f"{bs:>6} {result['total_ms']:>9.2f} ± {result['total_std_ms']:.2f}"
              f" {result['per_image_ms']:>11.2f}"
              f" {result['throughput']:>10.1f} img/s")


if __name__ == "__main__":
    main()
