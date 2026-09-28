"""Smoke test + visualisation for the ColoredMNIST_E dataset.

Usage:
    python -m domainbed.scripts.test_colored_mnist_e \
        --data_dir ./domainbed/data/MNIST \
        --out_dir /tmp/cmnist_e_viz
"""
import argparse
import os

import torch
from torchvision.utils import save_image

from domainbed import datasets, hparams_registry


def build(data_dir, seed=0, hparams_override=None):
    torch.manual_seed(seed)
    hparams = hparams_registry.default_hparams('ERM', 'ColoredMNIST_E')
    if hparams_override:
        hparams.update(hparams_override)
    return datasets.get_dataset_class('ColoredMNIST_E')(data_dir, [], hparams)


def parse_p(env_name):
    return float(env_name.split('=')[1])


def to_rgb(x):
    # x: (N, 2, 28, 28) → (N, 3, 28, 28) by appending an empty blue channel.
    blue = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
    return torch.cat([x[:, :1], x[:, 1:2], blue], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', default='/tmp/cmnist_e_viz')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = build(args.data_dir)
    E_plus_1 = len(ds)
    print(f"num envs (incl. test) = {E_plus_1}")
    for i, name in enumerate(ds.ENVIRONMENTS):
        n = len(ds[i])
        print(f"  env {i}: {name}  size={n}")

    # ---- assertions ----
    assert E_plus_1 == 9, f"expected 9 envs at default E=8, got {E_plus_1}"
    assert abs(parse_p(ds.ENVIRONMENTS[-1]) - 0.5) < 1e-9, \
        "test env must have p=0.5"

    train_p = sorted(parse_p(n) for n in ds.ENVIRONMENTS[:-1])
    expected = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    for got, exp in zip(train_p, expected):
        assert abs(got - exp) < 1e-9, f"got {train_p} != {expected}"

    for i in range(E_plus_1):
        x_all = ds[i].tensors[0]
        y_all = ds[i].tensors[1]
        assert x_all.shape[1:] == (2, 28, 28)
        assert x_all.dtype == torch.float32
        assert x_all.min() >= 0 and x_all.max() <= 1
        assert set(y_all.unique().tolist()) <= {0, 1}
        # Exactly one of the two channels must be all-zero per sample.
        ch0_zero = (x_all[:, 0].abs().sum(dim=(1, 2)) == 0)
        ch1_zero = (x_all[:, 1].abs().sum(dim=(1, 2)) == 0)
        assert torch.logical_xor(ch0_zero, ch1_zero).all(), \
            f"env {i}: each sample must have exactly one zeroed channel"

    # ---- save grids ----
    for i in range(E_plus_1):
        x = ds[i].tensors[0][:16]
        save_image(to_rgb(x), os.path.join(args.out_dir, f'env_{i}.png'),
                   nrow=4, padding=2)
    print(f"saved 16-image grids to {args.out_dir}/env_*.png")

    # ---- determinism ----
    ds2 = build(args.data_dir)
    a = ds[0].tensors[0]
    b = ds2[0].tensors[0]
    assert torch.equal(a, b), "rebuild with same seed should be byte-identical"
    print("determinism check OK")

    print("ALL CHECKS PASSED")


if __name__ == '__main__':
    main()
