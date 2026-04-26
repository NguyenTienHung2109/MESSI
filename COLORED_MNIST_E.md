# ColoredMNIST_E

A variant of `ColoredMNIST` (Arjovsky et al., 2019) where the number of training
environments **E** is a tunable hyperparameter rather than hardcoded to 3.

The protocol follows Wang et al., *"Lost Domain Generalization Is a Natural
Consequence of Lack of Training Domains"*, AAAI 2024
([paper](https://ojs.aaai.org/index.php/AAAI/article/view/29497)). The paper
shows that ERM's domain-generalization performance increases monotonically with
E and approaches the noise ceiling (~75% test accuracy) when E is large.

## Per-image generation

For each MNIST image with digit `d` in environment with parameter `p_e`:

1. `y' = 0 if d < 5 else 1`  *(binary label from digit)*
2. `y = y' XOR Bernoulli(0.25)`  *(label noise; same 25% in every env)*
3. `z = y XOR Bernoulli(p_e)`  *(color id; flipped from label with prob `p_e`)*
4. Render as a 2-channel `(2, 28, 28)` image: keep channel `z`, zero the other.

## Environment construction

For a chosen `E ≥ 2`:
- Take `E+1` evenly spaced candidates in `(0, 1)`:
  `np.linspace(1/(E+2), (E+1)/(E+2), E+1)`.
- Drop the candidate nearest `0.5` → `E` training values.
- Append `0.5` as the test environment (color uncorrelated with label).

The test environment is **always the last index**, so always pass
`--test_envs E` (matching the value of `num_environments`).

MNIST's 70 000 images are sharded equally across the `E+1` environments by the
existing `MultipleEnvironmentMNIST` base class — no changes there.

## Running

E = 2 (matches default `ColoredMNIST` baseline; ERM should learn the color
shortcut and score near chance on the test env):

```bash
python -m domainbed.scripts.train \
    --data_dir=./domainbed/data/MNIST/ \
    --algorithm ERM \
    --dataset ColoredMNIST_E \
    --test_envs 2 \
    --hparams '{"num_environments": 2}' \
    --output_dir ./train_output/cmnist_e2
```

E = 8 (default):

```bash
python -m domainbed.scripts.train \
    --data_dir=./domainbed/data/MNIST/ \
    --algorithm ERM \
    --dataset ColoredMNIST_E \
    --test_envs 8 \
    --hparams '{"num_environments": 8}' \
    --output_dir ./train_output/cmnist_e8
```

E = 32 (test acc should climb toward the 75% noise ceiling):

```bash
python -m domainbed.scripts.train \
    --data_dir=./domainbed/data/MNIST/ \
    --algorithm ERM \
    --dataset ColoredMNIST_E \
    --test_envs 32 \
    --hparams '{"num_environments": 32}' \
    --output_dir ./train_output/cmnist_e32
```

## Smoke test

```bash
python -m domainbed.scripts.test_colored_mnist_e \
    --data_dir ./domainbed/data/MNIST \
    --out_dir /tmp/cmnist_e_viz
```

Prints the `p_e` values of all `E+1` environments, asserts shape/label/channel
invariants, checks determinism under `torch.manual_seed(0)`, and saves a 4×4
RGB grid (red/green/blue=0) per environment to the output directory.
