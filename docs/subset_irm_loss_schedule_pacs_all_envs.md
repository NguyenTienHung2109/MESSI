# Subset-IRM loss-schedule PACS results

Configuration: `B6LossSchedule`, seed 0, full pretrained distilled DeiT-Small,
six experts, per-sample Top-2 routing, and checkpoints every 500 updates.
Each row holds out one PACS environment and trains only on the other three.

## Training-domain validation selection

The checkpoint maximizes mean `out_acc` over the three training environments.
The held-out environment is not used for selection. DomainBed reports held-out
`in_acc` as test accuracy; held-out `out_acc` is included for transparency.

| Test env | Step | Source val | Test in | Test out |
|---|---:|---:|---:|---:|
| 0 Art | 2500 | 98.36 | 90.30 | 93.15 |
| 1 Cartoon | 1000 | 97.83 | 85.39 | 86.97 |
| 2 Photo | 3000 | 97.71 | 98.88 | 99.40 |
| 3 Sketch | 1500 | 98.75 | 76.18 | 76.31 |
| Mean | — | 98.16 | 87.69 | 88.96 |

## DomainBed oracle selection

`OracleSelectionMethod` in this repository selects the last checkpoint rather
than early-stopping on the held-out domain. `test_out` is its validation score
and `test_in` is the reported test accuracy.

| Test env | Step | Oracle val/test out | Oracle test in |
|---|---:|---:|---:|
| 0 Art | 5000 | 92.67 | 90.05 |
| 1 Cartoon | 5000 | 84.62 | 83.85 |
| 2 Photo | 5000 | 99.10 | 98.65 |
| 3 Sketch | 5000 | 79.11 | 78.31 |
| Mean | — | 88.87 | 87.72 |

## Held-out-out peak diagnostic

This is not a valid DG selection result. It selects the checkpoint maximizing
the held-out environment's `out_acc` and is provided only to quantify the
checkpoint-selection gap.

| Test env | Step | Peak test out | Corresponding test in |
|---|---:|---:|---:|
| 0 Art | 2500 | 93.15 | 90.30 |
| 1 Cartoon | 3500 | 88.03 | 85.13 |
| 2 Photo | 1500 | 99.70 | 99.03 |
| 3 Sketch | 4500 | 81.78 | 80.60 |
| Mean | — | 90.67 | 88.76 |

## Expert utilization at the validation-selected checkpoints

| Test env | Active experts/sample | Experts with zero Top-2 load | Nonzero expert loads |
|---|---:|---:|---|
| 0 Art | 2 | 3 | E2=31.45%, E3=18.01%, E5=50.54% |
| 1 Cartoon | 2 | 3 | E1=53.81%, E2=32.96%, E3=13.23% |
| 2 Photo | 2 | 4 | E0=49.46%, E1=50.54% |
| 3 Sketch | 2 | 4 | E2=55.88%, E5=44.12% |

The expert count alone does not predict test performance: Photo uses only two
experts and reaches 98.88% test-in accuracy, whereas Sketch also uses two and
reaches 76.18%. The stronger issue for Sketch is checkpoint-selection mismatch:
its held-out peak occurs at step 4500, while source validation selects step 1500.

Artifacts:

- Env 0: `subset_irm_outputs/full/B6LossSchedule_env0`
- Env 1: `subset_irm_outputs/full/B6LossSchedule.eval500_exploratory`
- Env 2: `subset_irm_outputs/full/B6LossSchedule_env2`
- Env 3: `subset_irm_outputs/full/B6LossSchedule_env3`
