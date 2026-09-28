# PACS leave-one-domain-out — seed 0

Four runs, 5,000 steps each, pretrained DeiT-Small. Target metrics are scored only after source-based checkpoint selection.

| Target | Progress | Source validation (latest) | Target accuracy (selected checkpoint) | EDA |
|---|---:|---:|---:|---|
| art_painting | 5000/5000 | 0.9831 | 0.9033 | [art_painting](env0_seed0/eda/report.md) |
| cartoon | 5000/5000 | 0.9714 | 0.8537 | [cartoon](env1_seed0/eda/report.md) |
| photo | 5000/5000 | 0.9587 | 0.9904 | [photo](env2_seed0/eda/report.md) |
| sketch | 5000/5000 | 0.9793 | 0.7595 | [sketch](env3_seed0/eda/report.md) |

Reports refresh at source-validation checkpoints. Raw logs and queue status are in this directory. A single seed is an initial experiment; it does not establish variance across seeds.
