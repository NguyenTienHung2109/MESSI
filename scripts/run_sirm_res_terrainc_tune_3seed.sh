#!/usr/bin/env bash
set -euo pipefail

config="configs/sirm_res_terrainc_tune.json"
screen_root="subset_irm_outputs/sirm_res_terrainc_tune/screen"
final_root="subset_irm_outputs/sirm_res_terrainc_tune/final"
report="subset_irm_outputs/sirm_res_terrainc_tune/tuning_results.json"

candidates=(
  S-IRM-res_tune_c00 S-IRM-res_tune_c01 S-IRM-res_tune_c02
  S-IRM-res_tune_c03 S-IRM-res_tune_c04 S-IRM-res_tune_c05
  S-IRM-res_tune_c06 S-IRM-res_tune_c07
)

mkdir -p "$screen_root" "$final_root" subset_irm_outputs/tmux_logs
for candidate in "${candidates[@]}"; do
  for target_env in 0 1 2 3; do
    for seed in 0 1 2; do
      output_dir="$screen_root/$candidate/env$target_env/seed$seed"
      log_file="subset_irm_outputs/tmux_logs/${candidate}_env${target_env}_seed${seed}.log"
      if [[ -f "$output_dir/summary.json" ]]; then
        continue
      fi
      python -u -m domainbed.scripts.train_subset_irm_pacs \
        --config "$config" --data-dir ./domainbed/data \
        --run "$candidate" --output-dir "$output_dir" \
        --target-env "$target_env" --seed "$seed" \
        --skip-target-eval > "$log_file" 2>&1
    done
  done
done

python scripts/summarize_sirm_res_tune.py \
  --config "$config" --root "$screen_root" --output "$report" \
  --require-complete

selected_config="subset_irm_outputs/sirm_res_terrainc_tune/selected_full_config.json"
for target_env in 0 1 2 3; do
  for seed in 0 1 2; do
    output_dir="$final_root/env$target_env/seed$seed"
    log_file="subset_irm_outputs/tmux_logs/S-IRM-res_tuned_env${target_env}_seed${seed}.log"
    if [[ -f "$output_dir/summary.json" ]]; then
      continue
    fi
    python -u -m domainbed.scripts.train_subset_irm_pacs \
      --config "$selected_config" --data-dir ./domainbed/data \
      --run S-IRM-res_tuned --output-dir "$output_dir" \
      --target-env "$target_env" --seed "$seed" > "$log_file" 2>&1
  done
done

python scripts/summarize_sirm_res_final.py \
  --root "$final_root" \
  --output subset_irm_outputs/sirm_res_terrainc_tune/final_results.json \
  --require-complete
