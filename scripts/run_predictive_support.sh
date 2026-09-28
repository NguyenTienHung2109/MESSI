#!/usr/bin/env bash
set -euo pipefail
# Override arguments after the output directory to select config, env, seed or intervention.
output_dir="${1:?Usage: bash scripts/run_predictive_support.sh OUTPUT_DIR [runner arguments]}"
shift
python -m domainbed.scripts.train_predictive_support --output-dir "$output_dir" "$@"
