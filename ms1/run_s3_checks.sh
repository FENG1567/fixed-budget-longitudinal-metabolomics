#!/usr/bin/env bash
# Run the S3 raw-signal checks against existing bin vectors and sample maps.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  cat >&2 <<'EOF'
Usage: run_s3_checks.sh <bins-root> <sample-maps-root> <output-dir>
Expected paths:
  <bins-root>/894/*.npy and <bins-root>/895/*.npy
  <sample-maps-root>/ST003894_samples.csv
  <sample-maps-root>/ST003895_samples.csv
EOF
  exit 2
fi

bins_root=$1
maps_root=$2
output_dir=$3
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}
mkdir -p "$output_dir"

"$python_bin" "$script_dir/49_raw_summary.py" 894 \
  --bins "$bins_root/894" \
  --samples "$maps_root/ST003894_samples.csv" \
  --output "$output_dir/raw_summary_894.json"
"$python_bin" "$script_dir/49_raw_summary.py" 895 \
  --bins "$bins_root/895" \
  --samples "$maps_root/ST003895_samples.csv" \
  --output "$output_dir/raw_summary_895.json"
"$python_bin" "$script_dir/45_rho_norm.py" 894 \
  --bins "$bins_root/894" \
  --samples "$maps_root/ST003894_samples.csv" | tee "$output_dir/normalization_894.txt"
"$python_bin" "$script_dir/51_targeted.py" 894 \
  --bins "$bins_root/894" \
  --samples "$maps_root/ST003894_samples.csv" | tee "$output_dir/targeted_windows_894.txt"
