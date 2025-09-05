#!/usr/bin/env bash
set -euo pipefail
OUT=results
mkdir -p "$OUT"
python -m bench.runner --targets ops/targets.yaml --out_dir "$OUT"

