# mps-fallback-bench

Measure the **CPU fallback penalty** for PyTorch MPS (Apple Silicon) on a curated list of **most-requested ATen ops** from the official MPS tracking issues. Produce CSV + Markdown with:
- CPU vs **MPS+fallback** timings (median)
- **Penalty factor** and **overhead** per dtype/shape
- Links to the GitHub issue comments requesting the op
- A simple **fallback%** utilization metric

> Design: Use **issue comments** as the source of truth for *demand*, then validate **locally** whether an op still falls back (no third-party coverage tables).

## Requirements

- macOS 13+ (Apple Silicon)
- Xcode CLT installed (`xcode-select --install`)
- Python 3.10/3.11
- A local PyTorch install (source or nightly) with MPS enabled

## Quickstart

```bash
# 0) create venv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip

# 1) install deps
pip install -e .

# 2) (optional) pull fresh targets from GH issues (needs GH token)
export GH_TOKEN=ghp_xxx
python scripts/sync_requests.py --issues 77764 141287 --out ops/targets.yaml

# Or start from the seed list:
cp ops/targets.seed.yaml ops/targets.yaml

# 3) benchmark all curated ops (CPU vs MPS+fallback)
bash scripts/bench_all.sh

# 4) aggregate results → Markdown summary
python report/aggregate.py --results_dir results --out report/summary.md

# 5) (optional) compute a coarse MPS utilization metric on a tiny run
python metrics/mps_utilization.py
```

What you’ll get

results/*.csv per op with shape/dtype grid

report/summary.md with a ranked “Top Pain” table

A ops/targets.yaml file versioned with your Torch/macOS

Notes

“MPS+fallback” timing includes device↔host sync + CPU op + copies; it’s the user-visible cost.

This repo does not run full models; it runs single ATen ops with shapes/dtypes mentioned in issue comments.

Extend bench/op_wrappers.py to cover more ops; keep callables minimal and allocation-free inside the timed region.


---

## `pyproject.toml`

```toml
[project]
name = "mps-fallback-bench"
version = "0.1.0"
description = "Benchmark CPU fallback costs for PyTorch MPS ops"
requires-python = ">=3.10"
dependencies = [
  "torch>=2.2",         # assume local/nightly is already installed if needed
  "pandas>=2.0",
  "pyyaml>=6.0",
  "jinja2>=3.1",
  "requests>=2.31"
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", "matplotlib"]
```

