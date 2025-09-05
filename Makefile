PY=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .

sync:
	@GH_TOKEN=$$GH_TOKEN $(PY) scripts/sync_requests.py --issues 77764 141287 --out ops/targets.yaml

bench:
	bash scripts/bench_all.sh

report:
	$(PY) report/aggregate.py --results_dir results --out report/summary.md

coverage:
	mkdir -p results
	$(PY) scripts/scan_all_ops.py --out results/mps_coverage.csv

check-ops:
	mkdir -p results
	$(PY) scripts/check_ops.py --out results/check_ops.csv

clean:
	rm -rf results report/summary.md comments_*.json
