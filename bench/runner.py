import os, json, argparse, time, warnings
import yaml
import torch
import pandas as pd
from torch.utils import benchmark as tb
from .op_wrappers import make_callable
from ops.shapesets import defaults_for

def time_callable(fn, min_run_time=1.0):
    t = tb.Timer(stmt="fn()", globals={"fn": fn})
    m = t.blocked_autorange(min_run_time=min_run_time)
    return float(m.median)

def dispatch_has_mps(qualname: str) -> bool:
    try:
        tab = torch._C._dispatch_dump_table(qualname)
        if "MPS" not in tab:
            return False
        # Heuristic: treat entries mentioning fallback / fallthrough / composite as not implemented
        mps_lines = [ln for ln in tab.splitlines() if "MPS" in ln]
        for ln in mps_lines:
            low = ln.lower()
            if any(k in low for k in ("fallback", "fallthrough", "composite")):
                continue
            return True
        return False
    except Exception:
        return False

def bench_qualname(qualname, shapes, dtypes):
    rows = []
    impl_mps = dispatch_has_mps(qualname)
    for shape in shapes:
        for dt in dtypes:
            status = "ok"
            err = None
            # CPU baseline
            try:
                fn_cpu = make_callable(qualname, shape, dt, "cpu")
                cpu_s = time_callable(fn_cpu)
            except Exception as e:
                status = "cpu_error"
                err = str(e)[:200]
                cpu_s = None

            # MPS with fallback enabled
            mps_s = None
            fallback_warn = None
            try:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                fn_mps = make_callable(qualname, shape, dt, "mps")
                # Probe once to detect fallback warning
                warnings.simplefilter("always")
                msgs = []
                def hook(msg, *a, **k):
                    s = str(msg)
                    if "will fall back to run on the CPU" in s:
                        msgs.append(s)
                old = warnings.showwarning
                warnings.showwarning = hook
                try:
                    fn_mps()
                finally:
                    warnings.showwarning = old
                fallback_warn = bool(msgs)
                torch.mps.synchronize()
                mps_s = time_callable(fn_mps)
                torch.mps.synchronize()
            except Exception as e:
                status = "mps_error" if status == "ok" else status
                err = str(e)[:200]

            rows.append({
                "qualname": qualname,
                "shape": str(shape),
                "dtype": dt,
                "time_cpu_s": cpu_s,
                "time_mps_fallback_s": mps_s,
                "penalty_factor": (mps_s / cpu_s if (cpu_s is not None and mps_s is not None) else None),
                "over_ms": ((mps_s - cpu_s) * 1e3 if (cpu_s is not None and mps_s is not None) else None),
                "implemented_mps": impl_mps,
                "fallback_warn": fallback_warn,
                "status": status,
                "error": err,
            })
    return pd.DataFrame(rows)

def load_targets(path):
    try:
        with open(path) as f:
            y = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise SystemExit(f"Failed to parse YAML at {path}: {e}")
    return y["ops"]

def main(args):
    assert torch.backends.mps.is_available(), "MPS not available."
    ops = load_targets(args.targets)
    os.makedirs(args.out_dir, exist_ok=True)
    for entry in ops:
        q = entry["qualname"]
        base = q.split("::")[1].split(".")[0]
        shapes = entry.get("shapes") or defaults_for(base)
        dtypes = entry.get("dtypes") or ["float16","float32"]
        df = bench_qualname(q, shapes, dtypes)
        out = f"{args.out_dir}/{q.replace('::','_').replace('.','_')}.csv"
        df.to_csv(out, index=False)
        print("wrote", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="ops/targets.yaml")
    p.add_argument("--out_dir", default="results")
    main(p.parse_args())
