import argparse, csv, os, warnings
import torch
import torch.nn.functional as F


def dispatch_has_mps(qualname: str) -> bool:
    try:
        tab = torch._C._dispatch_dump_table(qualname)
        if "MPS" not in tab:
            return False
        mps_lines = [ln for ln in tab.splitlines() if "MPS" in ln]
        for ln in mps_lines:
            low = ln.lower()
            if any(k in low for k in ("fallback", "fallthrough", "composite")):
                continue
            return True
        return False
    except Exception:
        return False


def iter_aten_qualnames():
    pkt = torch.ops.aten
    for name in dir(pkt):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(pkt, name)
        except Exception:
            continue
        overloads = getattr(attr, "overloads", None)
        if not callable(overloads):
            continue
        for ol in overloads():
            yield f"aten::{name}.{ol}"


def make_family_probe(base: str, dtype: str = "float32"):
    device = "mps"
    dt = getattr(torch, dtype)

    if base in {"linalg_qr", "_linalg_qr"}:
        a = torch.randn(64, 32, device=device, dtype=dt)
        return lambda: torch.linalg.qr(a, mode="reduced")

    if base in {"_linalg_eigh", "linalg_eigh"}:
        x = torch.randn(128, 128, device=device, dtype=dt)
        a = (x + x.t()) * 0.5
        return lambda: torch.linalg.eigvalsh(a, UPLO="L")

    if base in {"unique_dim", "unique"}:
        x = torch.randint(0, 32, (32, 32), device=device)
        return lambda: torch.unique(x, dim=1)

    if base in {"grid_sampler_2d_backward", "grid_sampler_2d"}:
        x = torch.randn(1, 3, 16, 16, device=device, requires_grad=True)
        grid = torch.rand(1, 8, 8, 2, device=device) * 2 - 1
        def f():
            y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            y.sum().backward()
        return f

    return None


def probe_run(fn, fallback: bool | None):
    if fallback is not None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" if fallback else "0"
    warnings.simplefilter("always")
    msgs = []
    def hook(msg, *a, **k):
        s = str(msg)
        if "will fall back to run on the CPU" in s:
            msgs.append(s)
    old = warnings.showwarning
    warnings.showwarning = hook
    try:
        fn()
        return True, bool(msgs), None
    except Exception as e:
        return False, (bool(msgs) if msgs else None), str(e)
    finally:
        warnings.showwarning = old


def scan(out_csv: str, only_missing: bool = False):
    rows = []
    total = 0
    missing = 0
    for q in sorted(iter_aten_qualnames()):
        total += 1
        impl = dispatch_has_mps(q)
        if not impl:
            missing += 1
        if only_missing and impl:
            continue
        ran_no_fb = None
        ran_with_fb = None
        fb_warn = None
        err = None
        if torch.backends.mps.is_available():
            base = q.split("::")[-1].split(".")[0]
            fn = make_family_probe(base)
            if fn is not None:
                ok0, warn0, err0 = probe_run(fn, fallback=False)
                ran_no_fb = ok0
                if not ok0:
                    ok1, warn1, err1 = probe_run(fn, fallback=True)
                    ran_with_fb = ok1
                    fb_warn = warn1
                    err = err1 or err0
        rows.append({
            "qualname": q,
            "implemented_mps": impl,
            "ran_no_fallback": ran_no_fb,
            "ran_with_fallback": ran_with_fb,
            "fallback_warn": fb_warn,
            "error": err,
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "qualname",
                "implemented_mps",
                "ran_no_fallback",
                "ran_with_fallback",
                "fallback_warn",
                "error",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print({
        "total_ops": total,
        "implemented_mps": total - missing,
        "missing_mps": missing,
        "csv": out_csv,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/mps_coverage.csv")
    ap.add_argument("--only_missing", action="store_true")
    args = ap.parse_args()
    scan(args.out, args.only_missing)
