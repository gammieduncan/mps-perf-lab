import os, json
import torch
from torch.profiler import profile, ProfilerActivity

def utilization(run_fn):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        run_fn()
    aten = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    total = len(aten)
    # crude: if fallback happened, PyTorch prints warnings; better to combine with warnings hook during run_fn
    return {"total_aten_calls": total}

if __name__ == "__main__":
    def demo():
        x = torch.randn(3,3, device="mps")
        _ = (x + 1).relu()
    print(json.dumps(utilization(demo), indent=2))

