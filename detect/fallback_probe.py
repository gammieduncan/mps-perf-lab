import os, warnings, re

def falls_back(op_callable) -> bool:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.simplefilter("always")
    msgs = []
    def hook(msg, *a, **k):
        s = str(msg)
        if "will fall back to run on the CPU" in s: msgs.append(s)
    warnings.showwarning = hook
    op_callable()
    return bool(msgs)

