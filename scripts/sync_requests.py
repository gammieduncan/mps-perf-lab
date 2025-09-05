"""
Pull comments from PyTorch MPS tracking issues, extract valid ATen overload mentions,
score by unique commenters + thumbs (+ recency), and emit ops/targets.yaml.

We *do not* trust third-party coverage lists. We validate locally:
- If Dispatch table contains MPS kernel → implemented_mps=True
- Else, probe a tiny callable on MPS with fallback enabled → confirmed_fallback=True
"""
import os, re, json, argparse, requests, yaml, time
from collections import defaultdict
from datetime import datetime
import torch

GH = "https://api.github.com"

def gh_get(url, token=None):
    hdr = {"Accept": "application/vnd.github+json"}
    if token: hdr["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=hdr)
    r.raise_for_status()
    return r

def fetch_comments(owner, repo, issue, token=None):
    url = f"{GH}/repos/{owner}/{repo}/issues/{issue}/comments"
    out = []
    while url:
        r = gh_get(url, token)
        out += r.json()
        url = r.links.get("next", {}).get("url")
    return out

def normalize_aten(op: str, suffix: str|None):
    base = op.strip()
    if not hasattr(torch.ops.aten, base): return None
    overloads = getattr(torch.ops.aten, base).overloads()
    name = suffix[1:] if suffix else "default"
    if name not in overloads: return None
    return f"aten::{base}.{name}"

MENTION_RE = re.compile(
    r"(?:aten::|torch\.ops\.aten\.|torch\.|request:\s*)"
    r"(?P<op>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?P<suffix>(?:\.[a-zA-Z0-9_]+)*)"
)

def recency_weight(year: int) -> float:
    now = datetime.utcnow().year
    age = max(0, now - year)
    return {0:1.0, 1:0.6}.get(age, 0.3)

def dispatch_has_mps(qualname: str) -> bool:
    try:
        tab = torch._C._dispatch_dump_table(qualname)
        return "MPS" in tab
    except Exception:
        return False

def make_probe_callable(qualname: str):
    base = qualname.split("::")[1].split(".")[0]
    dev = "mps"
    if base == "cumsum":
        import bench.op_wrappers as W
        return W.make_cumsum([64,1024], "float16", dev)
    if base == "index_select":
        import bench.op_wrappers as W
        return W.make_index_select([32,1024], "float16", dev)
    if base in {"_softmax","softmax"}:
        import bench.op_wrappers as W
        return W.make_softmax([8,8,128,128], "float16", dev)
    if base == "layer_norm":
        import bench.op_wrappers as W
        return W.make_layer_norm([64,512], "float16", dev)
    if base == "conv3d":
        import bench.op_wrappers as W
        return W.make_conv3d([1,16,16,32,32], "float16", dev)
    if base == "cummin":
        import bench.op_wrappers as W
        return W.make_cummin_out([64,1024], "float16", dev)
    if base in {"_linalg_eigh","linalg_eigh"}:
        import bench.op_wrappers as W
        return W.make_linalg_eigh_eigenvalues([256,256], "float32", dev)
    return None

def falls_back(qualname: str) -> bool:
    import warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.simplefilter("always")
    msgs=[]
    def hook(msg,*a,**k):
        s=str(msg)
        if "will fall back to run on the CPU" in s: msgs.append(s)
    warnings.showwarning=hook
    fn = make_probe_callable(qualname)
    if fn is None: return False
    try:
        fn()
    except Exception:
        return True
    return any(qualname in m for m in msgs)

def main(args):
    token = os.environ.get("GH_TOKEN")
    all_comments=[]
    for i in args.issues:
        all_comments += fetch_comments("pytorch","pytorch", i, token)
    # op → user set, thumbs sum, last_year, sample link
    votes = defaultdict(lambda: {"users":set(), "thumbs":0, "years":[], "link":None})
    for c in all_comments:
        body = c.get("body","") or ""
        user = (c.get("user") or {}).get("login","unknown")
        thumbs = (c.get("reactions") or {}).get("+1",0)
        created = c.get("created_at") or "1970-01-01T00:00:00Z"
        year = int(created[:4])
        link = c.get("html_url")
        for m in MENTION_RE.finditer(body):
            qual = normalize_aten(m.group("op"), m.group("suffix"))
            if not qual: continue
            v = votes[qual]
            v["users"].add(user)
            v["thumbs"] += thumbs
            v["years"].append(year)
            v["link"] = v["link"] or link

    items=[]
    for qual, v in votes.items():
        uniq_users = len(v["users"])
        if uniq_users == 0: continue
        last_year = max(v["years"]) if v["years"] else 1970
        score = uniq_users*1.0 + v["thumbs"]*0.5 + recency_weight(last_year)
        items.append((qual, score, uniq_users, v["thumbs"], last_year, v["link"]))

    # basic threshold
    items = [it for it in items if it[1] >= args.min_score]

    # environment stamp
    stamp = {
        "torch": torch.__version__,
        "commit": getattr(torch.version, "git_version", "unknown"),
        "macos": os.popen("sw_vers -productVersion").read().strip(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    }

    ops=[]
    for qual, score, users, thumbs, last_year, link in sorted(items, key=lambda x:-x[1]):
        impl = dispatch_has_mps(qual)
        fb = False if impl else falls_back(qual)
        base = qual.split("::")[1].split(".")[0]
        from ops.shapesets import defaults_for
        shapes = defaults_for(base)
        ops.append({
            "qualname": qual,
            "score": round(score,2),
            "voters": users,
            "thumbs": thumbs,
            "last_year": last_year,
            "implemented_mps": impl,
            "confirmed_fallback": fb,
            "shapes": shapes,
            "dtypes": ["float16","float32"],
            "issues": [link] if link else []
        })

    out={"version": stamp, "ops": ops}
    os.makedirs("ops", exist_ok=True)
    with open(args.out, "w") as f: yaml.safe_dump(out, f, sort_keys=False)
    print("wrote", args.out, f"({len(ops)} ops)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--issues", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="ops/targets.yaml")
    ap.add_argument("--min_score", type=float, default=3.0)
    main(ap.parse_args())
