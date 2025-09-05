import argparse, glob, pandas as pd
from jinja2 import Template

def aggregate(results_dir):
    frames = [pd.read_csv(p) for p in glob.glob(f"{results_dir}/*.csv")]
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["shape"] = df["shape"].astype(str)
    return df

def summarize(df):
    df_ok = df
    if "status" in df.columns:
        df_ok = df[df["status"].fillna("ok") == "ok"]
    g = df_ok.groupby("qualname").agg(
        rows=("qualname","count"),
        max_penalty=("penalty_factor","max"),
        median_penalty=("penalty_factor","median"),
        mean_over_ms=("over_ms","mean")
    ).reset_index().sort_values("median_penalty", ascending=False)
    return g

TEMPLATE = """# MPS Fallback Bench — Summary

**Environment:** {{ env }}

## Top Pain (by median penalty)
| op | rows | median× | max× | mean over (ms) |
|---|---:|---:|---:|---:|
{% for _,r in top.iterrows() -%}
| {{r.qualname}} | {{r.rows}} | {{'%.2f'%r.median_penalty}} | {{'%.2f'%r.max_penalty}} | {{'%.2f'%r.mean_over_ms}} |
{% endfor %}

## Raw rows
Total rows: {{ rows }}
"""

def main(args):
    df = aggregate(args.results_dir)
    env = "Torch unknown / macOS unknown"
    if df.empty:
        open(args.out, "w").write("# No results")
        return
    top = summarize(df).head(30)
    md = Template(TEMPLATE).render(env=env, top=top, rows=len(df))
    with open(args.out, "w") as f: f.write(md)
    print("wrote", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out", default="report/summary.md")
    main(p.parse_args())
