import yaml, argparse

def main(args):
    y = yaml.safe_load(open(args.targets))
    rows=[]
    for op in y["ops"]:
        if op.get("implemented_mps"): continue
        rows.append((op["score"], op["qualname"], op["voters"], op["last_year"]))
    rows.sort(reverse=True)
    for s,q,u,yr in rows[:args.top]:
        print(f"{s:5.2f}  {q:32s}  users={u:2d}  last={yr}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default="ops/targets.yaml")
    p.add_argument("--top", type=int, default=10)
    main(p.parse_args())

