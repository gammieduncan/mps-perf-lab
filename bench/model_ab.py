import importlib, time, sys, torch

def run(model_qualname, device, mode="eval"):
    mod = importlib.import_module(model_qualname)
    bench = getattr(mod, "Model")(test=mode, device=device, batch_size=1)
    model, example_inputs = bench.get_module()
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(*example_inputs)
    return (time.time()-t0)/10.0

if __name__ == "__main__":
    m = sys.argv[1]
    cpu = run(m, "cpu")
    mps = run(m, "mps")
    print({"cpu_s": cpu, "mps_s": mps, "speedup": cpu/mps})

