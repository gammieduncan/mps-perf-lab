import torch

# Minimal “op callables”: closed-over inputs, no allocations inside.
# Add more wrappers as you curate ops.

def make_cumsum(shape, dtype, device):
    x = torch.randn(*shape, device=device, dtype=getattr(torch, dtype))
    return lambda: torch.ops.aten.cumsum.default(x, -1)

def make_index_select(shape, dtype, device):
    x = torch.randn(*shape, device=device, dtype=getattr(torch, dtype))
    idx = torch.arange(x.size(-1)//2, device=device, dtype=torch.int64)
    return lambda: torch.ops.aten.index_select.default(x, -1, idx)

def make_softmax(shape, dtype, device):
    x = torch.randn(*shape, device=device, dtype=getattr(torch, dtype))
    return lambda: torch.ops.aten._softmax.default(x, -1, False)

def make_layer_norm(shape, dtype, device):
    x = torch.randn(*shape, device=device, dtype=getattr(torch, dtype))
    norm = (shape[-1],)
    w = torch.ones(*norm, device=device, dtype=getattr(torch, dtype))
    b = torch.zeros(*norm, device=device, dtype=getattr(torch, dtype))
    return lambda: torch.ops.aten.layer_norm.default(x, norm, w, b, 1e-5, False)

def make_linalg_eigh_eigenvalues(shape, dtype, device):
    # Ensure square (or batched square) and Hermitian. Use public API which maps to ATen.
    alloc_dtype = dtype
    if dtype in {"float16", "bfloat16"}:
        alloc_dtype = "float32"
    x = torch.randn(*shape, device=device, dtype=getattr(torch, alloc_dtype))
    if x.dim() < 2 or x.size(-1) != x.size(-2):
        n = shape[-1]
        x = torch.randn(n, n, device=device, dtype=getattr(torch, alloc_dtype))
    a = (x + x.transpose(-1, -2)) * 0.5
    return lambda: torch.linalg.eigvalsh(a, UPLO='L')

def make_cummin_out(shape, dtype, device):
    x = torch.randn(*shape, device=device, dtype=getattr(torch, dtype))
    dim = -1
    values = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int64)
    return lambda: torch.ops.aten.cummin.out(x, dim, values=values, indices=indices)

def make_conv3d(shape, dtype, device):
    # Expect shape [N, C, D, H, W]
    if len(shape) != 5:
        raise ValueError("conv3d expects shape [N, C, D, H, W]")
    N, C, D, H, W = shape
    import torch.nn as nn
    m = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=3, padding=1, bias=False)
    m = m.to(device=device, dtype=getattr(torch, dtype))
    x = torch.randn(N, C, D, H, W, device=device, dtype=getattr(torch, dtype))
    m.eval()
    return lambda: m(x)

FACTORY = {
    "aten::cumsum.default":      make_cumsum,
    "aten::index_select.default":make_index_select,
    "aten::_softmax.default":    make_softmax,
    "aten::layer_norm.default":  make_layer_norm,
    "aten::_linalg_eigh.eigenvalues": make_linalg_eigh_eigenvalues,
    "aten::cummin.out":          make_cummin_out,
    "aten::conv3d.default":      make_conv3d,
    # Friendly aliases for module form
    "nn.Conv3d":                  make_conv3d,
    "nn.Conv3D":                  make_conv3d,
}

def make_callable(qualname: str, shape, dtype: str, device: str):
    if qualname not in FACTORY:
        raise KeyError(f"No wrapper for {qualname}. Add to FACTORY in op_wrappers.py")
    return FACTORY[qualname](shape, dtype, device)
