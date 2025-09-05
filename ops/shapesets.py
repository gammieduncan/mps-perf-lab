# Canonical shapes per op family when issue comments don’t specify.
# Feel free to extend as you go.

def defaults_for(base: str):
    # base = "cumsum", "index_select", ...
    if base in {"cumsum", "sum", "amax", "amin"}:
        return [[64, 1024], [8192]]
    if base in {"index_select", "gather", "scatter"}:
        return [[32, 1024], [16, 2048]]
    if base in {"_softmax", "softmax"}:
        return [[8, 8, 128, 128], [4, 16, 256, 256]]
    if base in {"layer_norm"}:
        return [[64, 512], [8, 2048]]
    if base in {"topk"}:
        return [[64, 1024]]
    if base in {"conv3d"}:
        # N, C, D, H, W
        return [[1, 16, 16, 64, 64], [1, 32, 8, 32, 32]]
    if base in {"_linalg_eigh", "linalg_eigh"}:
        # square (optionally batched) Hermitian
        return [[512, 512], [4, 128, 128]]
    if base in {"cummin"}:
        return [[64, 1024], [8192]]
    return [[1024]]
