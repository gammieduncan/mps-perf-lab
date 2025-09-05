# MPS Fallback Bench — Summary

**Environment:** Torch unknown / macOS unknown

## Top Pain (by median penalty)
| op | rows | median× | max× | mean over (ms) |
|---|---:|---:|---:|---:|
| aten::index_select.default | 2 | 11.38 | 22.12 | 0.06 |
| aten::cummin.out | 4 | 2.65 | 3.29 | 0.05 |
| aten::cumsum.default | 6 | 1.03 | 1.83 | -0.01 |
| aten::conv3d.default | 4 | 0.02 | 0.05 | -72.44 |


## Raw rows
Total rows: 20