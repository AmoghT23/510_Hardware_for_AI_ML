# ResNet18 Layer Analysis

## Top 5 Layers by MAC Count

| Rank | Layer Name | MACs | FLOPs (2×MACs) | Parameters |
| ---- | ---------- | ---- | -------------- | ---------- |
| 1 | `Conv2d (conv1): 1-1` | 118,013,952 | 236,027,904 | 9,408 |
| 2 | `Conv2d (conv1): 3-1` | 115,605,504 | 231,211,008 | 36,864 |
| 3 | `Conv2d (conv2): 3-4` | 115,605,504 | 231,211,008 | 36,864 |
| 4 | `Conv2d (conv1): 3-7` | 115,605,504 | 231,211,008 | 36,864 |
| 5 | `Conv2d (conv2): 3-10` | 115,605,504 | 231,211,008 | 36,864 |

---

## Arithmetic Intensity — Most MAC-Intensive Layer

**Layer:** `Conv2d (conv1): 1-1`

### Assumptions
- All weights and activations are loaded from DRAM with **no reuse**.
- Data type: **float32** (4 bytes per element).

### Memory Traffic

| Tensor | Shape | Elements | Bytes (×4) |
| ------ | ----- | -------- | ---------- |
| Input activations  | `[1, 3, 224, 224]`  | 150,528  | 602,112  |
| Output activations | `[1, 64, 112, 112]` | 802,816 | 3,211,264 |
| Weights            | —                         | 9,408 | 37,632 |
| **Total**          |                           |                  | **3,851,008** |

### Calculation

```
FLOPs  = 2 × MACs = 2 × 118,013,952 = 236,027,904

DRAM bytes = input + output + weights
           = 602,112 + 3,211,264 + 37,632
           = 3,851,008 bytes

Arithmetic Intensity = FLOPs / DRAM bytes
                     = 236,027,904 / 3,851,008
                     = 61.29 FLOPs/byte
```
