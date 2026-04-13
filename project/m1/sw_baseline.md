# Software Baseline Benchmark
**ECE 410/510 — M1 Deliverable**
**Project:** Anemia Detection — HybridModel (ResNet18 + AttentionFusion)

---

## 1. Platform and Configuration

| Parameter | Value |
|---|---|
| **CPU** | Intel Core i5-10210U @ 1.60 GHz (boost 4.2 GHz), 4 cores / 8 threads |
| **GPU** | None — CPU-only execution (CUDA not available) |
| **RAM** | 17 GB DDR4-2667, dual-channel |
| **OS** | Windows 11 Home (Build 10.0.26200) |
| **Python** | 3.11.9 |
| **PyTorch** | 2.11.0 (CPU build, MKL backend) |
| **torchvision** | 0.26.0 |
| **scikit-learn** | 1.8.0 |
| **scikit-image** | 0.26.0 |
| **Model** | HybridModel — ResNet18 backbone (pretrained ImageNet) + AttentionFusion MLP |
| **Dataset** | AneRBC — 12,000 peripheral blood smear images (6,000 healthy / 6,000 anemic) |
| **Train split** | 9,600 images (80%) |
| **Val split** | 2,400 images (20%) |
| **Batch size** | 32 |
| **Epochs per run** | 1 |
| **Batches per epoch** | 300 (9,600 / 32) |
| **DataLoader workers** | 0 (single-process, Windows compatible) |
| **Execution mode** | `jupyter nbconvert --execute` (non-interactive batch via `run_benchmark.py`) |
| **Random seed** | 42 (train/val split via `sklearn.train_test_split`) |

> All information above is sufficient to reproduce the M4 speedup comparison.
> Re-run `python run_benchmark.py` in the project root with the venv activated
> to regenerate all 10 timing measurements.

---

## 2. Execution Time — Wall-Clock, 10 Runs

Measured with `time.perf_counter()` wrapping the full `nbconvert --execute` call
per run, tracking the complete pipeline: data loading, handcrafted feature
extraction, CNN baseline training, hybrid model training, evaluation, GradCAM,
and result logging.

| Run | Start Time | Wall-clock (s) | Wall-clock (min) |
|-----|------------|----------------|-----------------|
| 1 | 14:21:36 | 2762.6 | 46.0 |
| 2 | 15:07:39 | 2670.4 | 44.5 |
| 3 | 15:52:10 | 2595.1 | 43.3 |
| 4 | 16:35:25 | 2537.6 | 42.3 |
| 5 | 17:17:43 | 2552.6 | 42.5 |
| 6 | 18:00:16 | 2497.8 | 41.6 |
| 7 | 18:41:55 | 2508.4 | 41.8 |
| 8 | 19:23:43 | 2481.1 | 41.4 |
| 9 | 20:05:04 | 2860.9 | 47.7 |
| 10 | 20:52:45 | 2719.2 | 45.3 |

| Statistic | Value |
|---|---|
| **Median (≥10 runs)** | **2573.9 s (42.9 min)** |
| Mean | 2618.6 s (43.6 min) |
| Std deviation | 128.8 s |
| Min | 2481.1 s — Run 8 |
| Max | 2860.9 s — Run 9 |
| Successful runs | 10 / 10 |

> Run 1 is slower (warm-up: ResNet18 ImageNet weights downloaded and cached).
> Runs 2–10 reflect steady-state performance. Median across all 10 runs is the
> reported baseline value.

---

## 3. Throughput

### Full-pipeline throughput (end-to-end)

```
Throughput = Training samples per run / Median wall-clock
           = 9,600 samples / 2,573.9 s
           = 3.73 samples/sec
```

### Training-only throughput (from tqdm batch timing, cProfile run)

Measured from tqdm output in executed notebooks and confirmed by cProfile
(1266.6 s for 300 batches × 32 samples in the profiling pass):

```
Throughput (training only) = (300 batches × 32 samples) / 1266.6 s
                           = 9,600 / 1266.6
                           = 7.58 samples/sec
```

### FLOPs/sec throughput (dominant kernel)

From cProfile: 300 batches in 2.442 s/batch average (run_backward cumtime).
FLOPs per batch (Conv2d fwd+bwd) = 348.4 GFLOP (from `ai_calculation.md`):

```
Measured throughput = 348.4 GFLOP / 2.442 s
                    = 142.7 GFLOP/s
```

| Throughput metric | Value |
|---|---|
| Full-pipeline (samples/sec) | 3.73 samples/sec |
| Training-only (samples/sec) | 7.58 samples/sec |
| Conv2d kernel (GFLOP/s) | 142.7 GFLOP/s |
| CPU peak theoretical (GFLOP/s) | 268.8 GFLOP/s (Intel ARK, AVX2 boost) |
| Utilization of peak | ~53% |

---

## 4. Memory Usage — Peak RSS

Measured per run by `psutil` tracking the full process tree (nbconvert launcher
+ Jupyter kernel child process), polled every 0.5 seconds throughout each run.

| Run | Peak RSS (MB) |
|-----|--------------|
| 1 | 2049 |
| 2 | 2151 |
| 3 | 2186 |
| 4 | 2191 |
| 5 | 2157 |
| 6 | 2156 |
| 7 | 2169 |
| 8 | 2162 |
| 9 | 2169 |
| 10 | 2178 |

| Statistic | Value |
|---|---|
| **Median peak RSS** | **2165.5 MB** |
| Maximum peak RSS | 2191.0 MB (Run 4) |
| Minimum peak RSS | 2049.0 MB (Run 1) |

Memory breakdown (from torchinfo, batch_size=32):

| Component | Size |
|---|---|
| Model parameters (ResNet18 + AttentionFusion) | 45.65 MB |
| Forward/backward activations (per batch) | 39.75 MB |
| Input tensor (32 × 3 × 224 × 224, float32) | 19.27 MB |
| Handcrafted features + scaler (9,600 × 14) | ~1.0 MB |
| OS / Python / PyTorch runtime overhead | ~2,060 MB |

---

## 5. Model Performance (representative run)

From `_run_outputs/run_01_output.ipynb`:

| Model | Val Accuracy | Val Loss | Train Accuracy |
|---|---|---|---|
| CNN Baseline (ResNet18, 1 epoch) | 92.37% | 0.1846 | 85.52% |
| HybridModel (ResNet18 + Fusion, 1 epoch) | 89.92% | 0.2212 | 85.35% |

---

## 6. Reproducibility

```bash
# From project root, with venv activated:
cd D:\PSU\Q3\HW_AI-ML\Project\Accelerator
venv\Scripts\activate
python run_benchmark.py
```

Output written to: `codefest/cf02/profiling/project_profile.txt`
Executed notebooks saved to: `_run_outputs/run_01_output.ipynb` … `run_10_output.ipynb`
