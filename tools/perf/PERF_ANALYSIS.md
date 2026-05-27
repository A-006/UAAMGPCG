# CUDA 3D AMGPCG Performance Analysis

Date: 2026-05-24 | GPU: 2× RTX 3090 (CC 8.6) | Driver: 535.288.01 | ncu: 2023.2.0 (CUDA 12.2)

## System

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3090 ×2 |
| Compute Capability | 8.6 |
| SMs | 82 |
| Max Shared Memory / SM | 100 KB |
| Max Registers / SM | 65536 |
| CUDA Driver | 535.288.01 |
| Nsight Compute | 2023.2.0 (CUDA 12.2) |
| Build Compiler | nvcc (CUDA 12.x via /usr/local/cuda) |

## Benchmark Baseline (FP64, no profiling)

```
256x128x128 (4.2M cells):
  V-cycle time:  2.78 ms
  GPU PCG(40):         223.97 ms
  GPU PCG-opt(40):     185.76 ms  speedup=1.21x

128x64x64 (0.5M cells):
  V-cycle time:  0.61 ms
  GPU PCG(30):          33.71 ms
  GPU PCG-opt(30):      28.22 ms  speedup=1.19x

V-Cycle Cost (128x64x64):
  Total V-cycle:    0.543 ms
  Per-MCell cost:   1.035 ms/Mcell
  (Paper: 0.19 ms/Mcell on RTX 4090)
```

## Profiled Kernels (64×32×32 grid, optimized path)

### 1. axpy_dot_kernel
- **Duration**: 8.96 µs | **Compute**: 36.9% | **DRAM**: 14.6%
- **Occupancy**: 53.3% (limited by Shared Mem: 3 blocks/SM of 16)
- **Registers**: 16/thread | **Shared Mem**: 4.1 KB/block
- **Issue**: Low occupancy due to shared memory config limit

### 2. matvec_tiled_dot_kernel
- **Duration**: 12.67 µs | **Compute**: 39.7% | **DRAM**: 5.4%
- **Occupancy**: 52.4% (Register: 4 blocks, Shared Mem: 3 blocks)
- **Registers**: 31/thread | **Shared Mem**: 17 KB/block (3× 10³ halo arrays)
- **Issue**: Register pressure (31 regs → 4 blocks/SM) + SMEM (17KB → 3 blocks/SM)
- **Note**: DRAM at 5.4% confirms shared memory tiling works — most reads from SMEM

### 3. rbgs_restrict_aggregated_kernel (finest level, 128 blocks)
- **Duration**: 28.41 µs | **Compute**: 52.9% | **DRAM**: 4.8%
- **Occupancy**: 49.3% (Register: 3 blocks)
- **Registers**: 40/thread | **Shared Mem**: 9 KB/block
- **Issue**: Register pressure is the binding constraint

### 3b. rbgs_restrict_aggregated_kernel (level 2, 16 blocks)
- **Duration**: 16.54 µs | **Compute**: 12.2%
- **Occupancy**: 32.6% | **Waves/SM**: 0.07
- **Issue**: Too few blocks to fill GPU → ~6.5% utilization

### 3c. rbgs_restrict_aggregated_kernel (level 3, 2 blocks)
- **Duration**: 16.08 µs | **Compute**: 1.6%
- **Occupancy**: 32.6% | **Waves/SM**: 0.01
- **Issue**: Only 2 blocks → ~1% GPU utilization → 99% wasted cycles

### 4. prolong_rbgs_aggregated_kernel (finest level, 128 blocks)
- **Duration**: 24.98 µs | **Compute**: 51.3% | **DRAM**: 5.5%
- **Occupancy**: 50.1% (Register: 3 blocks)
- **Registers**: 38/thread | **Shared Mem**: 9 KB/block
- **Issue**: Same register pressure as restrict

### 4b. prolong_rbgs_aggregated_kernel (level 2, 16 blocks)
- **Duration**: 15.15 µs | **Compute**: 11.4%
- **Issue**: ~6.5% GPU utilization

### 4c. prolong_rbgs_aggregated_kernel (level 3, 2 blocks)
- **Duration**: 14.78 µs | **Compute**: 1.6%
- **Issue**: ~1% GPU utilization

### 5. rbgs_coarsest_kernel (single block)
- **Duration**: 56.61 µs | **Compute**: 0.89% | **DRAM**: 0.01%
- **SM Active Cycles**: 936 / 78,622 = **1.2%**
- **Registers**: 40/thread | **Occupancy**: 33.1%
- **CRITICAL**: 1 block on 82 SMs → 99% of GPU cycles are idle. This kernel runs 20 sweeps in 1 launch but is effectively serial.

## Bottleneck Summary (sorted by impact)

| Priority | Bottleneck | Kernel(s) | Wasted Time | Optimization |
|---------|-----------|-----------|-------------|--------------|
| P0 | Coarsest: 1 block, 99% idle | rbgs_coarsest_kernel | ~56 µs/V-cycle | Multi-stream V-cycles |
| P1 | Small hierarchy levels: 1-6% GPU util | restrict/prolong L2-L3 | ~30 µs/V-cycle | Level fusion or wavefront parallelization |
| P2 | Register pressure: 38-40 regs → 50% occup. | All aggregated kernels | ~2× headroom | Reduce live variables, SOA layout |
| P3 | matvec_tiled SMEM: 17KB → 3 blocks/SM | matvec_tiled_dot | Moderate | Warp shuffle halo exchange |
| P4 | axpy_dot SMEM config limiting blocks | axpy_dot | Minor | SMEM config tuning |

## Per-V-Cycle Cost Breakdown (128×64×64, FP64)

```
V-cycle = 0.543 ms total

Level 0 (128×64×64, 128 blocks):  ~60 µs   (restrict 28µs + prolong 25µs + setup)
Level 1 (64×32×32, 16 blocks):    ~35 µs   (restrict 17µs + prolong 15µs + setup)
Level 2 (32×16×16, 2 blocks):     ~33 µs   (restrict 16µs + prolong 15µs + setup)
Coarsest (16×8×8, 1 block):       ~57 µs   (20 sweeps fused, 99% idle)

Per-V-cycle overhead: ~185 µs in actual kernels
Remaining ~358 µs in launches, memset, and host overhead
```

Note: These are from the 64³ grid profiling scaled to 128³; actual 128³ times are proportionally larger.

## Optimization Roadmap (from README + profiling data)

### Phase 1: Fix GPU Underutilization (est. 1.3-1.5×)
- Multi-stream V-cycles: overlap coarsest solve with other streams' work
- Level fusion: merge small hierarchy levels into single kernel

### Phase 2: Reduce Register Pressure (est. 1.1-1.2×)
- Reorder computation to reduce live variable count from 38-40 to ~32
- Use `__launch_bounds__` hints to guide compiler
- SOA 5-channel data layout packs coefficients and reduces per-thread state

### Phase 3: Memory System (est. 1.1-1.2×)
- Warp-level primitives (`__shfl_sync`) for halo exchange → reduce SMEM from 17KB to ~8KB
- CUB global scan for dot reduction → move reduction from host to GPU
- Texture memory for solid mask lookups

### Cumulative Impact: ~2× potential speedup

## Files in this directory

| File | Description |
|------|-------------|
| PERF_ANALYSIS.md | This analysis document |
| benchmark_baseline_fp64.txt | Raw benchmark output (FP64) |
| ncu_optimized_kernels.csv | Detailed ncu metrics for 5 optimized kernels (CSV) |
| ncu_optimized_summary.txt | Per-kernel text summary |
| ncu_unoptimized_summary.txt | Unoptimized kernel summary for comparison |
| ncu_memory_metrics.csv | Memory-specific metrics (bytes, cache throughput) |
| ncu_memory_metrics.txt | Memory metrics text summary |

## How to re-profile

```bash
# Full benchmark (no profiling)
./build/src/solver/cuda/test_paper_bench

# Profile only optimized kernels with ncu
/usr/local/cuda-12.2/bin/ncu --set basic \
    --kernel-name regex:"tiled_dot|axpy_dot|_aggregated_|_coarsest" \
    --launch-count 20 -f -o /tmp/ncu_latest \
    ./build/src/solver/cuda/test_paper_bench

# Export to CSV
/usr/local/cuda-12.2/bin/ncu --csv --print-summary per-kernel \
    --import /tmp/ncu_latest.ncu-rep > output.csv
```
