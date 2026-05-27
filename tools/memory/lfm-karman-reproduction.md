---
name: lfm-karman-reproduction
description: "LFM Karman vortex street reproduction status — what works, what doesn't, key bugs found"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e859b79-adab-4e52-8195-e975464c618e
---

# LFM 论文复现 — Karman 涡街

## 目标

复现 Sun et al. 2025 "Leapfrog Flow Maps for Real-Time Fluid Simulation" (UAAMG.pdf) 的 Karman 涡街。

## 论文 vs 我们

- 论文用 LFM Algorithm 1（impulse-based flow map）+ PCG+UAAMG 求解器
- 圆柱边界：论文用 stair-step solid cells（和现有 `setupCylinder` 一样），不是 body-fitted baffle。baffle 只在 OpenFOAM 参考算例中。
- 论文没有给出 Karman 的具体参数（网格分辨率、n_steps、solve_iters）
- 论文没有给出 Cd/Cl/St 定量指标，只有 Figure 8 定性图片

## 当前状态

### Chorin 模式 ✅
- Cd=1.48（文献 1.30-1.40），St=0.20（文献 0.19-0.20），3/3 验证通过
- GPU 加速 6.4x
- 命令：`build/test_karman_validate_gpu 256 10 chorin gpu`

### LFM 模式 ⚠️
- Algorithm 1 完整实现 + 29 个单元测试全部通过
- F=I（单位雅可比）：稳定但不演化流场（Cd≈0）
- F 演化（dF/dt=∇u·F）：Karman 验证爆炸（div→10^6）
- 根因：阶梯圆柱边界 ∇u≈1/dx≈32，Jacobian ODE 在 10 步后不稳定

## 发现的关键 bug

1. **`sample_velocity` 忽略参数 u/v** — 始终读 `grid_` 而非传入的速度场。导致流映射追踪、pullback、路径积分全部使用错误速度场。修复：改为从传入 vector 双线性插值。

2. **`velocity_gradient` 忽略参数 u/v** — 同上，Jacobian 演化使用错误的 ∇u。修复：改为从传入 vector 计算差分。

3. **这两个 bug 在单元测试 T1-T7 中被完美隐藏** — 因为所有测试都是均匀流，传入速度和 `grid_` 相同。只有圆柱测试 T8/T9 暴露。

## 可用文件

### 测试
- `test/integration/test_lfm.cpp` — LFM 单元测试（T1-T9）
- `test/integration/test_karman_validate.cpp` — Karman 验证（CPU）
- `test/integration/test_karman_validate_gpu.cu` — Karman 验证（GPU）

### LFM 实现
- `include/lfm/flow_map_2d.h` + `src/lfm/flow_map_2d.cpp` — 流映射数据结构
- `include/lfm/lfm_simulator.h` + `src/lfm/lfm_simulator.cpp` — Algorithm 1

### 配置
- `cfg.time_integrator = "chorin"|"lfm"` — 切换求解器
- `cfg.lfm_cycle_steps = 10` — LFM 周期步数

### 之前尝试过但未成功的
- `include/geometry/cylinder_model.h` + `src/geometry/cylinder_model.cpp` — SmoothCylinder（变系数拉普拉斯），数值不稳定，已废弃
- `clamp_out_of_solid` — 粒子位置修正，不够，已移除
- 浸入边界（`sample_velocity` 检测圆柱内返回 0）— Cd 仍然不对，已移除

## 后续方向

要让 LFM Karman 验证通过，需要解决 Jacobian 演化在阶梯边界的稳定性：
1. 更高网格分辨率（减小 ∇u）
2. 更短 reinitialization 周期（n=2 而非 n=10）
3. W-cycle 替代 V-cycle
4. 或接受 Chorin 模式作为可工作的验证方案
