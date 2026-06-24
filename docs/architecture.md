# 目标架构:三根正交轴(mesh × projection × solve)

设计原则:**一个文件夹 = 一个职责**。CFD 求解器天然分三根**互不相干**的轴,任意组合
(任意网格 × 任意时间积分 × 任意线性求解器),这正是 OpenFOAM 式工业软件的分法。

```
mesh/         网格 / 离散     —— 空间怎么离散(结构化 / octree / 非结构)
integrator/   投影法 / 时间积分 —— 流体怎么往前推(Chorin: advect→project;LFM: flowmap→project)
solve/        线性求解器       —— 解压力泊松 Ax=b(CG / PCG / AMG / AmgX / Jacobi)
---- 支撑轴 ----
ic/           初始条件原语     —— 造初始场(vortex_ring / filament / cylinder …),数据驱动
io/           输入+输出        —— 读 case 文件(input) + 写 VTK/诊断(output)
core/         共享底层         —— 场容器、数学、util
```

三主轴正交:换网格不动积分器,换求解器不动网格,换积分器不动求解器。

## 现状 → 目标 映射

| 现在 | 里面是什么 | 该归到 |
|---|---|---|
| `core/`(grid*, mesh*) | 结构化网格 + 场容器 | **mesh/**(网格)+ **core/**(场/util 留下) |
| `semistruct/` | octree 网格 **+** 算子 **+** 多重网格求解器(竖井) | **拆**:网格→mesh/,MG→solve/,算子→各自轴 |
| `unstructured/`(在 feat/unstructured-fvm 分支) | PolyMesh + FVM | **mesh/** + **solve/** |
| `simulator/`(lfm_*, chorin_*, flow_map*, runner*) | 时间积分 / 投影法 | **integrator/** |
| `simulator/scenarios/` | 场景机制(去场景化后只剩 IC 数学) | **ic/**(原语)+ 删场景机制 |
| `solver/`(cg, pcg, amgx, jacobi, factory) | 线性求解器 | **solve/** ✓(已基本就位) |
| `numerics/advection` | 平流算子 | **integrator/**(投影法用) |
| `numerics/pressure` | 压力投影算子 | **integrator/** |
| `numerics/ops` | 离散算子(Laplacian 等) | **solve/** 或 **mesh/**(看是装配还是应用) |
| `numerics/bc` | 边界条件 | **mesh/**(边界是网格属性)|
| `config/` | Config 模型 + 解析器 | **core/**(模型)+ **io/**(解析=input) |
| `io/` | VTK 写出 | **io/**(output)✓ |

两个最违背正交的:**`numerics/` 是杂物袋**(投影法和求解器两轴混在一起);**`semistruct/` 是竖井**(一个文件夹塞了三轴)。

## 轴之间的缝(接口)—— 正交的关键

三轴要能独立替换,缝必须定义清楚:

- **integrator → solve**:积分器投影时调 `solver.solve(系统)`,不关心是 CG 还是 AMG。
- **mesh → solve**:求解器要的是一个线性系统。两层(见 [mesh_abstraction_design](mesh_abstraction_design.md)):
  - 冷路径 / AmgX:`mesh.assemble_poisson() → CSR → solver.solve_csr(CSR)`,solve 完全不认识网格类型。
  - 热路径 / matrix-free GPU:`PoissonOp<MeshT>` 模板把网格访问内联进 kernel——**这是性能上的有意耦合,用编译期模板实现,不破坏文件夹职责划分**(mesh/ 出算子定义,solve/ 出迭代框架,模板在编译期组合)。
- **integrator → mesh**:积分器从网格拿场、邻居、算子;`DofId` 当不透明。

## 诚实的边界

- **matrix-free 把 mesh+算子在热路径上融合**是对的(寄存器 stride-add,不能为"纯正交"换成虚函数/CSR 间接——会毁 GPU 性能)。正交体现在**接口与文件夹职责**,不要求运行时零耦合。
- **多重网格粗化**(几何 2×2×2 vs aggregation PᵀAP vs AmgX 代数)是真不同算法,放 solve/ 下各自实现,别强求公共基类。
- `semistruct/` 竖井的拆分,应**伴随分支合并**做(动到吞吐关键 kernel),不要现在凭空拆。

## 落地顺序(增量,每步可验证)

1. **已完成**:去场景化(2D+3D IC 数据驱动),scenarios 只剩 IC 数学 → 下一步收进 `ic/`。
2. 当前仓库内的纯重组(低风险):`simulator/ → integrator/`、`scenarios/ → ic/`、`numerics/` 按轴拆、`config` 模型与解析分离、`io` 收 input+output。
3. `mesh/` 抽象 + `semistruct` 竖井拆分:**伴随 octree / unstructured 分支合并**做,按 [mesh_abstraction_design](mesh_abstraction_design.md) 的两层方案。
