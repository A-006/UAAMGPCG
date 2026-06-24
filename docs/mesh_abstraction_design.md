# Unified `Mesh` Abstraction — Design Study

跨三条分支研究三种网格范式,为「求解器对网格类型可扩展」设计 `Mesh` 抽象。
结论:**不要**强套一个 `virtual` 公共基类(会拖垮结构化 GPU matrix-free);用**两层**设计,
且真正落地应当**伴随分支合并**进行,而非现在凭空抽象。

## 三种范式定位

| | 结构化均匀 (MAC) | 半结构 / octree | 非结构 FVM |
|---|---|---|---|
| 分支 | `feat/amgx-solver-backend` | `feat/semistruct-matrixfree-mg`* | `feat/unstructured-fvm` |
| 核心类型 | `Mesh2D`/`Mesh3D`(`nx,ny,nz`,`dx,dy,dz` + 索引公式) | `AdaptiveGrid2D/3D`(若干均匀 `Level` 栈 + LEAF/REFINED/OUTSIDE 掩码) | `PolyMesh`(`centroid[]`,`vol[]`,`Face{owner,nb,Sf,Cf}`) |
| 文件 | `include/core/mesh.h`,`mesh_3d.h`,`grid*.h` | `include/semistruct/adaptive_grid_*.h`,`poisson_operator_*.h`,`sparse.h` | `include/unstructured/poly_mesh.h`,`fvm_poisson.h` |

\* semistruct 头文件已前向合并进 `feat/amgx-solver-backend` 树的 `include/semistruct/`。

## 关键发现

1. **核心结构**:结构化 = 纯索引算术、不存拓扑;octree = 每个 LEAF 显式 DOF(`dof_level/i/j[/k]`)+ 跨层邻居解析(`coveringLeaf`/`childrenOnFace`)+ rank-1 PSD T-junction 耦合;FVM = 存 cell-face 连接 + 预算 `Sf`/`Cf`/`vol`。

2. **求解器向网格要的操作**:共享*语义* — DOF 遍历、单元体积、面面积/法向(查询式)、单元中心坐标、BC 标记、散度。范式专属 — O(1) `i±1` stride + MAC `iu/iv/iw`(结构化);跨层邻居 + aggregation Galerkin(octree);变长面表 + 最小二乘梯度 + 非正交修正(FVM)。

3. **LCD 陷阱(读 GPU kernel 确认)**:结构化 matrix-free 把邻居访问做成单条 3-stride 寄存器加法(`idx3d`),从 `cx/cy/cz/diag` 平铺数组读**逐单元**系数,bool 数组掩固体,**任何层都不建 CSR**(粗网格 Galerkin 算子由细系数即时求和)。若用 `virtual neighbor()` 或 CSR `row_ptr/col_idx` 间接来"对齐"非结构路径,会把寄存器 stride-add 换成依赖载入、毁掉合并访存——纯粹的最低公分母损害。

4. **建议的 `Mesh`(两层)**:
   - **Layer A `MeshDescription`** — 值类型(`num_dofs`/`volume`/`coords`/`boundary` + `assemble_poisson → 0-based CSR`),给冷路径(I/O、BC、MMS、factory)和 AmgX 交接用;三种范式都能廉价实现。
   - **Layer B 算子应用走 模板/CRTP**(`PoissonOp<MeshT>`),**绝不 `virtual`**,让结构化和 octree 的 GPU kernel 保持零分发内联访问。
   - 唯一共享运行时缝:**「产出 CSR → `AmgxBackend::solve_csr`」**。

5. **风险 / 不要统一的**:别把算子应用、多重网格粗化(几何 2×2×2 vs aggregation `PᵀAP` vs AmgX 代数——是真不同的算法)、MAC 面索引空间、LS 梯度/非正交修正 塞进公共基类。`DofId` 当不透明(别绕回 `(i,j,k)`)。跨范式耦合完全推后。

## 迁移草图

1. 把重复的 `semistruct::CSR` 和 `ufvm::CSR` 合成一个 `core::CSR`(便宜,启用共享 `solve_csr`);
2. 把 `Grid3D`(已经 `: public Mesh3D`)收成 `StructuredMesh` 的 Layer-A 实现,字段门面与 GPU kernel 因 Layer B 是模板而**逐字节不变**;
3. 把 `AdaptiveGrid3D`+`PoissonOperator3D` 包成 `OctreeMesh`;
4. `PolyMesh` 已是 FVM 网格,只加 shim;
5. factory 按 `MeshKind` 分发。
6. 模板 `PoissonOp<MeshT>` 重构应**伴随一次强制分支合并**落地,别在吞吐关键 kernel 上凭空改。
