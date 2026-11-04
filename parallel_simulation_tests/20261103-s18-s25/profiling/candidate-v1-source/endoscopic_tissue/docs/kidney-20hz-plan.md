# 肾脏切除场景：面向 20 Hz 的提速方案（第二轮调研，2026-10-29）

## 前提

- **模型规模。** 只模拟肾脏，1 mm 网格，共 121,260 个活动节点：102,710 个材料节点，18,550 个包络壳节点。网格有 115 × 77 × 40 = 354,200 个格点。
- **硬性要求。** 效果不变；新模块必须能直接接进现有系统：能量模型、隐式欧拉、接触模型、切割与对偶轮廓表面、CUDA graph 都保持不动。
- **目标。** 每步不超过 50 ms，即 20 Hz。背景数据见 [kidney-realtime-survey.md](kidney-realtime-survey.md)。

## 1 结论

| 阶段 | 现在（ms/步） | 只做不改物理的项 | 再加多重网格预条件和粗模态合并 | 再加接触修复改进 |
|---|---|---|---|---|
| 夹持（settle–blade_approach） | 69–73 | 55–58 | 46–50 | 46–50 |
| 切割（incise_1–incise_2） | 76–110 | 64 + 修复 | 55 + 修复 | 55–58 |
| 切除后（withdraw–hold） | 83–104 | 69 + 修复 | 48 + 修复 | 48–51 |

表里"修复"指剪完后的接触面修复，同一版本每步 0.3–22 ms，依运行而定。

- 夹持阶段和切除后可以到 20 Hz 左右；切割阶段约 17–18 Hz。按各阶段的步数加权，整个流程约 19–20 Hz。
- 外层非线性迭代现在是 4 次。如果多重网格让收敛足够好，可以减到 3 次，前提是实测的最终残差不比现在差。这样每步再省约 8 ms，切割阶段也能到 20 Hz，整体约 23 Hz。
- 每一项估算都有约 ±30% 的误差，所以要按顺序做，每做一步就实测一次。

## 2 本轮新查到的论文

| 文献 | 关键事实 | 和我们的契合度 |
|---|---|---|
| Dick、Georgii、Westermann 2011，SIMPAT 19(2)，GPU 多重网格六面体有限元 | 所有多重网格步骤都免矩阵；规则模板让内存访问可以合并。2011 年的 GPU 上，26.9 万个六面体单元每秒 11 步 | 规则格点配多重网格的直接证据 |
| Dick 等 2011，TVCG（`参考文献/` [40]） | 切口两侧的粗网格单元按连通块复制；收敛率在没有切口时为每个 V-cycle 0.66，有切口时为 0.75 | 切割模型和我们相同（链接体素配对偶轮廓），粗层的切口处理可以直接照搬 |
| McAdams 等 2011，SIGGRAPH，带接触的弹性蒙皮 | 六面体格点上的共旋转弹性；刚度矩阵保持正定；多重网格预条件 CG（MGPCG）配碰撞 | 确认"格点 + MGPCG + 接触"这条路可行，但它在 CPU 上 |
| Mitchell 等 2015，SIGGRAPH Asia，非流形水平集 | 格点上允许同一位置放多份单元，用来表示零宽切口和薄缝 | 粗层"一个位置多份节点"的数据结构依据 |
| StiffGIPC 2025（connectivity-enhanced MAS） | 11.4 万顶点时，每次牛顿迭代的 CG 次数从 1079（块对角）降到 225；PCG 时间约为原来的 1/2.6–1/2.9 | 每次迭代的代价约为块 Jacobi 的 1.8 倍。我们的系统条件好得多（32 次迭代），净收益小，作为备选 |
| Courtecuisse 等 2014，Medical Image Analysis，手术切割实时仿真 | 稀疏分解作预条件，低频异步更新；切割时做低秩修正 | 36 万自由度下分解太大，每次迭代还要做两次三角回代，远慢于我们 112 µs 的一次 SpMV。不适合 |
| Xian、Tong、Liu 2019，SIGGRAPH Asia，Galerkin 多重网格 | 插值算子用由静止网格预计算的蒙皮权重 | 一剪切预计算就失效。不适合 |
| Lan 等 2022，SIGGRAPH，GPU 上的无穿透投影动力学 | 投影动力学配 IPC，用 A-Jacobi 求解 | 要把接触模型换成 IPC，会改变效果。不适合 |
| Wu、Dick、Westermann 2013（`参考文献/` [42]） | 按拓扑插值体网格上的穿透深度，接触与渲染表面解耦 | 可以从根上去掉修复，但接触几何会变，留作以后考虑 |

## 3 替代模块和接入方式

### 3.1 力学：切口感知的几何多重网格预条件（MGPCG）

- **替换的部分。** 只替换 PCG 里的块 Jacobi 预条件，即 `coupled_material.cuh` 里的 `diagonal[i].solve(r)`。
- **保持不变的部分。** CG 主循环、半精度 8 块模板（`k_MaterialStencilAssemble`）、CG 节点掩码、按连通块分开的 CG 标量、固定点和见证点的处理、线搜索、顶点块下降（VBD）扫描、粗模态，以及接触。
- **新增的部分。**
  - **聚合。** 每个 2×2×2 块里，只有通过未断连接相连的节点才归为同一个粗节点；块被切口分开时，就生成多个粗节点（[40] 的做法）。这样粗层不会跨切口耦合，各连通块的 CG 也继续互相独立。聚合表只在拓扑变化时重建，这借鉴了 MGPBD 拓扑不变就复用层次结构的思路。
  - **粗层算子。** 每次求解时，从细层模板做一遍 Galerkin 求和（PᵀAP），约读 17 MB 数据。
  - **V-cycle。** 用阻尼块 Jacobi 平滑，前后各一次，保持对称，使预条件保持正定。代价约等于 2.5 次 SpMV。
- **迭代次数怎么定。** 不追求比现在收敛得更好，而是标定到"二次模型下降比例的中位数和第 5 百分位都不低于现在 32 次、80 次时的水平"，然后固定这个次数。
- **预期收益。** 只有夹持阶段降到 8 次迭代以内才划算，那时 PCG 从 15 ms 降到 8–10 ms。自由肾那 80 次迭代可以降到约 10 次，每步省 17–19 ms。
- **风险。** 误差分布会变：低频误差少了，高频误差交给后面的 VBD 扫描。所以要按验收指标对比效果，见第 5 节。

### 3.2 力学：只遍历肾脏节点

- **现状。** 每步 35.4 万个格点里只有 34% 是活动节点，但 PCG、粗模态、组装、刚体模态等逐节点 kernel 仍按全部格点启动线程。
- **做法。** 照 `k_SweepSites` 的样子列出活动节点，这些 kernel 只遍历这张表。数据布局不变，只改线程到节点的对应。
- **预期。** 每步省 2–4 ms，需要实测。结果和现在一样，只有求和顺序带来的舍入差别。

### 3.3 力学：粗模态 1–4 合成一个块求解

把 4 个依次求解的粗模态合成一次块求解，每步省约 3 ms。这会改变求解路径，需要验证。

### 3.4 切割：按拓扑版本号决定是否重算连通分量和碎片接触

- 剪切代码（cut.cu）每改变一次连接就把拓扑版本号加 1。版本号没变时，不重算连通分量，每步省 1.3 ms。
- 碎片接触的 8 遍放进 CUDA 12.5 的条件节点里，查询数为 0 就不发射这些 kernel；也可以改用很小的网格循环。现在每次发射约 16 µs、每步 216 次，合计 3.5 ms，可以省下其中约 3 ms。
- 碎片接触之后的体积可行性检查保留，因为它会写 `d_contactMobility`，有副作用。
- 这一项不改物理。

### 3.5 切割：工具接触搜索的宽相剔除

- **现状。** 剪刀和夹钳每次搜索都对全部表面三角形做扫掠测试。剪刀一项每步 4.1 ms，夹钳一项每步 4.0 ms，切割时剪刀搜索在每次外层迭代里都要重复。
- **做法。** 每个子步先算出每块三角形当前和上一步位置的包围盒，搜索 kernel 的每个线程块先拿它和工具扫掠体比较，扫掠体按接触半径放大，不相交就直接返回（[48][42] 的思路）。
- **预期。** 剔除是保守的，所以找到的接触集合不变，每步省 4–6 ms。

### 3.6 切割：剪完后的接触面修复

- **A4（不改物理）。** 修复循环里拓扑不变，所以每遍不再重复做与拓扑相关的刷新（连通分量、碎片支撑）；每遍里的 3 次剪刀搜索也会因 3.5 而变便宜。每遍从约 7.5 ms 降到约 5.5 ms。
- **M2（改变路径，风险最高）。** 法向接触加增广拉格朗日乘子（AVBD 的思路），乘子存进现有的接触缓存，靠持久的接触编号跨迭代、跨步保留。
  - 现在的问题：有限刚度接触在载荷下留下 0.10–0.15 mm 的残余穿透，正好压在约 0.1 mm 的修复阈值附近，所以哪些步要修复全看轨迹。
  - 加乘子以后，平衡时的穿透接近 0，修复就很少触发。
  - 修复阈值和 0.2 mm 的重叠验收都不动。
  - 代价是接触处的组织多被压缩约 0.1 mm，所以双刃挤压起切的载荷和挤压量都要重新验证。

### 3.7 读回

邻接表只在拓扑版本号变化时读回；读回缓冲用锁页内存。每步省约 2 ms，不改物理。

## 4 估算依据

每步的节省量（ms），按阶段：

| 项 | 夹持 | 切割 | 切除后 |
|---|---|---|---|
| 3.4 连通分量与碎片接触 | 4.3 | 1.3–4.3 | 1.3 |
| 3.5 宽相剔除 | 5.5 | 5 | 5.5 |
| 3.7 读回 | 2 | 2 | 2 |
| 3.2 只遍历活动节点 | 3 | 3 | 3 |
| 3.1 多重网格预条件 | 5–7 | 5–7 | 17–19 |
| 3.3 粗模态合并 | 3 | 3 | 3 |
| 3.6 修复 | 0 | 0.3–22 降到 0–3 | 0.3–22 降到 0–3 |

各阶段步数：夹持 450 步，切割 390 步，切除后 270 步。

## 5 实施顺序与验证

1. **先做 3.2、3.4、3.5、3.6 的 A4 部分和 3.7。** 这些都不改物理。验证方法：前 150 步的节点位置逐位相同（两次运行大约从第 170 步起才因浮点原子操作分叉）；每个阶段的步时间；肾脏 6 次运行的验收指标和现在一致。
2. **离线做多重网格原型。** 先从夹持、切割、自由肾各导出几步的 CG 方程组（模板加右端项），离线测要几次迭代才能达到现在的质量。夹持阶段降到 8 次以内才继续往下做。
3. **把多重网格接进系统。** 验证：肾脏 6 次以上、墙 10 次、皮瓣 5 次运行，看每刀的断裂载荷、双刃挤压量、重叠、碎片数、肿瘤残留和瘤蒂残端回弹。
4. **粗模态合并（3.3）。** 用同样的指标验证。
5. **视切割阶段的修复占比决定是否做 M2。** M2 要你单独同意。
6. **外层迭代从 4 次减到 3 次。** 只有在多重网格之后，最终非线性残差的分布不比现在差时才考虑，同样要你同意。

## 参考

- Dick, Georgii, Westermann. A Real-Time Multigrid Finite Hexahedra Method for Elasticity Simulation using CUDA. SIMPAT 19(2), 2011. https://www.cs.cit.tum.de/cg/research/publications/2011/a-real-time-multigrid-finite-hexahedra-method-for-elasticity-simulation-using-cuda
- McAdams et al. Efficient elasticity for character skinning with contact and collisions. SIGGRAPH 2011. https://www.math.ucdavis.edu/~jteran/papers/MZSETTS11.pdf
- Mitchell, Aanjaneya, Setaluri, Sifakis. Non-manifold Level Sets. SIGGRAPH Asia 2015. https://pages.cs.wisc.edu/~sifakis/papers/nonmanifold_levelsets.pdf
- StiffGIPC. https://arxiv.org/html/2411.06224v4
- Courtecuisse et al. Real-time simulation of contact and cutting of heterogeneous soft-tissues. MedIA 18(2), 2014. https://hal.archives-ouvertes.fr/hal-01097108
- Xian, Tong, Liu. A Scalable Galerkin Multigrid Method for Real-time Simulation of Deformable Objects. SIGGRAPH Asia 2019. https://history.siggraph.org/?p=213134
- Lan et al. Penetration-free Projective Dynamics on the GPU. SIGGRAPH 2022. https://history.siggraph.org/?p=117843
- Wu, Dick, Westermann. Efficient Collision Detection for Composite Finite Element Simulation of Cuts in Deformable Bodies. 2013. https://cs.cit.tum.de/cg/research/publications/2013/collision-detection-for-composite-finite-elements
- AVBD. https://graphics.cs.utah.edu/research/projects/avbd/
- MGPBD. https://arxiv.org/abs/2505.13390
