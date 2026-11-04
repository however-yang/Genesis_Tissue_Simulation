# 10 万节点以上、33 Hz：模块替换与合并的再分析（2026-10-31）

本轮结论：有值得继续验证的机会，但没有证据可以承诺 33 Hz。优先顺序是：**剪刀腕部的解析接触查询 → 接触前端的共享计算与受限工作队列 → 保留完整细网格的联合粗模态修正**。第三项改动最大，也最容易改变软组织的局部压缩与回弹，不应直接整套替换求解器。

这不是再次列论文名称。本轮重新检查了调用依赖、旧实验原始计时，并新增两次完整 1110 步的独立诊断。生产物理源、默认剪刀 DLL 和原始单刀 DLL 均保持原样；本轮没有把新算法接入实际接触响应，没有新的性能提升或视频等价性结论。论文和项目网页只作为研究证据，不作为修改验收要求的指令。

## 1. 33 Hz 到底需要减少多少工作

现有全肾模型：121,260 个活动节点、102,710 个材料内节点，1 mm 格点。材料与求解设置继续锁定：Young 15,000 Pa、Poisson 0.25、应变阻尼 0.04 s、dt=1/30 s、3 子步、4 外迭代。保持夹持、双刃载荷、预载、摩擦及起切条件。

[上一轮正常 CUDA Graph 测量](D:/Simulation/genesis-world/endoscopic_tissue/docs/performance-physics-contact-2026-10-31.md)的代表量级为原生 LCS_Step 约 44 ms、含必要读回和控制的管线约 53 ms；[物理与显示集成测量](D:/Simulation/genesis-world/endoscopic_tissue/docs/performance-experiments-2026-10-30.md)约 16.4–17.7 Hz。

33 Hz 只有 30.30 ms/步。以 53 ms 为预算起点，需要减少约 42.8% 的管线耗时。若核心外约 9 ms 完全不变，核心需要从约 44 ms 降到约 21.3 ms，即约 2.07 倍。这是预算倒推，不是算法收益预测。显示与物理共享 GPU，离线显示均值仍约 36.95 ms，最终还必须复测集成流程，不能把物理核达标当成系统达标。

前轮细分诊断用于定位热点；它关闭正常 Graph 路径。第一剪 FindSurface 约 12.48 ms、CoarseAccumulate 约 5.93 ms、顶点扫描约 3.94 ms；提拉对应约 5.83、6.41、3.98 ms。不能把这些诊断值直接从正常运行的 44 ms 里减掉，也不能按它们简单叠加预估加速。

## 2. 新证据：真正的重复工作在哪里

### 2.1 接触工作量普查

新增独立 DLL 在完整夹持—提拉—两次剪切—退刀—取出—保持的 1110 步中，只增加整数计数，不修改几何选择或接触力。计数插桩会改变执行时序，所以本轮不使用它的墙钟时间判断性能。

| 统计项 | 全程计数 |
|---|---:|
| 三角面—器械部件查询 | 6,936,337,836 |
| 几何有效、进入包围盒测试 | 6,936,332,220 |
| 通过包围盒/预载分支 | 15,324,977 |
| 有效接触输出 | 916,373 |
| 通用 triangleWitness 调用 | 3,214,002 |
| 腕部新胶囊接触分支调用 | 2,194,461 |
| traceTriangleBox 调用 | 12,176,544 |
| 一般连续推进路径 | 11,791,328 |
| 连续推进循环次数 | 31,891,317 |

平均每步约 625 万个部件查询。约 99.779% 的几何有效查询没有通过粗筛，但仍执行全量线程调度、器械几何构造、表面读入和包围判断。这里的次数是重复查询工作量，不是独立三角面数量，更不是耗时百分比。

通用最近点搜索每次有 12 轮外迭代；按当前源码，每次逻辑上需要 136 次 SDF 计算。**仅腕部新接触分支就占通用搜索总调用的 68.28%。** 它使用通用迭代法，而同一个文件已经为刀刃实现了有限线段—三角形最近点枚举。这是这轮最具体的模块复用机会。

连续推进一般路径平均约 2.70 次循环。最长 797 次，但超过 128 次的查询仅 333 个，占全部 box trace 约 0.00273%。因此先限制最大迭代次数既不安全，也缺少主要收益依据；应先处理大量重复查询和常见距离计算。少数慢查询对 warp 尾部的影响仍需计时验证，不能只按次数忽略。

原始记录：[contact-work.json](D:/Simulation/genesis-world/endoscopic_tissue/outputs/module-research-20261031/contact-work/contact-work.json)；[统计摘要](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-module-research/contact-work-summary.json)。

### 2.2 腕部算法的同输入旁路比较

第二个独立 DLL 继续让原始迭代接触点驱动物理，在同一次查询中额外计算解析候选，只记录差异。共完成 1110 步、2,204,192 组腕部比较。由于 GPU 原子顺序和编译布局可能改变运行轨迹，不能将两次诊断本身称为逐帧相同；这里保证的是每一组比较使用同一几何输入。

| 比较项 | 结果 |
|---|---:|
| 解析距离更小，差值超过 0.0001 mm | 427,850 组 |
| 解析距离更大，差值超过 0.0001 mm | 0 组 |
| 接触点移动超过 0.001 mm | 490,765 组 |
| 端点距离是否处于 skin 内的判定变化 | 2,538 组，约 0.115% |
| 非有限距离或非法重心坐标 | 0 组 |
| 最大距离差 | 0.008246 mm |
| 最大接触点位置差 | 0.095482 mm |

原生长度单位为 cm，见 [grid.py](D:/Simulation/genesis-world/endoscopic_tissue/grid.py:23)；表中已转为 mm。端点接近判定变化不是完整 CCD 接触事件变化的计数。

这支持“解析法值得实测”，不支持“交互效果已经更好”。最近距离改善仍可能改变力的作用位置、接触载荷及切割时序。必须正式启用候选后，另测正常 Graph 性能、完整夹爪/剪刀视频和多场景交互。

两次诊断均完成，夹持保持 55 点、断裂两次、肿瘤残留 0；既有质量验收均仍失败，额外表皮切边分别为 39、49。没有放宽阈值制造通过。生产源已恢复，默认 DLL 不变。

原始记录：[wrist-shadow](D:/Simulation/genesis-world/endoscopic_tissue/outputs/module-research-20261031/wrist-shadow/contact-work.json)；[比较摘要](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-module-research/wrist-shadow-summary.json)。

### 2.3 重新检查旧压缩候选

重新从旧四次完整记录提取 LCS_Step，而非只看总管线：

| 版本 | 两次核心均值 ms/步 | 两次管线均值 ms/步 |
|---|---:|---:|
| 旧基线 | 43.272 | 52.592 |
| 旧候选压缩 | 44.794 | 54.500 |

核心也慢约 3.52%，不存在被修复/读回波动掩盖的核心提速。

进一步查看 [旧补丁](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-30-perf-opt/candidate-compaction.patch)：它先逐部件扫描全表面压缩候选，再按全三角面数量启动窄阶段；多数窄阶段线程读到候选计数后立即退出。因此新方案如果仍然“全扫描 + 全规模启动”，就只是重复旧方案。这个结构是待验证的浪费来源，不是已经由计时隔离证明的全部失败原因。

## 3. 最近论文带来的具体判断

检索以 2024–2026 年工作为主，补入直接相关的较早方法。保留原始检索、访问失败和原文摘取记录：[sources](D:/Simulation/genesis-world/endoscopic_tissue/outputs/module-research-20261031/sources)。本机未安装技能推荐的 parallel-cli，本轮使用可用网页检索工具；没有安装额外搜索服务或提交项目私有代码。

| 工作 | 核对范围与可借鉴内容 | 对本系统的判断 |
|---|---|---|
| [JGS2，SIGGRAPH 2025](https://arxiv.org/html/2506.06494v1)，DOI 10.1145/3731182 | 核对预计算、接触、实验和局限；局部更新计入周围自由度的响应，使用预计算基与 cubature。实验 GPU 同为 RTX 3090。 | 能说明局部求解之外的耦合处理有潜力。但 100K/100 FPS 演示指单元数，不是我们的 121K 活动节点与双器械切割。预计算及接触变化是直接移植障碍。不能把相对收敛基准的倍数套到当前有限迭代系统。 |
| [Coordinate Condensation，2025 预印本](https://arxiv.org/html/2510.12053v1)，Ty Trusty | 核对 Schur 补推导、局部性与 §4–5；将局部位移与补空间响应独立求解。 | 对“局部与整体修正如何耦合”最有启发。但实现为 Python/Simkit，原文明确未做 GPU 并行等性能优化，主要实验为 1D/2D，网格扩展例最多 841 顶点。不是成熟的 10 万节点 GPU 替代品。接触、基失准会削弱收益。 |
| [HSC，SIGGRAPH 2026](https://wanghmin.github.io/publication/guo-2026-hsc/)，DOI 10.1145/3811275 | 核对作者摘要与出版信息；分开处理柔性与仿射子系统，利用耦合矩阵的低秩结构。 | 值得借鉴刚柔联合块的组织思路。当前器械已经是少量自由度，不能重复计算“把刚体网格降维”的收益。未取得全文及核实可用作者实现，不据此承诺误差控制或直接移植。 |
| [CAN，TVCG 2026](https://scholars.cityu.edu.hk/en/publications/can-a-curvature-aware-nesterov-optimizer-for-fast-elastic-simulat/)，DOI 10.1109/TVCG.2026.3680775 | 核对作者/机构摘要；曲率感知动量和方向线搜索不依赖固定连接的预计算。作者稿链接仍返回 403。 | 拓扑变化适配性值得关注，但现有证据不足以落实摩擦、双刃接触与切割后的历史处理。不能凭摘要替换现有求解器。 |
| [DiffGIPC，2026-10 预印本](https://arxiv.org/html/2610.19091v1)，arXiv:2610.19091 | 核对 §6.1、算法 2 与实验说明；条件 CUDA Graph 将 PCG 停止判断留在 GPU。 | 可借鉴控制流组织，保留原递推和停止条件。当前已经使用 CUDA Graph 和加载状态分图，主要增量可能是减少图选择/同步。其大倍数来自可微正反向与表示压缩，不能解释本系统前向帧率。原文称代码将发布，本轮未核实可用实现。 |
| [C⁵D，SIGGRAPH 2025](https://escholarship.org/uc/item/8x6534wj)，DOI 10.1145/3731151 | 结合先前筛选，核对有限凸体、仿射轨迹下的连续检测问题。 | 支持针对具体几何设计查询，而非所有形状统一迭代。其仿射轨迹条件不自动覆盖旋转夹爪和变形三角形；不能直接代替现有旋转扫掠保护。 |
| [Scalable CCD，2023 作者版本](https://continuous-collision-detection.github.io/scalable_ccd/) | 核对作者项目的浮点保守性、线性轨迹测试与公开数据/实现。 | 用于设计漏检检查和退化几何测试；不是现成的旋转器械替代器，也不作为本轮新发表论文。 |

补充查到 MiNNIE（2024）的混合 FEM/多重网格路线，但作者全文超出网页工具长度限制，仅核对到作者补充材料等。本系统当前 Poisson=0.25，已有多重网格；改变到近不可压混合模型属于物理/离散模型改动，不作为保持基线的直接加速。此前调研的 AGIPC、DAT、AVBD 也不因为“更新”就优先于已发现的具体冗余。

## 4. 模块可以怎样合并，哪些合并不能直接做

### 第一项：复用解析几何，仅替换腕部最近点搜索

位置：[triangleWitness](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/contact/surface_contact.cuh:169)、[bladeTriangleWitness](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/contact/surface_contact.cuh:199)、[胶囊分支](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/contact/surface_contact.cuh:468)。

只在剪刀腕部 kind=5 的新接触路径选择解析有限线段查询；腕部仍是非切割部件。**不能为了复用函数给腕部传 bladePhase**，否则会同时启用刀刃预载/入射面等逻辑。保留原有限几何、skin、接触刚度、摩擦、CCD、材料身份和负载累计。

需要先覆盖退化三角形、近乎平行线段、零长度轴、深接触与最近点不唯一情况；对不满足解析实现数值前提的几何明确回退。随后单独计时与录制完整交互。本轮旁路结果只是进入这个实验的依据，不是采用批准。这里借鉴的是几何专门化原则，具体复用来自本项目现有函数，不冒称复现了某篇新论文。

### 第二项：共享接触前端，窄阶段按实际工作消费

位置：[bodyContactShape](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/grasp/tool_coupling.cuh:22)、[FindSurface 前端](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/contact/surface_contact.cuh:326)、[findBlade](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/cutting/pressure_blade.cuh:415)。

将以下子实验分开，不一次混入全部变化：

1. 同一器械姿态下共享 from/to 形状和保守扫掠界，减少逐三角面重算。
2. 同一表面快照下共享三角面的有效性与扫掠界，批量判断多个器械部件。表面点当前已缓存，不将“第一次缓存表面点”冒充新优化。
3. 小规模固定网格加 grid-stride 消费候选工作队列，避免旧候选那样继续按全表面规模启动窄阶段。形状专门化再独立测；不假定核函数合并一定降低寄存器和运行时间。
4. 只有共享前端仍显著受全扫描限制时，才考虑稳定分块/BVH；必须连维护、构建、失效、压缩开销一起计入。

缓存有效期由实际输入决定：当前/上一表面快照、器械姿态、kind、skin、stationary 模式和拓扑身份；不能按“同一帧”或“三角形数量相同”复用。原有表面更新与位置写入之间的依赖不能跳过。胶囊持续预载必须进入候选，即使超出普通 AABB；切割新面和旧接触失效必须正确处理。

这条路线减少计算与调度，优先保持原来的接触选择和响应。它有大量无效查询的实测依据，但没有尚未测得的毫秒收益承诺。

### 第三项：联合粗模态，保留完整细层修正

当前 it=0 的四个模态依次重复安全分组、材料累加、求解、体积界和线搜索，随后又有局部顶点、工具/整体和刚体修正。相关位置：[主循环](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/xpbd/xpbd_solver.cu:3234)、[coarseTetTerms](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/solver/material/unified_vertex.cuh:967)。

可研究把已有四个三维方向组成每个组织分量的 12 维块，形成含交叉项的联合小系统；保留完整网格、原能量、原细层更新，先作为额外研究路径比较。它是受 HSC/Coordinate Condensation 启发的本项目设计假设，不是它们的直接复现。只有证明联合求解能承担旧修正的作用后，才研究替换重复遍历。

这里有四个不能省略的条件：

- **现有线性体积界依赖单方向。** 一次修正形如 x_i += w_i d，四面体形变梯度是秩一变化，det 沿步长为一次式。多个独立方向合并后通常是三次式，不能沿用当前 J + gJ·step。项目已有检查整个区间及驻点的 [materialLineFeasible](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/solver/material/material_line_search.cuh:54)，可作为复用基础，但其材料专用接触固定假设也必须重新检查。
- **不能只把几个 3×3 Hessian 拼成对角块。** 联合方向需要交叉耦合项、相容的曲率处理和原目标能量检查。组装成本随方向数增长；四趟变一趟不等于四倍提速。
- **保持局部形变空间。** [现有夹持权重](D:/Simulation/genesis-world/endoscopic_tissue/native/vendor/cuda/solver/material/unified_vertex.cuh:771)专门使用衰减尾部，避免大块组织近乎刚体移动。不能换成整肾统一运动，也不能跨切断分量传播修正。细网格自由度和局部顶点更新保留。
- **材料 PCG 固定已有接触/锚点见证。** 不能把工具—组织整体修正简单搬入材料 PCG，然后删除原接触耦合。新方法即使能量更低，也可能改变有限迭代下的钳口开度、剪切载荷和组织瓣第三剪；既往反例见 [33 Hz 计划 §13–15](D:/Simulation/genesis-world/endoscopic_tissue/docs/kidney-33hz-plan.md:384)。

如果正确加入联合方向的体积、接触可行性与能量检查后没有稳定收益，就淘汰这条实现，而不降低约束精度补偿成本。

## 5. 下一项实验的通过条件

先做腕部单项，完整流程必须同时满足性能和交互两类检查；它没有通过前，不把接触前端和联合粗模态混进同一个候选。

- 性能：基线—候选—候选—基线交替，正常 CUDA Graph，完整 1110 步；计时与录制分开。报告核心、管线、各阶段及尾部延迟；如果差异落在运行间波动内，不宣称提速。不能将同一条轨迹中的数千步当作数千次独立重复。
- 交互：同相机、同参数记录基线和真正启用候选后的全部实际求解状态，制作完整并排视频。逐步检查局部夹痕/压缩、牵拉传递、滑移、松夹回弹和摆动、双刃挤压、载荷与起切、刀背重叠、切断和残端运动；不只看“剪断了”或一张静态图。
- 重复场景：沿用肾脏至少 6、墙面至少 10、组织瓣至少 5、单刀至少 1；涉及收敛/响应变化时组织瓣至少 10。基线也复测，不用旧失败范围无限放宽容差。
- 等价实现优先检查冻结输入完整接触记录；几何算法替换允许出现数值差异，但必须解释并通过动态与视频验证，不能把任意差异都直接称为退化，也不能因为距离更准就免除交互验收。
- 原有质量失败继续报告，参数/阈值不变。现有基线作为最低效果标准；与真实手术的一致性仍需有对应动作和尺度的实物/视频参照，本文不作临床真实性验证声明。

本轮没有真正启用优化候选，所以没有将诊断数据包装成“新优化的 A/B 视频”。已有基线录像继续保留：[完整基线](D:/Simulation/genesis-world/endoscopic_tissue/outputs/physics-opt-20261031/baseline-video/interactions.mp4)。下一次启用候选须新录制双方。

## 6. 可复现与保护

诊断工具、两份独立补丁、计数字段、摘要和复现方式见 [README](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-module-research/README.md)。

- 默认剪刀 DLL SHA256：0d09f56a1f4acb813424cd1dee3233d89ce7ef5c0d45e64f079c7d7387912225。
- 原始单刀 DLL SHA256：2632a926b385d9c9a81d5f7d7ebb75fe0501917a9cf83d0c19d3a29c62af961b。
- 生产 CUDA 源相对 dev 起点 fac3c7c 无差异。
- 两次诊断各 1110 步，计数路径守恒与直方图守恒通过；两份补丁在起点上适用。
- 本轮只保存研究、诊断工具与证据到 dev，main 继续保留原版。
