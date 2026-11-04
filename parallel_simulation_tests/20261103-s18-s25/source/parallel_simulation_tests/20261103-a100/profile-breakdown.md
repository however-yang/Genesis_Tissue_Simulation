# 原生分段耗时诊断

共 19 个插桩步骤，覆盖牵拉、第一剪附近及剪后状态。此模式绕过部分 CUDA Graph 并增加事件；不混入常规性能表。

| CUDA 事件区间（原标签） | 样本数 | 平均 ms/步 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| begin frame, cut tick and rebuild between substeps | 19 | 0.719 | 0.476 | 2.059 |
| substep prologue: liveness, fragment contact, predict | 19 | 1.962 | 1.248 | 10.187 |
| pre: tool, contact discovery, manifold | 19 | 2.727 | 1.611 | 3.814 |
| pre: predictor, recovery query | 19 | 1.236 | 1.099 | 2.435 |
| loop: manifold, contact marks | 19 | 4.831 | 1.342 | 5.725 |
| loop: coarse modes | 19 | 3.541 | 3.310 | 3.686 |
| cg assembly: start | 19 | 0.018 | 0.015 | 0.030 |
| cg assembly: stencil and residual | 19 | 1.127 | 1.089 | 1.162 |
| cg assembly: mask | 19 | 0.080 | 0.078 | 0.085 |
| cg: assembly | 19 | 2.248 | 2.239 | 2.279 |
| cg: iterations | 19 | 1.124 | 1.114 | 1.168 |
| cg: trust, projection | 19 | 0.028 | 0.026 | 0.033 |
| cg: feasible | 19 | 0.453 | 0.413 | 0.537 |
| cg: line search | 19 | 0.578 | 0.553 | 0.606 |
| loop: contact discovery | 19 | 5.636 | 2.758 | 6.743 |
| loop: regional mode | 19 | 1.709 | 1.558 | 2.049 |
| loop: local sweeps | 19 | 4.561 | 3.678 | 4.772 |
| loop: tool, safe and global coarse | 19 | 3.551 | 3.322 | 3.902 |
| loop: rigid modes | 19 | 0.951 | 0.941 | 0.995 |
| post: free-component nullspace | 19 | 0.286 | 0.238 | 0.298 |
| post: fragment contact | 19 | 1.369 | 1.291 | 1.752 |
| post: blade and contact reactions, velocity | 19 | 0.489 | 0.256 | 0.706 |

| C ABI 主机墙钟（包含同步/读回） | 样本数 | 平均 ms/步 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| LCS_SetGraspTool | 19 | 0.113 | 0.051 | 0.390 |
| LCS_SetScissors | 19 | 0.009 | 0.004 | 0.040 |
| LCS_SetGravity | 19 | 0.017 | 0.008 | 0.049 |
| LCS_Step | 19 | 39.531 | 29.782 | 51.753 |
| LCS_Finalize | 19 | 0.339 | 0.270 | 0.536 |
| LCS_RefineContactSurface | 19 | 0.338 | 0.203 | 0.417 |
| LCS_GetCornerPos | 19 | 0.335 | 0.160 | 0.905 |
| LCS_GetTopologyRevision | 19 | 0.022 | 0.014 | 0.076 |
| LCS_GetGraspCoreFlags | 19 | 0.117 | 0.062 | 0.352 |
| LCS_GetSolvedTool | 19 | 0.269 | 0.254 | 0.348 |
| LCS_GetSolvedJaw | 19 | 0.014 | 0.012 | 0.029 |
| LCS_GetPressureBlade | 19 | 0.039 | 0.031 | 0.096 |
| LCS_GetScissors | 19 | 0.048 | 0.041 | 0.104 |
| LCS_GetScissorsLoads | 19 | 0.015 | 0.012 | 0.028 |
| LCS_GetContactReaction | 19 | 0.019 | 0.015 | 0.056 |
| LCS_GetJawLoad | 19 | 0.016 | 0.012 | 0.045 |
| LCS_GetMaterialQuality | 19 | 0.137 | 0.090 | 0.370 |
| LCS_GetSurfaceCounts | 19 | 0.016 | 0.012 | 0.042 |
| LCS_GetSurface | 19 | 0.363 | 0.217 | 0.931 |
| LCS_GetSurfaceAux | 19 | 0.185 | 0.121 | 0.385 |
| LCS_GetCutDebug | 19 | 0.062 | 0.049 | 0.186 |
| LCS_GetNbrIdx | 6 | 0.473 | 0.381 | 0.580 |
| LCS_GetCutEdgeFractions | 6 | 0.526 | 0.264 | 0.764 |

| 拓扑版本变化 | 样本数 | 帧开始+cut tick+子步间重建，平均 ms |
|---|---:|---:|
| changed | 6 | 1.059 |
| unchanged | 13 | 0.562 |

接触发现/流形、材料与接触耦合迭代、cut tick 与子步间重建、Finalize 表面重建应分开看。现有计时点不提供独立的“纯形变、纯接触、纯切割、纯拓扑”四项精确分解；本轮不通过推测填补。若需要该粒度，应在隔离的原生构建中增加 CUDA/NVTX 区间并量化插桩误差。

原始 tick、拓扑版本和每个 API 数值见 profile-breakdown.json。
