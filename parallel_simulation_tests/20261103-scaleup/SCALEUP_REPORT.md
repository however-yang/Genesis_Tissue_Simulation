# 32 / 64 环境实际扩容结果

用户在上一轮后授权越过保守预算，实际启动 32 与 64 个进程；本轮没有停在显存预测阶段。正式求解器和验收规则未修改。

| 配置 | 完整通过 | 总 steps/s（含失败回合成本） | 合格回合 steps/s | 合格 episodes/h | 显存峰值 GiB |
|---|---:|---:|---:|---:|---:|
| 32 worker / 32 CPU，私有 MPS | 31/32 | 71.67 | 69.43 | 173.58 | 57.84 |
| 64 worker / 32 CPU，私有 MPS | 初始化/准备阶段失败 | 未测 | 未测 | 0 | 75.18 |

32 环境运行 642.9 秒，平均单环境 2.266 steps/s，GPU 利用率均值 96.4%。第 14 号 worker 在第三剪 tick 1124 出现一条 positional_nonmanifold_edges，只有该步质量失败；非法四面体、工具穿入和范围检查均未失败。该回合不计入合格产出，不能将 32 环境称为完整物理验收通过。

64 个 worker 均通过 Python 导入屏障并实际进入原生初始化。16 个进程报告 LCS_Init -46，MPS 服务端直接记录最多 48 个客户端；另 14 个进程报告 LCS_Step -2（CUDA 显存分配失败）。34 个进程完成各自短 settle，不代表 64 环境共同成功运行。初始化时既有 MPS 连接数量限制，也遇到了显存容量限制。

CUDA 状态码核对本机 targets/x86_64-linux/include/driver_types.h；MPS 限制以本次 server.log 为证。NVIDIA [MPS 文档](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)亦说明客户端上限及版本差异，不能把新驱动限制套用本机。未升级驱动或修改 MPS 全局配置。

32/64 批次的所有进程及私有 MPS 均已退出，GPU 随后回到 15 MiB；两次单环境 reset/settle 通过。没有 GPU reset，没有终止其他任务。

上一轮 16 环境 / 16 CPU 的 57.07 steps/s 对比本轮 32 环境原始 71.67 steps/s，仅是共享服务器上的观测，非严格配对因果比较。本轮合格回合折算为 69.43 steps/s。32 环境增量收益有限且已暴露拓扑失败，优先完成清单中的物理修复，再签署更大规模稳定性。

原始证据：runs/m32a、runs/m64a、runs/post64-reset、mps/m64a/l/server.log、scaleup-summary.json。
