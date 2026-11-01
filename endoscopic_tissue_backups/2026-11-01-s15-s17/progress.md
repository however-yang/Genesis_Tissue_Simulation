# Progress

## 2026-11-01

- 读取完整需求单；读取 brainstorming、planning-with-files、karpathy-guidelines、verification-before-completion。用户已授权具体清单，不重复请求设计批准。
- 核实无适用 AGENTS.md。冻结需求与 9 项保护基线。创建独立任务计划，保留旧任务日志。
- 阅读训练 env.py、provenance.py，定位来源记录漏洞；尚未改实现或启动 GPU 验收。
- 操作问题：不存在 endoscopic_tissue/tests（训练测试在 training 内）；cmd 下带空格 rg 模式被错误分词，改用无空格模式；Node 输出对象过长被截断，后续分块输出纯文本。

- S17a 已实现 physics.py、训练原生参数适配、全 Settings 来源对比、实验 JSON 参数入口和只读历史审计。
- CPU 56 项通过；首轮 21 个错误均来自沙箱 Temp 写入权限，使用工作区临时目录后无失败。未放宽任何测试或质量门槛。
- 原生 16 工况端点测试串行运行中（session 5541），使用固定 bin-contact-groups31。默认、空覆盖及材料/起切力/刃摩擦端点已通过准备和 8 步质量检查；后续结果见 outputs/s15-s17-20261101/parameters/summary.json。
- 124 个现存本地 context 来源审计无标记差异。外部 WAM NPZ 缺完整 context，不能以 0 个记录冒充审计通过；单列缺失情况。

- 16/16 参数原生准备及随后 8 步质量通过；完整入口旧/新 ABBA 与四份完整录像由 compare_default.py 安排，session 76697。用户已授权今后自行等待 GPU 空闲，不再发问。开始时出现新 t06_branch_runner.py PID 64048，监督器自动等候。
- 导出 fbcaf71 + contact-groups31 原始三文件覆盖和分组头，形成只读 native-baseline（33 文件）；尚未编译或修改新原生实现。不是 R25 候选，不携带之前被拒绝的修复。

- S17a 本地检查点 1d3e518 已创建，未推送；独立 baseline 与 training-state1 构建通过。训练 DLL SHA 980db88d3d1fffb2401e09cff2c9070530b7ade56607e49e1decc27a8bec00bd。
- S16 实现可选实际统一夹持力 EMA/过载脱离、逐边撕裂原因及材料拉伸遥测；新增 6 项 CPU 测试，总计 62 项通过，GPU 物理与性能尚未验证。旧统一路径从不调用 k_CoreBreakCheck，不能把旧传参通过当成断裂功能通过。
- S15 枚举工具找到 265 处 cudaMalloc 和 414 个存储候选；仍是待审核清单，不是状态完整性证明。
- GPU 对照监督器持续自动排队；20:27 UTC 用户 t06_branch_runner.py PID 38424 仍在运行，不操作该进程。

- S15 独立 training-snapshot1 编译通过；当前 SHA 656b9f78e26d51780c0a3dae9a43c95412c44e905ddfd0c940a27151951eed39。所有存活 CUDA 分配保守保存，显式处理 UnifiedState、CoupledTool 和多重网格指针字段；图执行资源恢复时失效重建。原生 T1/T2/T3/T5/T8 脚本已准备，GPU 尚未运行。
- Python 不可变压缩快照、无 pickle 编码、分支回合/重置谱系、物理/ABI/版本绑定、质量基线回退、同步视觉重新渲染接口已实现。71 项 CPU 测试通过；新增测试覆盖损坏文件拒绝、父子回合隔离、评估数据泄漏拒绝、RNG 与拓扑缓存恢复。
- 检查时发现对照脚本动作域把闭合量下界写成 -0.2，启动前修复为 [0,1]；未放宽任何物理或质量要求。

- 77 项 CPU 测试通过；新增快照任务历史的离线奖励复算测试，防止分支重新初始化任务基线而改变奖励。参数化几何参考网格和全部原生动作值相同。
- 修正“夹持附近”观测范围：先保留膨胀邻域再做物理豁免环的切缘剔除；不改变撕裂判据。状态清单更新为 266 分配 / 415 存储候选。
- 旧自有等待进程 20940 已做归属/无子进程/无证据核对后替换；新 GPU 队列 session 72588 自动等候用户 t06_branch_runner.py。没有结束用户任务。
