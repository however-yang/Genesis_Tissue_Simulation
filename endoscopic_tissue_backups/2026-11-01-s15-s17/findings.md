# Findings

- 当前基线 b3d301c，dev 比 tissue/dev 领先 1 个提交。所有已跟踪文件干净；大量历史未跟踪产物留在原位，不纳入新提交。
- 9 项默认保护文件全部匹配前轮 machine.json，见 baseline.json。
- NativeEnvironment.reset 仅按 contact_model/dt/substeps/spacing 判断 physical_defaults_changed；且 substeps 比较依赖可被修改的 scissors_settings()。需要冻结原始逐场景 Settings，并比较全字段。
- NativeTissue.initialize 固定创建 CutDesc 和 GraspParams。应在训练独立适配器中注入参数，不改保护的 native_backend.py；需核对 LCS_SetContactModel 与 GraspParams.friction 的实际生效关系。
- 原生已支持拉伸撕裂和锚点断裂；逐步空间归属与 coreForceEMA 尚需验证可观测性。
- S15 不能只保存导出的节点状态或重放动作前缀，必须盘点原生跨步数据和 Python 缓存/监视器。

## 新发现

- 实际纯摩擦路径由 LCS_SetContactModel 的 muStatic/muDynamic 控制；仅改 GraspParams.friction 无效。训练覆盖同步设置实际系数，保持原动态/静态比例 0.4/0.6，并明确记录两者。
- 当前 unifiedSubstep 在 xpbd_substep 提前返回，旧 k_CoreBreakCheck 位于后续旧路径。统一求解器使用 GripPoint 而非旧角点 pin；S16b 不能只通过旧参数冒充功能，需独立原生扩展和新验收。
- 外部 E:/datasets/可微世界模型_试采/precheck-01 现存 7 个 NPZ 只有 scal/flags/heights/pos/quality_ok，无 context。summary.json 确有 context_start_force_n=0.4，但未保存 physical_defaults_changed，无法直接重审该历史布尔值；只读审计保留缺失证据。
- 2026-11-01 19:55:41 UTC 起出现外部 t06_branch_runner.py PID 23012；19:58:09 UTC GPU 89%/6125 MiB。参数端点可作功能证据，受影响耗时不得作为性能对照。已询问用户方便时暂停，不操作其进程。
