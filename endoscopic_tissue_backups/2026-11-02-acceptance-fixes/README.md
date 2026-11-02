# S15–S17 隔离修复与证据

起点 `a900f322efa09a06ce1cc038c9479d121028ce26`，分支 `dev`。用户要求逐项修复、保持默认仿真效果和帧率；所有验收通过后才推送 `tissue/dev`。

生产入口、三个默认 DLL、物理参数和质量阈值保持冻结。JSON 标量修复检查点为 `86509db`。本目录的候选均未获验收、未替换默认。

- `run_diagnostic.py` 自动等待其他 Python 任务结束，只监督自己创建的子进程。允许普通桌面 GPU 负载，所以其所有耗时都不是正式性能证据。
- `trace_damage.py` 对现有失败律脚本增加 API 边界日志和失败现场。同步跟踪/逐步保存会增加耗时。
- 现有 `validate_damage.py` 在已存在的严重数值或局部范围失败发生后，保存失败转移与现场并返回失败。不会删除游离材料、修改参数或把越界改成通过。此前一次 41 分钟停顿的原生调用根因尚未复现确认。
- `regrasp_trial.py` 使用原来的完整快照配对验收；新换抓轨迹向附着端移动 3 mm，以解决旧轨迹未接触组织的问题。原脚本的摩擦/辅助模式不变，辅助换抓通过也不能证明纯摩擦验收通过。
- `render_capture.py` 核对每一条录制状态与标签、生成所有帧并完整解码验证。完整视频的生成不等同于物理真实性或帧率验收。
- `cpu_tests.py` 将临时目录限定在任务输出中，并保存真实测试结果。

`overrides/boundary-snapshot2` 只将先前 R25 的切口表面分组与接触身份修复移植到 S15 快照基线，不移植其余 R19/R23 物理求解改动。两块新增数组及其主机指针/启用标记完整登记到快照；真实切边掩码仍是接触缓存的身份。材料、摩擦、起切力、挤压量及迭代预算不变。完整运行和配对验收仍在进行。

先通过 11-01 的准备脚本生成 `artifacts/training-snapshot1/source`，再运行：

```powershell
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-11-02-acceptance-fixes\build_candidate.py --name boundary-snapshot2 --base training-snapshot1
```

构建器逐文件核对固定基线和覆盖文件 SHA，只写入独立 `native/bin-acceptance-boundary-snapshot2`。`prepare_boundary_candidate.py` 保留了从旧实验源码逐段审核移植的过程；普通复现构建直接使用已保存、带哈希的覆盖文件。

运行记录在 `endoscopic_tissue/outputs/acceptance-fixes-20261102`。没有隐藏失败，也没有推送 GitHub。
