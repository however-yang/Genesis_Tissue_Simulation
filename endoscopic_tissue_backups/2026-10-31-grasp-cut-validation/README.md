独立夹取 / 切割验证工具（2026-10-31）

本目录只包含测试与审计，不替换原求解器或默认 DLL。结论和限制见 [完整报告](D:/Simulation/genesis-world/endoscopic_tissue/docs/grasp-cut-validation-2026-10-31.md)。原始输出位于 [本地测试目录](D:/Simulation/genesis-world/endoscopic_tissue/outputs/grasp-cut-validation-20261031)，紧凑证据位于 [evidence](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-10-31-grasp-cut-validation/evidence)。

运行环境为现有 genesis_world Conda 环境、RTX 3090。CUDA 测试和视频渲染串行运行；先完成仿真，再渲染记录的原始表面。每个输出目录必须是新的，脚本拒绝覆盖既有试验。

PowerShell 示例：

```powershell
Set-Location 'D:\Simulation\genesis-world'
$qaPython = 'D:\develop\anaconda3\envs\genesis_world\python.exe'
$qaScripts = 'D:\Simulation\genesis-world\endoscopic_tissue_backups\2026-10-31-grasp-cut-validation'
$qaDll = 'D:\Simulation\genesis-world\endoscopic_tissue\native\bin-contact-groups31\LiverCudaSim.dll'
$qaOut = 'D:\Simulation\genesis-world\endoscopic_tissue\outputs\manual-grasp-central'
& $qaPython -X utf8 "$qaScripts\validate.py" --case grasp-central --output $qaOut --dll $qaDll --capture
& $qaPython -X utf8 "$qaScripts\render_review.py" $qaOut
```

复测切割时把用例改为 kidney-reference 或 wall-reference，并给每次运行一个独立的新目录。要检查保护版 DLL，把 --dll 改为 D:\Simulation\genesis-world\endoscopic_tissue\native\bin-scissors\LiverCudaSim.dll。切除松夹可能触发已知异常而提前停止；不要把该停止删除或当作通过。

| 用例 | 内容 |
|---|---|
| grasp-assisted / grasp-friction-reference | 同一肾脏咬合点、16 mm 提升，辅助锚点和纯摩擦对照 |
| grasp-gentle | 4 mm 慢速提升、保持、张爪、再次夹取 |
| grasp-offset-left / grasp-offset-right | 左右各偏移 1 mm |
| grasp-deep / grasp-deeper / grasp-shallow | 加深 1 mm / 3 mm，或变浅 1.5 mm |
| grasp-central | 向中心移 3 mm、加深 1 mm，4 mm 提升 |
| grasp-rotated / grasp-air | 旋转夹爪 10°，或空夹对照 |
| grasp-wall / grasp-flap | 未切薄壁 / 薄片纯摩擦 2 mm 提升 |
| kidney-reference | 原肾脏两刀，额外横向拉动及真正张开夹爪 |
| kidney-one-snip / kidney-cut-disabled | 只剪一刀 / 切割开关关闭的对照 |
| kidney-yaw-plus / kidney-yaw-minus | 剪刀方向与位置围绕切口旋转 ±5° |
| kidney-repeat | 已切过区域重复走两次刀路 |
| kidney-friction-cut | 不加辅助锚点的原肾脏剪切脚本 |
| wall-reference / wall-traction-reference / flap-reference | 内置三刀薄壁、牵拉薄壁、薄片场景 |

validate.py 在原质量检查失败后记录错误并继续，以观察后续撤刀/释放。越过 60 mm 活跃节点包络或其他运行异常才中止。**exit 0 和 completed=true 都只表示执行完毕，不能解释为交互通过。** 必须读取 diagnostic_errors、original_metrics 和分阶段审计。原严格阈值没有改动；空夹、禁切和只剪一刀本来就不应满足完整切除标准。

每个功能运行保存 manifest.json、report.json、逐步 trajectory.jsonl 和阶段 NPZ；捕获时另存 replay.bin。保护停止的下一步另存 failure-state.npz/json，它不在报告的已记录步数和视频里。最初 kidney-reference 发生在加入失败状态保存之前，保留截至 1246 的全部视频帧；后续方向对照、保护版和薄壁失败都保存了下一步状态。

render_review.py 不运行物理仿真，渲染每一个保存的表面；拒绝覆盖现有 MP4。视频自带“30 Hz 模拟时间回放不等于显示帧率”的标记。每段完整解码后生成 video-verification.json，另有阶段关键帧和每 0.5 s 的分镜。replay.bin 使用 Python pickle，只读取自己生成、可信的本地录制文件。

其他工具：

- time_continuous.py：相同 fixture 的无审计、无录制连续计时。切割只运行到原 hold；不是显示 FPS，也不构成质量通过。参数为 --case、--dll、--output。
- assess_grasps.py --root <输出根目录>：从保存的材料位置重新计算当前爪内材料及最后 1 s 承载/滑动；不把初始材料滑出等同于整个夹持失败。
- inspect_cuts.py <单次输出目录>：材料/辅助层连接、辅助层断边归属和导出表面分量审计。
- inspect_wall_bridges.py <薄壁输出目录>：排除左侧附着和底块后，检查目标切段内上下是否仍有材料路径。
- analyze_release.py <已录制输出目录>：用闭合表面包含关系检查真实张爪后穿入，保存投影。
- audit_failures.py <有 failure-state 的目录>：区分材料节点与辅助节点的超界。
- audit_campaign.py --root <输出根目录>：串行执行现有状态的 CPU 审计。
- summarize.py：更新本轮固定输出目录的汇总；保留 fixture_valid=false 的排除标记。
- verify_artifacts.py：本轮固定规模的完整性核验（30 次有效、23 种、4 次计时、9 段录像），检查参数/命令、逐步序号、记录数、视频 SHA、原 DLL/源码哈希及脚本语法；明确输出 simulation_acceptance_passed=false。它不是任意后续实验的通用通过判定。

所有脚本的路径基于仓库中本目录的位置；用现有环境运行。新手动试验应放到另一个输出目录，以保留本轮固定证据。小体积 evidence 已保存完整报告、独立审计、哈希与关键图；原始 NPZ、录制流和完整视频留在本地输出目录，不纳入 Git。
