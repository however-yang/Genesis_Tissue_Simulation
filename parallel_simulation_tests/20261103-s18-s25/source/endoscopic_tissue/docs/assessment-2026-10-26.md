# 夹取、切割与真实感复测

本机 RTX 3090、Windows x64、`genesis_world` Python 环境。测试对象是本目录的 **CUDA 组织求解器 + Genesis 显示** 原型，并非 Genesis 原生机械臂与软组织的完整耦合。未改变组织参数、控制路径或原生 DLL，也未修改 Unity 工程。

结论：能演示夹住、提拉、切开及退刀后保持断开；**夹取重复性与切口几何仍有问题，不能作为稳定、逼真的内窥镜仿真验收通过**。下面的 `passed` 仅表示对应动作条件通过。

## 实际结果

数据根目录：[`outputs/recheck-20261026`](../outputs/recheck-20261026)。共执行 7 次默认 6 mm 流程、2 次 4 mm 配置；失败时在闭合阶段停止，不继续假定已夹住。

| 测试 | 结果 | 实际测量 |
|---|---|---|
| 带 Genesis 渲染的完整流程 | 动作通过，335 步 | 提拉 4.44 mm；9 条材料边断开，退刀后仍为 9 条；20 个切面三角形 |
| 同进程连续重建，第一次 | 动作通过 | 提拉 4.56 mm；9 条断边；夹持到保持阶段未丢失 |
| 同进程连续重建，第二次 | 动作通过 | 提拉 4.35 mm；9 条断边 |
| 同进程连续重建，第三次 | 失败 | 第 105 步闭合结束时捕获数量为 0，未进入切割 |
| 每次独立启动，第一次 | 动作通过 | 提拉 4.34 mm；10 条断边；20 个切面三角形 |
| 每次独立启动，第二次 | 动作通过 | 提拉 4.82 mm；9 条断边；16 个切面三角形 |
| 每次独立启动，第三次 | 动作通过 | 提拉 5.20 mm；9 条断边；20 个切面三角形 |
| 4 mm 网格，同进程重建 | 失败 | 闭合结束时未捕获组织 |
| 4 mm 网格，独立启动 | 失败 | 闭合结束时仍未捕获组织，排除了仅因之前场景重建而失败的解释 |

这不是成功率统计。独立启动三次通过不能证明重置是唯一原因，也不能证明夹取已可靠；不同运行的提拉量、断边数和切面数有差异。尚未定位这种变化来自哪个原生求解/接触/初始化环节。

## 夹取与松钳

成功的重复测试捕获 1 个表面面片，涉及 16 个被跟踪的网格节点。表中提拉量是同一批节点相对于闭合时位置的竖直位移中位数，不是器械目标位移。捕获后至保持阶段，没有检测到夹持数量降为 0。

在同进程第一次测试结束后，额外执行 30 步开钳和 30 步等待：有效夹持点归零，被跟踪节点的高度中位数下降 **2.14 mm**。因此组织确实响应夹持与释放。该样本包含 335 个基本流程步和 60 个额外松钳步。

当前夹持使用显式柔性锚点辅助，不能据此认定纯摩擦夹持、滑移或夹伤压力已经正确。4 mm 网格夹取失败说明更细离散化下的接触与捕获仍需适配。

## 切割是否真实发生

成功的评估样本在进刀前没有断边，切割阶段断开材料邻接关系；逐步检查未见已断连接恢复，也未见断边两端状态不对称。退刀后仍保留断边，并生成带切面标记的新三角面。因此变化存在于求解器的材料拓扑，不只是颜色贴图。

但这些样本切后原始内部材料节点仍构成 **1 个连通分量**。验证的是局部切开，尚未验证整块组织完全切断、游离组织分离以及分离后的碰撞。

## 切口长什么样

近景从同一次实际运行的 `blade_approach.npz` 与 `hold.npz` 重新渲染；三张图使用相同相机位置，物理状态没有修改。时间、断边数取自原始轨迹。

切前：

![切前近景](../outputs/recheck-20261026/cut-closeup/before-cut.png)

退刀后，原始切面：

![退刀后原始切面](../outputs/recheck-20261026/cut-closeup/after-cut-raw.png)

相同状态，开启目前默认的显示细分：

![退刀后显示细分](../outputs/recheck-20261026/cut-closeup/after-cut-refined.png)

原始切面可见明显的三角折面和尖片；显示细分只能平滑周边表面，不能消除这个形状。它目前不像边缘连续、受拉后自然张开的柔软组织切口。

切前导出表面为 336 个三角面，切后为 356 个，其中 20 个标记为切面。按坐标临时合并顶点后的几何检查发现切后有 5 个重合三角面、8 条邻接面数超过 2 的边；切前两项均为 0。此检查仅用于定位：切面两侧可能具有重合位置，因此这些数字不能直接解释为材料重新粘连。检查未发现零面积三角面，也尚未进行完整自相交检测。切口尖片的视觉问题已在原始几何中确认。

切面包围盒只是切面位置范围，**不是伤口深度或宽度的测量结果**。

## 改善真实感的优先顺序

1. **先修切面重建。** 检查切割前沿、两侧面片生成和重合面来源，减少尖片与错误缝合；从原始几何验收，不能只靠着色平滑遮盖。
2. **稳定夹取与初始化。** 查明重复执行的差异，检查实际接触面、捕获范围及夹持约束对网格尺寸的依赖。通过重复夹住、拉伸、松钳测试后再提高分辨率。
3. **增加切口附近物理细节。** 当前 6 mm 网格相对组织厚度较粗，显示细分不会增加物理自由度。细化或局部细化必须同时验证捕获、切割和时间步稳定性。
4. **再完善外观。** 加入更合理的组织形状、切面与表层材质差别、湿润薄膜反射及更细的器械外观；后续再评估次表面散射等光学效果。现有高分辨率贴图不能代替合理的切口几何。

这些是后续工作，本轮没有宣称已经修复。尚未测试机械臂关节控制、RCM、训练环境批量重置、精确切割力或真实组织参数。

## 复现与原始记录

在 `D:\Simulation\genesis-world` 运行，下列输出路径必须尚不存在：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.run --headless --output endoscopic_tissue\outputs\new-rendered
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.assess_interactions --output endoscopic_tissue\outputs\new-repeat
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.assess_interactions --isolated --output endoscopic_tissue\outputs\new-isolated
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.assess_interactions --single-spacing 0.4 --output endoscopic_tissue\outputs\new-fine
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.inspect_cut_surface --input endoscopic_tissue\outputs\new-rendered --output endoscopic_tissue\outputs\new-closeup
```

评估脚本为保留所有失败案例，在完成评估后正常退出；须读取 `summary.json` / `assessment.json` 的每个 `status`，不能把进程退出码 0 当成物理测试全部通过。

- [完整视频](../outputs/recheck-20261026/rendered/endoscopic_demo.mp4)、[带渲染流程记录](../outputs/recheck-20261026/rendered/report.json)。
- [同进程结果](../outputs/recheck-20261026/assessment/summary.json)、[独立启动结果](../outputs/recheck-20261026/isolated/summary.json)、[独立 4 mm 结果](../outputs/recheck-20261026/isolated-fine-4mm/assessment.json)。
- [切面几何记录](../outputs/recheck-20261026/cut-closeup/geometry.json)。
- 此次重新运行 6 项 CPU 接口/网格检查，全部通过；这不代表物理真实性已通过。
- 使用的 DLL SHA-256：`b296e863e8f47242fe49482382f4813a54ccf13212e3f2f82045f901c6ca8c52`。

原始报告保留原样：较早 `summary.json` 中 `rms_all_material_nodes_mm` 实际包含整个网格（包括外部节点），不应解释成组织内部位移 RMS；较早失败案例的 `material_components_after: 0` 表示未执行该项检查，不能解释成组织消失；较早 `steps: 335` 不含额外 60 步松钳。评估脚本已修正字段命名或未测量值。早期 `cut-inspection` 近景的 HUD 为占位计数，正式图像使用上述 `cut-closeup` 目录。
