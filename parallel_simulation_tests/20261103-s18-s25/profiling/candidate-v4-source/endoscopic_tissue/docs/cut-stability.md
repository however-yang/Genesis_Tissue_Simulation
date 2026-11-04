# 单刀片切割稳定性记录

2026-10-26，Windows / RTX 3090。修改仅在 endoscopic_tissue 内；原 Unity 项目未修改。

## 修复内容

- 刀刃切割扫掠移到有限胶囊的真实前缘。旧中心线扫掠落后接触面一个刃口半径，刀背会先顶住未切开的组织。
- 默认材料网格改为横向 2 mm、切入方向 0.5 mm，共 28,875 个网格节点。邻接静长、四面体集总质量和重建限位使用真实各轴尺寸。保留粗网格诊断，不声称任意分辨率都可稳定切割。
- 刀背两端各退让 2 mm，让有限刃口先进入组织。显示和碰撞使用同一形状，没有只缩小碰撞体。
- 切面前缘两侧仍属于同一连通分量的特征点不产生内部重复面。四边形有重合特征点时，仅输出非零面积半面。此规则改变实际重建和接触几何；不是导出视频后的图像过滤。
- 退刀从求解后的真实刀位开始，避免先追赶旧切入目标。单刃反向运动进入退刀状态，组织内部子步停顿不再清空该状态。
- 增加逐帧切面退化、退刀新增断边检查，保留刀背重叠 0.2 mm 和节点位移 60 mm 的原验收门槛。

## 诊断证据

数据保存在 outputs/cut-stability。fixed-baseline 的 9 条断边全部在空气区，没有真实切面。moving-baseline 出现外层单节点和三节点细链脱离；内部组织位移仍正常。它们的远距离运动包含无支撑后的自由落体，不能仅凭大坐标称为数值爆炸。

固定组织对照还揭示了切入深度分辨率和刀背端点的离散卡刀。在全部节点固定的极限下，0.5 mm 网格仍可能卡住；当前通过的对象是可变形、被夹持拉伸的组织。未使用节点冻结、删除远处表面或放宽验收来掩盖问题。

第一轮 verified-sequence-01 通过，但 verified-sequence-02 和 final-video 捕获了子步暂停导致退刀重新触发切割的间歇错误，后两者均正确标记 failed。保留这些历史数据，最终构建会另存验收与录像，不覆盖失败结果。

## 最终验收

最终 DLL SHA-256：`81cf6d51dbff7d31ab8fed242610e15f61be7e72220109d8c8dce263a199f56b`。两次独立进程回归和一次实际 Genesis 录像均完成 565 步、通过当前验收。两次回归另加 60 步松钳测试。

| 运行 | 状态 | 提拉 mm | 切割末尾 → 退刀后断边 | 最终内壁三角面 | 退化内壁 | 最大刀背重叠 mm | 最大节点位移 mm |
|---|---|---:|---|---:|---:|---:|---:|
| baseline-1 | passed | 6.343 | 105 → 105 | 346 | 0 | 0.0068 | 8.851 |
| baseline-2 | passed | 6.550 | 103 → 103 | 344 | 0 | 0.0000 | 8.736 |
| validated-video | passed | 6.559 | 107 → 107 | 360 | 0 | 0.0000 | 8.714 |

两次原生回归逐帧检查内壁，录像在阶段检查点检查内壁；三次均无退刀新增断边，无夹持丢失。退刀后原断边全部保留。两次松钳后夹持点均为 0，追踪材料分别回落 3.115、3.151 mm。

保存快照的独立网格检查：切前、切后、退刀、保持、松钳阶段均无孤立活动节点；按位置焊接作诊断后，边界边、非流形边、重复面和零面积三角面均为 0。这只检查导出网格，不对求解器实施焊接。录像近景检查同样为 0 个边界边、非流形边、重复面与退化面。切口内壁是实际重建几何。

11 项 CPU 检查通过，包含全部 4096 连通掩码、各轴静长/质量、显示接触几何一致性、退刀起点及失败验收。knife-back-regression 的刀背阻挡测试通过，测得刀背重叠 0、切割压力 0、断边 0。

两次回归最终节点 RMS 差异约 0.177 mm、最大差异约 1.181 mm，捕获面片和切割边数也会变化；本实现不具备跨运行逐位确定性。原生回归墙钟约 81、87 秒，录像约 174 秒，均对应 18.83 秒仿真（回归另含 2 秒松钳）；运行期间有并发 GPU 任务，这不是独立实时性能基准。

- [完整 H.264 视频：1280×960、15 fps、18.8 秒](../outputs/cut-stability/validated-video/endoscopic_demo_h264.mp4)。282 帧已全部解码核对；仅转码，不改动物理动作。
- [同次录像的验收报告](../outputs/cut-stability/validated-video/report.json)
- [两次独立回归汇总](../outputs/cut-stability/final-regression/summary.json)
- [退刀后切口近景](../outputs/cut-stability/cut-closeup/after-cut-refined.png)、[原始物理面近景](../outputs/cut-stability/cut-closeup/after-cut-raw.png)、[网格检查](../outputs/cut-stability/cut-closeup/geometry.json)

约 3.5–5.5 秒为拉伸，8.5–11.8 秒为切割，11.8–17.8 秒为退刀。切口保持张开，有深色内壁。外观仍为块状组织、简化工具和贴图材质，不能称为照片级真实。

## 复现

在 D:\Simulation\genesis-world 运行：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -m unittest endoscopic_tissue.test_core -v
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.assess_interactions --isolated --repeats 2 --release --output endoscopic_tissue/outputs/new-validation
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.run --headless --render-every 2 --output endoscopic_tissue/outputs/new-video
```

## 适用范围

只验证这一块状组织、固定夹取点和一次单刀片切入/退刀。并非任意刀向、重复进刀或网格相位的普遍稳定性证明。仍为按断边停用有限元单元的嵌入网格方法，没有局部虚拟节点或体网格重划分；细小游离薄片、组织自碰撞和断裂能未校准。

当前仍是器械末端控制与 Genesis 显示，夹持使用柔性锚点。组织光学、真实切割力、机械臂关节动力学、RCM、策略训练和临床有效性未包含在本轮验收。
