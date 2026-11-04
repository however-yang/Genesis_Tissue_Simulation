# 切缘核查与训练基础模块阶段记录

依据用户的[训练前功能与优化清单](D:/Desktop/BRACE/WAM_世界动作模型_2026-10-28/15_仿真器训练功能清单_2026-10-31/仿真器训练前功能与优化清单.md)，在dev分支增加独立训练数据接口。保护起点为d429ae6，默认优化DLL固定为SHA-256 `4515f14aa15309162c2c54b6370792e128e63a7e8d3054c506da19e3aeb37d2f`。本轮不替换默认交互入口、DLL或物理算法；接口数据不能自动获得物理真实性或训练准入资格。

## 切缘复核结论

没有发现把最终伤口统一拉成一条世界坐标直线的步骤。当前切面随实际剪刀位姿定位，切割查询当前已变形组织，已切位置随后随组织运动。原R13快照的64个已设置列包含34个不同材料切割层，跨度2.5391毫米，并非一个固定切割高度。

但是每侧弯曲刃缘仍被拟合成一条直线，局部覆盖仍含半网格容差，材料切面仍采用固定主轴的列表示。双侧载荷是整把剪刀的触发条件，候选点则判断任一拟合刃缘扫掠覆盖，尚未逐点证明两个真实切缘和同一局部组织的配对。因此“逐点严格按真实曲刃剪切”仍未通过；不能以不规则外观替代该项验证。完整证据和未解决问题见[切缘审计](D:/Simulation/genesis-world/endoscopic_tissue/docs/cutting-edge-review-2026-11-01.md)。

## 已实现的独立入口

新增`endoscopic_tissue/training/`：固定DLL内容/ABI、无渲染reset/step、SI动作边界与速率限制、实际位姿与发送命令分离、连续状态/转移、有界异步压缩记录、异常隔离、带哈希与时序校验的窗口读取、小批四类动作采样。用法和限制见[README](D:/Simulation/genesis-world/endoscopic_tissue/training/README.md)，设计见[设计记录](D:/Simulation/genesis-world/endoscopic_tissue/docs/training-foundation-design-2026-11-01.md)。

状态速度为位置差分估计，内部摩擦、挤压和求解历史未完全导出；不能称为完整Markov状态。正常N动作保存N转移/N+1状态。原生或观测导出失败保留最后已接受状态和能取得的原始节点现场，不伪造成功的下一状态。记录失败回合默认不能提供训练窗口。全部当前数据的训练准入标记为false。

## 验证证据

CPU检查：41项通过，包括原有20项和21项动作/记录/故障测试。测试覆盖单位及原生浮点载荷往返、限幅限速、DLL错误拒绝、标准模式禁止诊断绕过、异步快照所有权、N/N+1对应、拓扑版本、跨回合/异常/时间错位拒绝、写入错误、导出失败、退化网格拒绝和缓存窗口只读。测试入口为[run_contract_tests.py](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/run_contract_tests.py)，结果见[cpu-test-result.json](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/cpu-test-result.json)。

真实试采使用wall-traction、辅助夹持及cut_ready准备，30,417材料节点/38,425活跃节点。4回合各120动作/121状态，480条转移和484个状态全部完成哈希与时序读回。保持、夹爪、剪刀、联合动作的新断边分别为0、0、51、55；最大刀体重叠分别为0、0、0.0331、0.0266毫米。已实现的有限值、材料体积、表面拓扑等检查未报错，未实现的真实性和任务指标不能据此判定通过。见[试采摘要](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/pilot-summary.json)和[完整试采索引](D:/Simulation/genesis-world/endoscopic_tissue/outputs/training-foundation-20261101/pilot-01/pilot.json)。

试采中原生step均值为31.94–38.01毫秒，含完整表面检查和状态快照为78.72–89.60毫秒。准备耗时34.94–37.19秒，每回合采集加落盘校验17.30–18.95秒。这是检查/数据入口的成本，不是实时交互性能；默认路径不调用这些模块。该小试采也不是超过10万节点接近33Hz的证明。

原生切后重置已实际执行：同一适配对象先稳定、重置到夹持进刀、切断52条连接，再重置到稳定状态。重置后的断边和夹持点为0，邻接恢复到初始值，回合ID不同，时钟恢复0、原生准备计数30；重复稳定状态的最大位置差约0.00000561毫米，不承诺CUDA逐位一致。另用不合法准备动作范围触发重置失败，检查旧公开状态已清空、原始观测现场仍保留。见[重置结果](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/native-reset-result.json)。

离线读取单独优化了相邻窗口缓存，未改变数据或物理过程。117个四动作/五状态窗口的完整元数据和数组哈希四次一致；按普通/缓存/缓存/普通顺序测量，普通读取为5.727、5.756秒，缓存为2.157、2.149秒，约快2.67倍，解压次数1170→134。两组都包含完整内容哈希计算，不能当作神经网络训练吞吐。见[读取对照](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/reader-benchmark.json)。

## 完整夹取剪切回放及限制

三组均完成同一wall脚本中首次进刀、三次剪切、夹取提拉、撤刀和保持。只省略录像前共同的30步重力准备，每段1155计算步加初始帧，共1156帧；三段全部编码并重新解码，共3468帧。视觉检查为每段18个对应时刻及提拉/末帧局部放大，不冒称逐帧视觉认证。`release`仍是剪刀张开，片尾夹爪仍有73个辅助夹持点，不能用该视频证明摩擦松夹或换抓。

| 运行 | 原生LCS_Step均值 ms | NativeTissue.step均值 ms | 含表面导出的流水均值 ms | 完整检查步均值 ms | 最终断边 |
|---|---:|---:|---:|---:|---:|
| 原有入口A（legacy-wall-02） | 55.177 | 61.396 | 62.750 | 111.835 | 85 |
| 适配层（adapter-wall-01） | 58.310 | 64.408 | 65.796 | 115.163 | 82 |
| 原有入口B（legacy-wall-03） | 37.026 | 40.656 | 41.338 | 64.175 | 76 |

这些是带诊断和采集间隔的串行轨迹测量，native计时排除录像压缩/落盘；不是默认界面的显示FPS。重复基线的时间变化很大，目前**不能作“帧率不回退”的验收**。没有引入新的材料/接触算法，不能宣称这些结果实现了物理解算提速或超过10万节点接近33Hz。

所有帧的节点位置和连接哈希均做了对照。适配层对A的材料节点最大位置差3.564毫米、最大全材料帧RMS差0.315毫米；B对A分别为6.974和0.770毫米。活跃节点（包含支撑）最大差分别为9.240和7.906毫米，不能以材料RMS较小忽略支撑节点差异。切断位置/数量和末帧切口也有可见变化，**严格物理等价性尚未通过**。记录在[适配对照](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/paired-adapter.json)及[基线自身重复](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/paired-repeat.json)。

三组执行同一脚本，撤刀按照各自已接受的实际位姿起步，故有119帧发送目标存在差异；这不是逐比特相同发送命令的实验。两次原有入口本身也产生不同拓扑，不能把差异直接归因于适配层，更不能把它全部当作允许误差。基线B在第二剪有6步刀体重叠超过0.2毫米，最大0.2093毫米；该失败原样保留。A和适配层未触发已实现检查，仍不能据此证明切口无残连或真实曲刃对应。

观察到组织在夹取后局部提拉、剪切时局部受压，撤刀后切口保留分段弯折；没有把末端轮廓统一拉直。第二、第三剪后，基线重复之间也能看到切口宽度和边界差异。这支持继续保留自然不规则轮廓的要求，同时暴露出切割重复性需要单列验证，不能以外观相似直接放行。

![对应提拉和保持阶段：左为基线A，中为适配层，右为基线B](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/review/cut-detail.jpg)

完整视频：[基线A](D:/Simulation/genesis-world/endoscopic_tissue/outputs/training-foundation-20261101/legacy-wall-02/review/interactions-h264.mp4)、[适配层](D:/Simulation/genesis-world/endoscopic_tissue/outputs/training-foundation-20261101/adapter-wall-01/review/interactions-h264.mp4)、[基线B](D:/Simulation/genesis-world/endoscopic_tissue/outputs/training-foundation-20261101/legacy-wall-03/review/interactions-h264.mp4)。全部帧校验和抽帧范围见[视频记录](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/video-review-index.json)。本次适配回放的Python源码已单独归档，与该次manifest中的35项源码/资产哈希全部相符；之后只加强重置失败/退化网格检查及离线读取，未变更物理调用。旧优化DLL与源码的构建绑定仍未确认。

初次legacy-wall-01因诊断归档类缺少上下文管理接口在准备后停止，没有执行交互；保留该目录，修正诊断脚本后使用全新legacy-wall-02目录，未修改求解器。最终9个默认保护文件的SHA-256均与任务开始一致，见[保护检查](D:/Simulation/genesis-world/endoscopic_tissue_backups/2026-11-01-training-foundation/protected-files-after.json)。采集入口保持显式实验模式；默认版本没有采用或替换。

## 清单状态（不是全清单验收）

| 项目 | 本轮状态与剩余限制 |
|---|---|
| S01 原生入口/版本 | DLL哈希、ABI、资产/适配代码哈希和headless入口已实现；旧优化DLL与可重建源码的绑定未确认，明确记录为独立预编译基线。 |
| S02 重置/准备 | 独立会话重建、固定准备、有限次数失败留档已实现；整体动作域的充分重复和初始化成功率仍待验证。 |
| S03 动作 | 11维SI绝对目标、逐坐标限幅限速、原始/发送/实际区分已实现；不是任意方向和全工作区物理验收。 |
| S04 状态 | 节点ID、材料/支撑/固定掩码、位置、估计速度、工具/载荷、邻接/断边分数已记录；内部速度、完整接触/断裂历史未导出。 |
| S05 连续数据 | 初始状态、连续转移、拓扑版本、压缩写入、读回窗口和失败隔离已实现；大规模存储与长时间压力测试待做。 |
| S06 质量隔离 | 已实现检查按原阈值执行；曲刃逐点对应、求解残差、游离块连续碰撞及任务专用残连验收未完成。范围拦截不等于物理修复。 |
| S07 有效动作域/材料 | 未验收。4回合仅验证链路，不能替代清单建议的充分重复、扰动范围、敏感性和材料标定。 |
| S08 奖励/任务终止 | 未实现学习奖励与目标专用成功判据；reward/task_success/task_failure为null，数值失败与时间上限单独记录。 |
| S09 自动采样 | 基础四类局部采样、seed/版本/完整回合分组已实现；角度扰动、多样化场景、边界覆盖与充分样本量未完成。 |
| S10 摩擦夹取/换抓 | 继续未通过。此次采用已有辅助夹持模式，不用它宣称纯摩擦问题修复。 |
| S11 切后自由释放 | 继续未通过。R19仍独立保留，未替换默认；局部采集排除游离块不代表解决此项。 |
| S12 视觉同步 | 完整诊断视频用于比较；训练相机同步/标定接口未实现。 |
| S13 效率/存储 | 记录成本分项计时，拓扑按版本保存且压缩异步写入；未达到实时采集或大规模吞吐验收。 |
| S14 机器人接口 | 未实现；需要真实控制频率、工作区和接口条件后另行验证。 |

所有原有物理失败继续保留。后续物理改动必须单独试验、与基线重复和完整交互视频对照；没有通过前不得替换默认，也不得靠放宽质量阈值或改变切口外观获得通过。
