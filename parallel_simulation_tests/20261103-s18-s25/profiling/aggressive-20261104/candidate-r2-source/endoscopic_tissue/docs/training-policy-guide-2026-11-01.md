# 仿真侧任务、策略与同步采集

入口为独立的 `endoscopic_tissue.training.experiment`。原有交互启动命令、默认 DLL、材料、接触、摩擦和可见效果没有改动。实验结果仍标为未认证训练域；接口能够运行不等于夹取、切割和释放的物理验收通过。

## 启动

在 `D:\Simulation\genesis-world` 中执行，输出目录必须尚不存在：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 -m endoscopic_tissue.training.experiment --spec endoscopic_tissue_backups\2026-11-01-training-completion\policy-pilot.json --output endoscopic_tissue\outputs\my-policy-pilot
```

规范使用 `tissue_experiment_v1`，明确 DLL 路径、SHA-256、用途、回合清单、进程时限与剩余磁盘预算。每回合包含 scene、preparation、seed、steps、cutting_enabled、policy，可选 bounds、goal、contact_model 和 vision。所有策略仍使用同一 11 维 SI 绝对目标及相同限幅/速率约束。

采集用 `purpose: collection`。评估清单预先保存 `excluded_training_groups`，再冻结；同场景换种子是重复性评价，不是独立组织几何评价：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 -m endoscopic_tissue.training.experiment --spec my-evaluation-draft.json --freeze --output my-frozen-evaluation.json
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 -m endoscopic_tissue.training.experiment --spec my-frozen-evaluation.json --output endoscopic_tissue\outputs\my-repeatability-evaluation
```

冻结后的内容变更会导致哈希校验失败。同一 scene/preparation/contact_model/seed 属于采集器标记的同一重置谱系，但仅改 seed 不能保证物理独立。评估报告包含全部尝试，包括准备失败、数值失败、超时和磁盘不足，不能只保留成功回合。

模型训练前用 `endoscopic_tissue.training.splits` 生成组织几何分组索引。它校验实际材料参考与已记录的几何源码，按组织形状参数分组；同一几何的不同种子、准备路径、接触模式或网格间距仍属于一组。历史 results 中的 seed 划分保留为来源记录，正式训练/验证使用索引中的 split，不能把相邻帧或同一几何随机拆散。

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 -m endoscopic_tissue.training.splits endoscopic_tissue\outputs\my-policy-pilot\results.json endoscopic_tissue\outputs\my-repeatability-evaluation\results.json --evaluation-scope repeatability --output endoscopic_tissue\outputs\my-split-index.json
```

省略 `--evaluation-scope repeatability` 时，同一物理几何同时出现在采集与评估中会被拒绝。只有一个组织几何时，工具明确返回 `has_held_out_validation=false`，不会凭空制造独立验证集。本轮薄壁百回合和冻结评估属于这种同场景重复筛查。索引不改变记录，不放宽质量门限；抽取窗口仍须用 `EpisodeReader` 校验哈希与异常边界，未认证训练域的限制仍然有效。

## 材料目标与奖励

`TaskGoal` 支持 incision、excision、hold、release。切割目标是初始网格中的双向材料边 `(node_a,node_b)`，持有或释放目标使用稳定材料节点。绑定时保存材料参考哈希；跨网格、跨分辨率复用错误节点编号会被拒绝。目标仅用于评价，不参与原生断裂决策，也不会把切口强行修直。

切断要求原材料边的两个方向均不存在。切除还要求指定材料区域与主体在材料图、包含支撑节点的图中都分离。夹取观察沿用实际夹垫的旋转几何，并在准备阶段夹爪闭合后绑定材料见证点，跟踪提拉后的保留比例、相对切向滑移及双侧载荷。纯摩擦模式为 contact_model=1，辅助模式为2；纯摩擦不能用辅助锚点数证明成功。双侧载荷门槛沿用原诊断的0.05 N，非真实组织标定值。

奖励分量分别为目标进度增量、区域外新增材料断边的负数、持有条件、负滑移（m）、负时间（s）和当前实际发送目标相对上一限速后策略目标的归一化偏差平方的负均值（含预载限制导致的目标差）。需要显式给出六项 weights；没有权重时仅返回指标，reward 为 null。示例权重是接口实验配置，不能当成已验证的任务奖励。

成功要求条件维持指定时长；区域外损伤和配置要求的掉落为正常任务失败。release 的主动开夹不自动算掉落失败。数值、观测或作用域失败优先于任务成功，reward 无效、bootstrap=false；正常时间截断可从有效末状态 bootstrap。当前保守局部域仍拒绝游离材料，所以 excision/release 的评价代码可用，但不能据此把全任务物理域标成通过。

`tasks.recompute_episode(EpisodeReader(path))` 从保存的状态、原始材料参考和执行动作重新计算，并逐步比较在线任务记录；无需重跑物理或依赖脚本阶段名。`cut_origin_bracket` 保存新断边与两条导出刃段最近点的帧前后关系；线段不相交时，最近点不能叫作交叉点。该记录不是完整曲刃及子步接触证明。

示例的33条目标材料边是用于测试评价机制的窄目标集。原生切割偏离该集合会得到任务失败，这不能单独证明现有软组织交互不真实；也没有据此改变切割路径或放宽物理质量阈值。训练时应先定义符合实际任务的材料目标及允许损伤预算。

## 策略观测与视觉

策略可选 hold、smooth_random、force_feedback、script 或 python。随机策略显式保存种子、平移/角度幅值和闭剪参数；反馈策略只使用声明的载荷观测。script 接收完整 SI 动作数组，不足时失败。python 使用模块、factory、源文件 SHA 和可选 checkpoint SHA，factory(parameters, seed=...) 返回具有 act(observation) 的新对象，每回合重新构建。

默认策略观测仅含仿真的工具位置、开合与载荷，剔除锚点数、断裂计数和内部挤压状态。显式 privileged=true 才给节点、差分速度和邻接；这些也不是包含全部内部历史的完整 Markov 状态。任务真值、质量真值与策略输入分别保存。

可选 `vision: {"width":640,"height":480,"every_steps":1}` 保存同一接受状态的 RGB、针孔深度、对齐到 RGB 的深度、有效掩码与相机外参。每张图绑定 step_id、native_tick、仿真时间。RGB 保留原有畸变与暗角；深度使用同一像素映射，采用最近邻保护深度边界。静态文件保存针孔 K、RGB→针孔采样映射和 native→Genesis 旋转；外参为 Genesis 世界米制到 OpenCV 相机。深度是相机轴向Z，像素中心遵循 Genesis 的 u+0.5/v+0.5。较低相机率明确记录未安排采样的步，不复用旧帧冒充新观测。没有自动输出真值分割。

## 失败恢复与敏感性

每个回合独立子进程，支持准备重试1至3次、硬时限、磁盘下限、心跳与每秒采样的进程内存峰值记录。超时只终止本次创建的工作进程，下一回合创建全新环境。录制继续沿用有界队列、无损保存、N个转移/N+1个状态和失败原始数据隔离。发生磁盘不足不会静默丢帧。

dt_s、substeps、spacing_scale 是显式敏感性实验配置；未设置时保持原参数。改变时间步后准备路径按相同物理时长重采样，改变网格后须重新绑定目标。所有覆盖都会在上下文中标明，不能作为默认优化被自动采用。

R20 是另一个独立原生实验包。其支撑标记缓存通过逐状态重算核对，但已有表面拓扑与性能验收失败；与上述训练软件功能是否可用分别记录，禁止替换默认 DLL。
