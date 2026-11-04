# 独立原生训练数据接口（实验性）

本模块按用户提供的训练前清单实现基础入口，不改变`run.py`、原生求解器或默认DLL。它不是已经通过物理验收的训练环境；纯摩擦再次夹取、游离块松夹碰撞、薄壁漏剪和真实曲刃对应仍有未通过项。当前示例为**辅助夹持后的局部动作**。

## 试采

在仓库根目录执行，输出目录必须全新；安装依赖沿用项目环境中的NumPy、SciPy、zstandard和threadpoolctl，不会导入Genesis或创建渲染器：

```powershell
& D:/develop/anaconda3/envs/genesis_world/python.exe -X utf8 -m endoscopic_tissue.training.collect `
  --dll endoscopic_tissue/native/bin-contact-groups31/LiverCudaSim.dll `
  --dll-sha256 4515f14aa15309162c2c54b6370792e128e63a7e8d3054c506da19e3aeb37d2f `
  --output endoscopic_tissue/outputs/my-training-pilot-001 `
  --scene wall-traction --preparation cut_ready --episodes 4 --steps 120 --seed 0
```

四类采样依次为保持、夹爪小幅移动、剪刀移动/闭合和联合动作。局部位置边界为准备目标周围±0.5毫米，位置采样最大偏移不超过0.3毫米，逐坐标最大目标速度2毫米/秒；角度边界和速率明确记录，但当前采样器不扰动角度。辅助夹爪闭合度保持准备值。该范围只是筛查范围，不能自动成为已验证训练动作域。`--reset-attempts`限制为1至3，每次失败均留档，不无声重试到成功。

切割许可由每回合环境配置固定；真正断裂仍由原生闭合、双侧载荷和压缩行程决定。标准11维动作无碰撞开关。关闭切割的回合中即便没有发生切断也不是失败；任务奖励、成功/失败规则尚未定义。

## 接口和数据

- `NativeEnvironment.reset(seed, scene_config, goal)`销毁并重建原生会话，执行原场景准备脚本，返回`State, context`。`settled`到重力稳定，`held`到夹持提拉，`cut_ready`再到第一次剪刀进入。`wall`的第一次进刀早于夹取，故不允许用它构造`cut_ready`。
- `step(action)`接收11维绝对目标：夹爪xyz米、闭合度、剪刀xyz米、pitch/roll/yaw弧度、剪刀闭合度。每次一个原生dt。坐标沿用原生世界轴；欧拉顺序由现有`scissors_frame`定义，夹爪方向固定。
- `StepResult`分别保存原始策略目标、位置限幅、速率限制、原生预载限制后的发送命令及前后实际状态。位置、速度和工具坐标采用SI，标为`native`的审计字段明确保留厘米/度。
- 节点ID固定，连接和断边分数按拓扑版本保存。速度是已接受位置的差分估计，**不是内部速度**；摩擦/断裂历史不完整，因此观测不是完整Markov状态。
- 正常N步保存N转移和N+1状态。有界后台队列先复制数据，再压缩写入并校验；记录阻塞不能静默丢样本。失败记录有明确终止原因；原生或导出失败不伪造下一状态，并保存能读取到的原始节点现场。
- 完整质量检查包括有限值、无效四面体、工作区、速度异常、断边不可恢复、禁切期间断边、材料连通性、剪刀刀体/腕部重叠和表面拓扑。每个状态列出未检查内容。游离块触发“超出局部任务范围”，并不代表S11已修复。
- `EpisodeReader`检查哈希、配置、回合、连续步号和时间对应，拒绝未完成写入及跨异常窗口。所有当前记录的`training_domain_approved`均为false。探索性读取必须显式选择：

```python
from endoscopic_tissue.training.data import EpisodeReader

reader = EpisodeReader("endoscopic_tissue/outputs/my-training-pilot-001/episode-0000")
for window in reader.windows(4, allow_unvalidated_scope=True):
    # 四个动作和五个相邻观测，仅用于未认证数据的检查。
    assert len(window["states"]) == len(window["transitions"]) + 1
```

窗口读取复用相邻观测和拓扑版本，缓存仅保留当前窗口；返回数组只读，修改前须显式复制。已在117个四动作窗口上核对完整状态内容哈希，解压次数由1170降到134，读取加校验约快2.7倍。这是离线数据读取收益，与物理帧率无关。

原有完整脚本的回放需要`diagnostic_continue=True`和`replay_command`，明确标为诊断，不能冒充标准策略动作。读取这类数据还需要`allow_diagnostic=True`，无效物理状态依然被排除。

## 性能和验收边界

状态检查、表面导出、快照、异步入队和后台写入分别计时。完整表面几何审计目前成本较高，采集路径不满足实时性保证；不要将其替换现有交互入口。30Hz回放也不是实际计算帧率。

固定DLL哈希和ABI能防止误载版本，但当前预编译优化DLL与可重建源码的逐位绑定尚未确认，manifest明确标为`unverified_prebuilt_binary`，不宣称构建可复现。完整验收、视频及未完成清单见[本轮记录](D:/Simulation/genesis-world/endoscopic_tissue/docs/training-foundation-results-2026-11-01.md)。
