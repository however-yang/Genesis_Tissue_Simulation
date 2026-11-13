# 演示接口说明

这份说明记录内窥镜软组织演示里可以替换的接口：固定方式、力学信息、场景调用、组织模型和周围环境。坐标除镜头以外都是厘米，Y 朝上。镜头 `shot` 使用 Genesis 米制。

肝脏演示的运行方式：

```powershell
py -m endoscopic_tissue.run --scene liver
```

空格暂停，Q 或 Esc 退出。`--headless` 只存画面。`--steps 0 --headless` 只画开始时的一帧，不代表完整的夹取和剪切已经通过验收。

## 场景调用接口

场景表在 `endoscopic_tissue/scenario.py` 的 `SCISSORS_SCENES`。命令行 `--scene` 的可选项就是这张表的键。

| 键 | 内容 |
| --- | --- |
| `wall` | 默认。厚块上的薄壁，沿壁剪开 |
| `wall-traction` | 同一道壁，剪切前先夹住提起 |
| `flap` | 悬空薄瓣，从自由端剪到根部 |
| `kidney` | 肾脏外生肿瘤切除 |
| `liver` | 腹膜腔里的肝脏，夹住一侧后竖着剪三刀 |

```powershell
py -m endoscopic_tissue.run --scene wall
py -m endoscopic_tissue.run --scene kidney
py -m endoscopic_tissue.run --scene liver --headless --steps 0 --output endoscopic_tissue/outputs/liver-still
```

`--tool knife` 走单刀片流程，不读取 `--scene`。`--output` 必须是还不存在的目录。

一个剪刀场景是 `ScissorsScene`。和替换模型、房间有关的字段：

| 字段 | 作用 |
| --- | --- |
| `shape` | 组织模型。空着时用方块组织 |
| `environment` | 周围环境。空着时用该场景原来的背景 |
| `stages` | 夹子和剪刀的路径 |
| `tab`, `plane`, `plane_axis`, `cut_low`, `cut_high` | 切口位置和验收范围 |
| `shot` | 镜头位置和注视点，Genesis 米制 |
| `young_pa`, `spacing`, `jaw_axis`, `jaw_normal`, `jaw_open_gap`, `jaw_stiffness` | 材料、网格和夹子 |

登记一个新场景后，`--scene` 就能选到它：

```python
from dataclasses import replace

from endoscopic_tissue.scenario import LIVER_SCENE, SCISSORS_SCENES

SCISSORS_SCENES["my-scene"] = replace(LIVER_SCENE, environment=room)
```

```powershell
py -m endoscopic_tissue.run --scene my-scene
```

`stages` 决定器械走向哪里。只换 `shape` 而留下肝脏的 `stages` 时，夹子和剪刀仍走向肝脏原来的位置。换组织模型时要同时重写 `stages`、切口盒子和 `shot`。

## 组织模型接口

协议在 `endoscopic_tissue/tissue.py` 的 `TissueModel`。现成实现：

- `endoscopic_tissue/liver.py` 的 `Liver`，由导入的四面体网格做成距离场
- `endoscopic_tissue/kidney.py` 的 `Kidney`，椭球肾脏加外生肿瘤

求解器用它长出格子。画面用 `surface_kind` 选表面贴图。

| 成员 | 作用 |
| --- | --- |
| `low`, `high` | 包围盒，厘米 |
| `distance(points)` | 有符号距离，内部为负，`points` 形状 `(n, 3)` |
| `inside(positions)` | 哪些结点在组织里 |
| `boundary(positions, a, b, inside)` | 内外结点之间的表面。可交给 `implicit_boundary` |
| `bed(positions)` | 哪些结点保持钉住。肝脏是韧带斑，肾脏是埋在脂肪里的下部 |
| `surface_points()` | 表面上的点。腹膜腔用它把横膈膜放在组织远侧 |
| `surface_kind` | `"fat"`、`"liver"` 或 `"kidney"` |
| `texture_cm` | 表面贴图重复一次的宽度，厘米 |

把新模型放进当前腹膜腔：

```python
from dataclasses import replace

from endoscopic_tissue.environment import PERITONEAL_CAVITY
from endoscopic_tissue.scenario import LIVER_SCENE, SCISSORS_SCENES

organ = ...  # 满足 TissueModel 的对象
SCISSORS_SCENES["organ"] = replace(
    LIVER_SCENE,
    shape=organ,
    environment=PERITONEAL_CAVITY,
    # stages、tab、plane、cut_low、cut_high、shot 按这个模型重写
)
```

肝脏自己的操作路径由 `liver_cut` 根据肝脏形状算出，再写成 `liver_stages`。新模型可以按同样的方式从 `distance` 里找夹持点和切口。

## 环境接口

协议在 `endoscopic_tissue/environment.py` 的 `SurgicalEnvironment`。现成房间是同一个文件里的 `PERITONEAL_CAVITY`：椭圆体腔、横膈膜、四块脂肪、视野和两盏灯。这些只进入画面，不进入求解器，所以同一套夹取和剪切可以换房间。

| 成员 | 作用 |
| --- | --- |
| `fov` | 垂直视野，度。越小，组织在画面里越大。腹膜腔是 56 |
| `ambient` | 环境光 RGB |
| `lamp_scale` | 内窥镜灯亮度倍数。腹膜腔是 4 |
| `overhead(tissue)` | 固定灯在 Genesis 米制下的位置，以及让组织中心达到一盏满灯亮度的距离平方。返回 `None` 时灯只留在镜头上 |
| `populate(view, tissue)` | 场景建好后加入体腔和脂肪 |

有固定灯时，镜头灯和固定灯各用 `lamp_scale` 的一半。`populate` 里用 `view.add_mesh(mesh, material)` 加网格，顶点已经是 Genesis 米制。`view.shot` 是镜头位置和注视点。`from_genesis` 把它换回厘米。

只改当前房间的亮度、视野或脂肪时，复制 `PERITONEAL_CAVITY`：

```python
from dataclasses import replace

from endoscopic_tissue.environment import PERITONEAL_CAVITY, Pad
from endoscopic_tissue.scenario import LIVER_SCENE, SCISSORS_SCENES

room = replace(
    PERITONEAL_CAVITY,
    fov=50.0,
    lamp_scale=3.0,
    ambient=(0.08, 0.03, 0.03),
    pads=(
        Pad(centre=(0.0, -4.0, 10.0), half_extents=(6.0, 2.0, 4.0), is_cavity_coloured=True),
    ),
)
SCISSORS_SCENES["liver-room"] = replace(LIVER_SCENE, environment=room)
```

`is_cavity_coloured` 为真时脂肪用体腔照片，否则用肝脏表面。

体腔开口 `z_max` 变了以后，镜头要按新腔重算：

```python
import numpy as np

from endoscopic_tissue.grid import to_genesis
from endoscopic_tissue.liver import endoscope_pose

position, look = endoscope_pose(LIVER_SCENE.shape, room)
shot = (
    tuple(float(value) for value in to_genesis(np.asarray(position, dtype=np.float32).reshape(1, 3))[0]),
    tuple(float(value) for value in to_genesis(np.asarray(look, dtype=np.float32).reshape(1, 3))[0]),
)
SCISSORS_SCENES["liver-room"] = replace(LIVER_SCENE, environment=room, shot=shot)
```

`endoscope_pose` 只适用于带 `z_max` 的腹膜腔。别的房间自己提供 `shot`。

完全不同的房间写一个自己的类。下面沿用现在的体腔和脂肪，只把灯留在镜头上：

```python
from endoscopic_tissue.environment import PERITONEAL_CAVITY

class LensOnlyCavity:
    fov = 56.0
    ambient = (0.05, 0.02, 0.02)
    lamp_scale = 2.0

    def overhead(self, tissue):
        return None

    def populate(self, view, tissue):
        PERITONEAL_CAVITY.populate(view, tissue)
```

```python
SCISSORS_SCENES["liver-lens"] = replace(LIVER_SCENE, environment=LensOnlyCavity())
```

```powershell
py -m endoscopic_tissue.run --scene liver-lens
```

肾脏场景的 `environment` 是空的，所以它继续用原来的肾周脂肪和逐顶点着色。给它装上环境后，画面改画这个环境，组织表面改用 `surface_kind` 对应的贴图。

## 固定方式接口

代码在 `endoscopic_tissue/fixation.py`。格子生成时已经带有场景原来的附着：方块的底面、肾脏埋在脂肪里的下部，或肝脏的两块韧带。演示可以在原生求解开始前换成一个点或一块区域。求解器只在初始化时复制这组钉住标记。

| 选择 | 行为 |
| --- | --- |
| `bed` | 默认。保留场景原来的附着 |
| `point` | 钉住离目标最近的一个表面结点，并松开原来的附着。目标离表面超过一个格子对角线时拒绝 |
| `region` | 钉住盒子里的全部表面结点，并松开原来的附着。盒子在每一轴上都要比 `1e-6` cm 厚，而且里面要有表面结点 |

命令行：

```powershell
py -m endoscopic_tissue.run --scene liver --fixation bed
py -m endoscopic_tissue.run --scene liver --fixation point --fixation-point -2 1 8
py -m endoscopic_tissue.run --scene liver --fixation region --fixation-low -4 4 6 --fixation-high -1 6 9
```

`bed` 不能再带坐标。`point` 只接受 `--fixation-point`。`region` 要同时给 `--fixation-low` 和 `--fixation-high`。坐标是厘米。贴在某个面上的一小片，写成大约一个格子厚、贴着那个面的盒子。

在脚本里直接替换格子上的钉住标记：

```python
from endoscopic_tissue.fixation import PointFixation, RegionFixation, apply_fixation

apply_fixation(grid, PointFixation((-2.0, 1.0, 8.0)))
apply_fixation(grid, RegionFixation(low=(-4.0, 4.0, 6.0), high=(-1.0, 6.0, 9.0)))
```

`apply_fixation` 返回钉住的结点个数。这一步要发生在原生会话创建之前。

## 力学信息接口

代码在 `endoscopic_tissue/forces.py`。求解器每步已经给出接触反力。`ForceMonitor` 把这些数交给画面和后续调用。没有装入拟合模型时，这些数是接触律自己的读数，窗口第一行写 `solver readout, not a measured force`。

窗口运行时，左上角就是当前这一步的 `estimate`。自己读取：

```python
from endoscopic_tissue.forces import ForceMonitor, MeasuredForce

monitor = ForceMonitor()
estimate = monitor.estimate(obs)
readout = monitor.readout(obs)
```

`estimate` 是应该使用的力。`readout` 是求解器原文，多保留了刀刃阶段和剪刀最近一次切断时的载荷。

`estimate` 的字段，单位牛顿：

| 字段 | 含义 |
| --- | --- |
| `jaw_upper_n`, `jaw_lower_n` | 两片夹子垫上的向内载荷 |
| `reaction_n` | 组织受到的接触力，三个分量 |
| `blade_force_n` | 切割器械的力，三个分量 |
| `blade_pressure_n` | 切割器械的压力 |
| `scissors_left_n`, `scissors_right_n` | 左右刃载荷。没有剪刀时是 `None` |
| `scissors_pressure_n` | 剪刀挤压力。没有剪刀时是 `None` |
| `is_calibrated` | 只有拟合模型给出的数才是真 |

把同一时刻的传感器样本和求解器读数配成一对，供以后拟合：

```python
pair = monitor.pair(obs, MeasuredForce(jaw_upper_n=0.4, jaw_lower_n=0.5))
```

传感器没有测到的通道留空。装上拟合模型后，`estimate` 改走模型，`is_calibrated` 成为真。模型没有修正的通道仍从求解器读数抄过来：

```python
from endoscopic_tissue.forces import ForceEstimate, ForceMonitor

class JawScale:
    def estimate(self, readout):
        return ForceEstimate(
            readout.jaw_upper_n * 0.5,
            readout.jaw_lower_n * 0.5,
            readout.reaction_n,
            readout.blade_force_n,
            readout.blade_pressure_n,
            readout.scissors_left_n,
            readout.scissors_right_n,
            readout.scissors_pressure_n,
            is_calibrated=True,
        )

monitor = ForceMonitor(JawScale())
# 或者之后再装：monitor.model = JawScale()
```

窗口会把第一行改成 `model estimate`。
