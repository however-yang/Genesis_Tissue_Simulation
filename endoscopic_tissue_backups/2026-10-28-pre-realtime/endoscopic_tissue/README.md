# 内窥镜软组织夹取与切割原型

这是独立放在 Genesis 仓库中的原型。**不需要启动 Unity**，也不会连接、重置或改写原 Unity 工程。组织与器械接触由复制并独立编译的 CUDA 核心计算，Genesis 提供场景和相机渲染。

当前场景：约 **60 × 40 × 14 mm** 的软组织块、底部附着，左侧 Cadiere 夹钳。默认演示使用 **dVRK 179 单极弯剪**：厚块顶面前缘立着一道 25 × 7 × 4 mm 的薄壁，物理网格 1 mm。剪刀从右前方斜着进入、躺在薄壁半高的水平面内连剪三刀：每刀两刃先压住组织、挤压变形达到起切条件后才剪开，断边跟随刀刃，切口就在刀口下。第一刀后夹钳夹住剪开的上片自由端提起 4 mm，露出切面；剪刀每次原地张开、沿壁滑 5 mm 再剪下一刀。三刀沿壁长剪出 11–15 mm 的伤口，切面为红色。`--scene flap` 运行原薄瓣三刀剪断流程（夹钳先夹住厚块提拉；6 × 4 × 20 mm 悬空薄瓣，从自由端剪到根部）。`--tool knife` 运行原单刀片流程（**16 mm** 有限切割刃）。剪刀设计、起切判定和验证见 [剪刀设计](docs/scissors-realism-design.md)。

历史单刀固定轨迹已在两次独立回归和一次 Genesis 录像中通过夹取、拉伸、单刀片切割和退刀验收。 默认网格为横向 2 mm、切入方向 0.5 mm。新增切面端点重建和退刀保护，详细数据、适用范围和录像见 [单刀片稳定性记录](docs/cut-stability.md)。旧失败记录保留在 [接触优化记录](docs/contact-optimization.md)。

此前器械版本的多次切割结果：0° 换位切割、45° 沿原切口重复进刀，各有一组通过并完成录像，使用不同的已记录构建；90° 重复切割仍有节点失稳。详见 [重复切割记录](docs/repeated-cutting.md)。加长演示保存在 `outputs/long-demo-20261027/`，来源与构建对应关系见 `provenance.json`。

## 达芬奇器械与脂肪外观

默认夹钳使用 [dVRK Cadiere 8 mm 模型](https://github.com/zixingjiang/dvrk_description)，包含镂空齿形钳口和分节腕部，保留原仓库 MIT 许可及来源记录。模型为研究用 Classic 器械网格，并非厂家认证的 Xi/X 数字孪生。默认钳口长 19.9 mm、宽 5.2 mm；轴杆与腕部采用约 8 mm 包络。根据较长钳口调整了接近位置，钳口接触代理由同一组顶点计算。默认刀向为45°，单次切入目标为原生y=+0.15 cm；原0°深切仍有接触与切面问题，失败记录保留。

单刀片保留原切割方式，采用公开腕部和参考 Snap-fit 外观制作的简化刀片装配件，**不是原厂 Snap-fit CAD**。镂空和细齿仍采用包围盒接触近似，腕部由保守轴杆包络覆盖。刀刃和刀背显示对应实际接触位置。

可操作组织块及切面改用淡黄色脂肪纹理；周边背景保留红色组织。贴图由内置 ImageGen 生成，原生 1254 × 1254，文件与完整提示词位于 `assets/textures/fat/`。纹理通过材料坐标绑定，拉伸时随组织运动。此处修改的是外观，材料参数仍为原有未标定软组织参数。

改动设计和结果见 [器械与脂肪外观记录](docs/davinci-fat-design.md)。原始预览录像与检查输出位于 `outputs/davinci-fat/`，不含侧边讲解。

## 双刃剪刀

剪刀使用 dVRK 179 单极弯剪网格，组织与两刃、刀背和腕部的接触、起切和切面重建由 `native/bin-scissors/` 的 DLL 计算。起切条件：剪刀正在闭合；左右刃各自沿闭合方向压在组织上的载荷（刃口与刀身合计）都不小于起切力的一半（默认每片 0.1 N）；并且两刃都开始受力后刀尖已再闭合 2 mm（`scissors_squeeze_cm`）。在此之前刀刃只压缩、推动组织，不改变拓扑。断边跟随刀刃：一条连接只有静止位置跨过本刀剪切面才可能断开，并且只在刃口实际扫过它、或到达它半个网格以内时断开，所以切口就在刀口下。验收另外检查起切时两刃载荷、两刃同时受力后的挤压行程、断边只在目标组织内、目标区域内跨剪切面的连接全部剪断、剪后连通性和每阶段的表面拓扑。

当前结果（DLL `29d9f329…`）：每刀都在两刃各自达到起切力一半、并挤压 2 mm 后才剪开，所有运行的切口都只在剪切面那一层，拓扑和连通性检查全部通过；伤口前表面 11–15 mm、后表面 11–13 mm。`wall-traction` 5 次运行 5 次通过全部验收。`wall` 抬高剪切面后 4 次中 3 次通过：后两刀远刃刀身推着剪切面下方剪不到的下半壁，刀背压入组织 0.13–0.24 mm，偶尔超过 0.2 mm 的非刃口重叠验收线，或后表面剪得偏短。薄瓣三刀流程和单刀片流程通过。其他限制：同一连通体内切口两侧切面之间没有接触；弯曲刃口按平直剪切面近似；组织参数未标定。详见 [剪刀设计与验证](docs/scissors-realism-design.md)。演示录像在 `outputs/scissors-follow/demo-8/endoscopic_demo_h264.mp4`；上一版（材料空间判定）在 `outputs/scissors-wall/demo-7/`，6 mm 薄瓣录像在 `outputs/scissors-demo2/run-1/`。

## 启动

在这台电脑上双击 `run_demo.cmd`。它使用现有 `genesis_world` Python 环境，打开内窥镜画面；空格暂停/继续，Q 或 Esc 停止。演示结束后保留最后一帧，按任意键关闭。

也可在 **D:\Simulation\genesis-world** 下运行：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.run
```

导出完整视频和截图，不打开窗口：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.run --headless
```

只看初始场景：

```bat
endoscopic_tissue\run_demo.cmd --steps 0
```

`--demo`（剪刀流程，需要画面）全程固定中景，剪刀最后退出后推近到伤口并定格 3 秒；物理流程与验收不变。`--scene wall|wall-traction|flap` 选择剪刀场景（默认 wall）；`wall-traction` 让夹钳在剪切前先夹住薄壁自由端顶部提起 1.2 mm，剪刀再进入，每剪一刀夹钳再提 0.5 mm。`--tool knife` 切换为单刀片流程。`--start-force` 改起切力（牛顿，剪刀每片刀刃需达到其一半），`--dll` 指定原生库。`--viewer` 额外打开 Genesis 自由观察窗口。`--native-only` 只运行物理流程。`--width 960 --height 720 --render-every 6` 可以降低显示负担。每次运行在 `outputs/时间戳/` 创建新目录；显式 `--output` 必须指定尚不存在的目录，避免覆盖实验。

已有依赖：Genesis 本地源码、NumPy、trimesh、Pillow、OpenCV。原生后端需要 Windows x64 和 NVIDIA CUDA GPU；随附 DLL 为 sm_86，在本机 RTX 3090 上验证。换 Python 路径可设置 `ENDO_PYTHON`。这不是 CPU 组织求解器，`gs.cpu` 仅用于轻量 Genesis 场景。

## 画面与材质

- 可操作组织和切面使用生成的脂肪颜色/法线贴图；背景保留原 Unity 肝组织材质。
- PBR 材质、湿润高光、相机附近光源与补光。
- 组织纹理绑定静止材料坐标，随组织运动；侧面独立投影，减少拉伸。
- 每个物理三角面单独细分为 16 个显示三角面，只插值法线和材质坐标，几何位置保持在物理表面上。
- dVRK Cadiere 钳口模型与接触代理共用铰点、开合轴和末端尺寸，跟随 CUDA 求解后的位姿和开度。
- 单刀片包含有限刃口、刀背和刀杆接触体；只有刃口能触发切割。
- 默认输出 1280 × 960；内部以两倍宽高渲染后抗锯齿缩小，附内窥镜暗角。

视觉细分只改善显示。它不提高物理网格分辨率，也不构成真实次表面散射或组织光学标定。周边组织是移出操作区的视觉背景。钳口使用模型末端包围盒作为接触代理，细小凹槽未建模。刀刃为有限胶囊、刀背与刀杆为有向盒，尚非完整手术刀接触网格。

## 验收与输出

`report.json` 明确区分 `passed`、`partial`、`failed`。`--steps` 提前结束、用户退出或只看初始帧都不算通过完整验收。验收要求：

1. 闭合后捕获实际材料节点。
2. 同一批节点的中位提拉高度至少 1 mm。
3. 进刀前无断边，切割阶段新增断边。
4. 重建出切口内壁三角面。
5. 退刀并保持后，原切割阶段的断边仍全部存在，夹持仍有效，且没有新增断边。
6. 每阶段检查没有退化面、开放边界、非流形边、重复面或绕序异常；整个流程刀背/刀杆重叠不超过 0.2 mm。
7. 材料节点相对初始位置的位移不超过 60 mm 操作范围；越界立即停止。

`cut_edges` 是**被切断的无向网格边数**，包含边界单元的空气区连接，不能直接当作真实组织切割数量、毫米数或伤口面积。验收同时检查真实内壁、断边保留和退刀不新增断边。独立评估程序还逐帧检查内壁退化。

输出包含 `trajectory.jsonl`、阶段 `.npz`、阶段 `.png`、`endoscopic_demo.mp4`、初始深度 `.npy`。状态数组后缀 `_cm` 为原生厘米坐标；深度 `_m` 为米。图像有暗角和文字，深度为未加视觉后处理的相机采样。后续学习图像策略时应关闭文字叠加，避免利用阶段标签。

## 当前物理边界

粗网格曾产生外层孤立节点和越界切面。当前使用真实刃口前缘扫掠，并将切入方向细化至 0.5 mm，已在当前组织块和固定单次切入轨迹中消除该现象。切面按各侧连通分量生成，剔除裂口前缘的内部重复面和零面积半面；没有冻结或隐藏异常节点来通过验收。

这不是任意方向切割的通用重网格方案。固定全部材料的刚性极限、任意刀向、反复切入、不同网格相位和更粗分辨率仍需单独验证。当前有限元会按断边停用体单元，未实现局部虚拟节点或体网格重新划分，不能据此声称网格收敛。

当前是**器械末端控制原型**。CUDA 同时计算组织形变、工具有限质量驱动、夹持与切割反力；Genesis 显示这些实际位姿。尚未接入 Genesis 机械臂关节动力学、RCM 约束或原生刚体双向耦合，也没有训练好的策略。夹持包含显式柔性锚点；组织参数未用实验标定；自碰撞和医学精度未验收。原生模块为单会话，不能在一个进程中建立多个独立环境。

## 代码入口

| 文件 | 作用 |
|---|---|
| `run.py` | 启动、阶段执行、结果记录、视频与窗口 |
| `scenario.py` | 控制路径及基于真实状态的验收 |
| `native_backend.py` | ctypes ABI、原生组织与工具接口、状态和切面读取 |
| `grid.py` | 程序化实体、材料邻接、Hermite 交点、4096 项连通表 |
| `rendering.py` | Genesis 相机、光照、PBR 组织和器械显示 |
| `visual_surface.py` | 不跨裂面连接的显示细分 |
| `instrument_geometry.py` | 显示与碰撞共用的钳口和刀片几何 |
| `assess_interactions.py` | 重复夹取/切割、松钳和不同网格配置的实际评估 |
| `probe_cut_topology.py` | 固定/运动组织对照，导出空气与材料断边诊断 |
| `probe_grasp.py` | 独立验证夹取、提拉、保持和释放 |
| `probe_knife_contact.py` | 刀背接触的固定组织正负对照 |
| `inspect_cut_surface.py` | 从保存状态生成同机位切前/切后近景与几何检查 |
| `native/vendor/` | 独立 CUDA/C++ 副本，已加入单刀片非刃口接触 |
| `native/bin/` | 本目录独立编译的 DLL，单刀片流程默认使用 |
| `native/bin-scissors/` | 剪刀 DLL：双刃起切判定（载荷与挤压行程）、刀背与腕部接触、跟随刀刃的剪切，剪刀流程默认使用 |
| `assets/` | 脂肪纹理、公开 dVRK 器械网格、原项目背景材质及来源记录 |

Genesis 的外部网格接口集中在 `rendering.py`，因为公共 debug mesh 默认不参与正常光照。这个适配层针对当前 Genesis 1.4.5；升级引擎后需复验，不需要修改 Genesis 源文件。

## 编译与检查

```bat
endoscopic_tissue\native\build_native.cmd D:\Simulation\genesis-world\endoscopic_tissue\native\bin-scissors
D:\develop\anaconda3\envs\genesis_world\python.exe -m unittest endoscopic_tissue.test_core -v
D:\develop\anaconda3\envs\genesis_world\python.exe -m endoscopic_tissue.smoke_native --steps 3
```

构建脚本使用本机 VS 2022 与 CUDA 12.5，输出只进入本目录。不传参数时输出到 `native/bin/`，会覆盖单刀片流程已验证的 DLL；当前源码含剪刀修改，按上面的命令输出到 `native/bin-scissors/`。换机器需调整脚本中的工具链路径和 CMake 的 GPU 架构。`native/source_manifest.json` 保存复制时的来源哈希，`native/build_info.json` 保存本次独立编译结果。

`NativeTissue.step(Command(...))` 是后续外部控制或训练环境的接入点。`Command` 接受夹钳目标位置、闭合量和刀刃目标位置，位置使用厘米；`Observation` 返回实际节点、拓扑、夹持状态、工具位姿与反力。先保证这个单环境物理基线，再增加机械臂和训练封装。
