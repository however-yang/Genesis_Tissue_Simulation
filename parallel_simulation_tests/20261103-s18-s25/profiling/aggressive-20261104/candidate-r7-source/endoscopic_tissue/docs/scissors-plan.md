# 双刃剪刀实施计划

> 按 [剪刀设计](scissors-realism-design.md) 执行。本会话内联执行，每个任务完成后先跑对应检查再进入下一项。
>
> 实施中的偏离以设计文档为准：薄瓣伸出改为 8 mm；剪切面改为平面；剪刀切割改为按接触前材料位置判定（CUDA 第 4 项修改）；流程改为先闭合退出再张开。

**目标**：夹钳夹住厚块提拉，dVRK 179 弯剪先压缩薄瓣、两刃各自达到起切力一半后剪开一道切口，再张开退出。

**结构**：CUDA 副本负责起切判定与载荷导出；`grid.py` 生成带薄瓣的组织；`instrument_geometry.py` 从网格求刃口与刀背；`scenario.py` 定义剪刀流程与验收；`run.py` 记录并录像。

**环境**：`D:\develop\anaconda3\envs\genesis_world\python.exe`，在 `D:\Simulation\genesis-world` 下运行；VS 2022 + CUDA 12.5，sm_86。

## 全局约束

- 原 Unity 工程只读；`native/bin/LiverCudaSim.dll` 不覆盖，剪刀 DLL 输出到 `native/bin-scissors/`。
- 默认单刀片流程的网格（不加薄瓣）逐值不变。
- 验收门槛只按设计设定，失败结果保留并标记 failed，不放宽。
- 输出目录必须尚不存在。

---

### 任务 1：原生起切判定与载荷导出

**文件**：`native/vendor/cuda/cutting/pressure_blade.cuh`、`native/vendor/cuda/xpbd/xpbd_solver.cu`、`native/vendor/cuda/xpbd/xpbd_solver.h`、`native/vendor/src/plugin_api.cpp`

**产出**：`LCS_GetScissorsLoads(float* values6)`，依次为当前左刃载荷、当前右刃载荷、最近一次起切时左刃载荷、右刃载荷、总压力、起切次数；单位与 `startForce` 相同（SI 力除以每单位米数）。

1. 新增持久缓冲 `d_scissorsLoads`（6 个 float），在 `allocateBlade` 中分配并清零，随刀具释放；`xpbd_set_scissors` 首次启用或停用时清零。
2. `k_BladeReaction` 末尾增加参数 `float* loads=nullptr`；接受子步的调用传入 `d_scissorsLoads`。剪刀分支进入时写当前左右载荷。
3. 起切条件改为：

```cpp
float required=.5f*.999f*startForce;
if(!history->fractured && closing && leftLoad>=required && rightLoad>=required){
    *phase=BladeCutting;history->fractured=1;
    if(loads){loads[2]=leftLoad;loads[3]=rightLoad;loads[4]=pressure;loads[5]+=1.f;}
}
```

4. 重新接触时把切割结束也复位为挤压：

```cpp
if(*phase==BladeClearStroke || *phase==BladeWithdrawing || *phase==BladeCutExit)*phase=BladePressing;
```

5. `build_native.cmd endoscopic_tissue\native\bin-scissors` 编译；`smoke_native` 指向新 DLL 跑 3 步。

### 任务 2：带薄瓣的组织网格

**文件**：`grid.py`、`test_core.py`

**产出**：`build_grid(spacing, size, density, vertical_spacing, flap=None)`；`flap` 为 `Flap(length, thickness, width)`（cm），从厚块 +x 侧面伸出、顶面齐平、z 居中。内部用盒子列表求并集。

1. 网格范围取全部盒子的包围盒，`origin = 包围盒中心 - dims * cell / 2`，不加薄瓣时与原值相同。
2. 薄瓣盒子向厚块内延伸一个横向格，避免两盒交界面上的节点被判为外部。
3. 交点：沿边轴，从内部端点出发取包含该端点截面的盒子的最远出口，再沿首尾相接的盒子延伸；法线为该轴方向、指向外部端点。
4. 固定节点仍为厚块底面。
5. 检查：不加薄瓣的网格与修改前逐值相同（开发期一次性比对）；新增测试覆盖交点位于盒面、法线朝外、固定节点只在厚块底面、薄瓣节点为活动且非固定、邻接互反。

### 任务 3：剪刀几何

**文件**：`instrument_geometry.py`、`test_core.py`

**产出**：`load_scissors(max_gap=1.6)` 返回 `ScissorsGeometry(jaws, edges, backs, max_gap)`；局部坐标为（尖端、张开、宽度），第一片 `tip_179_2` 向 +张开方向转动。

1. 刃口：每片在尖端坐标 4–11 mm 范围内取朝向另一片的最外侧顶点，最小二乘拟合直线，得到胶囊端点。
2. 刀背：每片在同一范围内钝侧顶点的有向包围盒，与刃口保持不小于 0.3 mm 间隙。
3. 测试：两条刃口在闭合时位于两片交界面附近（宽度方向差 < 0.5 mm）；刃口到网格最近距离 < 0.2 mm；刀背盒不包含刃口胶囊；按原生规则 `maxGap <= 1.7321 * length`。

### 任务 4：剪刀流程与验收

**文件**：`native_backend.py`、`scenario.py`、`run.py`

**产出**：`Observation.scissors_loads_n`（6 个值，牛顿，次数除外）；`SCISSORS_STAGES`；`Metrics` 新增 `fracture_left_n`、`fracture_right_n`、`precut_compression_m`、`slit_length_m`、`cut_outside_flap`、`material_components`。

1. 剪刀模式加载 `native/bin-scissors/LiverCudaSim.dll`，绑定 `LCS_GetScissorsLoads`。
2. 流程：settle → approach → close → lift → scissors_approach（张开到最大，上下刃跨住瓣自由端）→ snip（开启切割，闭合）→ release（张开，关闭切割）→ withdraw → hold。
3. 验收按设计逐项实现；起切前断边、起切载荷、压缩量从逐步观测计算。

### 任务 5：位姿调试

1. `--native-only` 测 settle 后瓣尖下垂，按设计规则处理。
2. 调夹钳位置直至捕获材料、提拉不小于 1 mm。
3. 调剪刀接近位姿：张开时上下刃与瓣上下表面均有间隙，闭合时两刃都接触瓣。

### 任务 6：验证与记录

1. `unittest endoscopic_tissue.test_core`。
2. 两次独立 `--native-only` 运行、一次 `--headless` 录像，输出到 `outputs/scissors-*`。
3. 新 DLL 跑默认单刀片流程回归。
4. 更新 `README.md` 与设计文档的结果部分。
