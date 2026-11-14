# 点云转 Mesh 工具包

读取现有表面点云，用 Open3D 的 Poisson 或 Ball Pivoting（BPA）重建三角网格；也可直接转换已有三角 mesh。支持 Windows / Python 3.11，可在 CPU 上运行。

## 框架结构

```text
pointcloud_to_mesh/
  README.md                使用说明
  requirements.txt         Python 依赖
  check_environment.py     环境与窗口检查
  generate_demo.py         生成带法线的合成球面点云
  point_to_cloud_main.py   主程序：读取、预处理、法线、重建、导出、预览
  validate_input.py        输入几何验证，主程序自动调用
  format_config.toml      输入／输出路径与格式配置
  data/input/             输入点云
  data/output/            导出网格
```

数据目录默认不加入 Git，代码与说明可以提交。附带 `data/input/demo_points.ply` 和 `data/output/mesh_poisson.ply`、`mesh_bpa.ply`，均为合成球面演示数据，半径是 1 个坐标单位。

素材 PLY 已整理到 `data/input/` 顶层：26 个不同的素材文件和 1 个演示点云，共 27 个 PLY。已删除内容完全相同的重复副本、清单文件和空子目录；保留文件经 SHA256 核对，原素材目录不做修改。这批素材 PLY 已包含面连接关系，自动模式会直接使用已有表面网格。

## 配置环境

独立使用时，先安装 Python 3.11，再在本文件所在目录打开 PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_environment.py --no-window
```

工具包不包含 Python 或虚拟环境，首次安装依赖需要网络。

继续使用 `C:\Simulation_Study` 工程的虚拟环境时，从工程根目录运行，并在脚本路径前加 `pointcloud_to_mesh\`。例如：

```powershell
.\.venv\Scripts\python.exe pointcloud_to_mesh\check_environment.py --no-window
```

下文命令均在工具包目录内执行；无窗口运行时去掉 `--view`。

## 路径与格式配置

主程序为 `point_to_cloud_main.py`，启动时默认读取同目录的 `format_config.toml`：

```toml
[input]
path = "data/input"
format = "ply"

[output]
path = "data/output"
format = "msh"
```

`input.path` 可以是输入文件夹或单个文件；文件夹模式按文件名排序，处理顶层所有与 `input.format` 匹配的文件，扩展名大小写均可。`output.path` 默认是输出文件夹，每个结果保留输入文件名主体，例如 `points.ply` 输出为 `points.msh`。单文件输入也可配置完整输出文件路径，其扩展名须与 `output.format` 一致。

相对配置路径以配置文件所在目录为基准，与启动时的工作目录无关。Windows 绝对路径建议写成 `D:/scan/input`。输入格式支持 `ply`、`pcd`、`xyz`、`xyzn`、`xyzrgb`、`pts`、`obj`、`stl`，输出格式支持 `ply`、`obj`、`stl`、`msh`。

从工程根目录直接运行，即使用上述默认配置：

```powershell
.\.venv\Scripts\python.exe pointcloud_to_mesh\point_to_cloud_main.py
```

使用其他配置文件：

```powershell
.\.venv\Scripts\python.exe pointcloud_to_mesh\point_to_cloud_main.py --config D:\scan\format_config.toml
```

`--input`、`--output` 可分别覆盖配置路径；指定完整文件路径时，以该文件扩展名为准。其他重建和 MSH 参数仍通过命令行设置。默认 MSH 为体积网格，要求封闭表面；仅需表面网格时加 `--msh-type surface`。

## 验证与跳过

主程序自动调用 `validate_input.py`：已有网格检查有限坐标、三角形索引及退化面；体积 MSH 额外检查封闭性、边／顶点流形性和自交。纯点云先检查有效点数，重建后再检查表面。验证失败打印 `[SKIP] 文件名: 原因`，跳过该文件并继续后续文件；转换失败打印 `[FAILED]`，也继续处理。最后汇总成功、跳过和失败数量。已有文件的内容与几何保持原样。

可以独立运行验证程序，默认检查 `data/input` 中的 PLY，不生成输出：

```powershell
.\.venv\Scripts\python.exe pointcloud_to_mesh\validate_input.py
```

指定其他输入或仅验证表面导出要求：

```powershell
.\.venv\Scripts\python.exe pointcloud_to_mesh\validate_input.py --input D:\scan\input --format ply --msh-type surface
```

独立验证中的点云通过表示有效点数满足重建入口要求，体积网格适用性仍由主程序在重建后判断。验证结果不能保证 TetGen 一定成功。主程序有成功输出且无转换失败时返回退出码 0（允许跳过不合格文件）；全部跳过或存在转换失败时返回 1。配置、依赖和写入权限错误仍需修正后重试。

## 无真实数据时先跑通

生成演示点云，然后重建：

```bat
.\.venv\Scripts\python.exe generate_demo.py
.\.venv\Scripts\python.exe point_to_cloud_main.py --input data\input\demo_points.ply --output data\output\mesh_poisson.ply --method poisson --view
```

老师提供点云后，把文件放进 `data/input/`，替换运行命令的 `--input` 和 `--output` 路径即可。
确认坐标单位、法线和观测覆盖后，再选择方法、点间距与重建深度。

## 导出 Gmsh MSH

依赖清单已包含 `meshio` 和 `tetgen`。已有环境可补装：

```powershell
.\.venv\Scripts\python.exe -m pip install "meshio>=5.3" "tetgen>=0.8.2"
```

将点云重建为 `.msh`，把输出后缀设为 `.msh`：

```powershell
.\.venv\Scripts\python.exe point_to_cloud_main.py --input data\input\demo_points.ply --output data\output\mesh_poisson.msh --method poisson --msh-type volume
```

默认 `--input-type auto`：PLY 中已有三角面或输入为 OBJ/STL 时，直接保留坐标和面连接关系，进行格式导出；只有点的 PLY、PCD、XYZ 等才进行重建。已有 mesh 可显式使用 `--input-type mesh`。若希望忽略 PLY 里的面、仅从顶点重新重建，使用 `--input-type pointcloud`。

复制后的素材转体积 MSH 示例，在 `C:\Simulation_Study` 工程根目录运行：

```powershell
.\.venv\Scripts\python.exe pointcloud_to_mesh\point_to_cloud_main.py --input pointcloud_to_mesh\data\input\recorded_D_kidney_watertight_collision.ply --output pointcloud_to_mesh\data\output\recorded_D_kidney_volume.msh --input-type mesh --msh-type volume
```

MSH 使用 **Gmsh 2.2 ASCII**，默认 `--msh-type volume`：先用 TetGen 生成内部四节点线性四面体，再导出这些单元和完整的外边界三角形。物理组为 `surface`（编号 1、维度 2）与 `volume`（编号 2、维度 3）。体网格需要封闭、流形且无自交的表面；程序拒绝开放表面及退化四面体，并将四面体统一为正体积朝向。可用 `--tet-max-volume` 控制最大单元体积，单位是输入坐标单位的三次方。

仅需表面三角网格时，使用 `--msh-type surface`，不执行四面体化。MSH 保持输入坐标和单位，但只保存几何、单元连接和上述物理组，不导出顶点颜色、纹理或法线。四面体化可能增加节点和细分表面；材料、固定区域和载荷需要在 FEM／软体程序中配置。

## 输入与重建

输入为表面点云，支持 `.ply`、`.pcd`、`.xyz`、`.xyzn`、`.xyzrgb`、`.pts`。
建议 PLY 保存 XYZ、可选 RGB 和法线。坐标单位由数据决定，整个流程保持原坐标和尺度。
XYZ 是空格分隔的纯数字文本，每行 `x y z`，无需表头。
XYZN 每行 `x y z nx ny nz`。CSV 需先按实际列名或列序转换为这些格式。

Poisson 适合较完整的曲面观测，可能封闭或外推缺失区域：

```bat
.\.venv\Scripts\python.exe point_to_cloud_main.py --input data\input\points.ply --output data\output\mesh_poisson.ply --method poisson --depth 8 --view
```

Ball Pivoting 适合较密、较均匀的表面点，通常会留下未覆盖区域的孔洞：

```bat
.\.venv\Scripts\python.exe point_to_cloud_main.py --input data\input\points.ply --output data\output\mesh_bpa.ply --method bpa --view
```

已有法线默认保留朝向。缺少法线时估计邻域法线并统一方向，仍需检查朝向和重建效果。
局部近似平面点云可用 `--normal-direction 0 0 1`，方向应按实际扫描表面选择。
`--voxel` 与 `--radii` 使用原始坐标单位。BPA 未指定半径时，采用点间距中位数的 1.5、3、6 倍。
`--denoise` 启用统计去噪，需检查薄结构和边界点是否被删除。
`--trim-quantile 0.01` 删除 Poisson 密度最低的 1% 顶点，可能造成孔洞，默认不裁剪。
提高 `--depth` 会增加时间和内存开销，输入缺失的几何细节不能靠提高深度恢复。

输出支持 PLY、OBJ、STL、MSH。PLY 可保存顶点颜色，STL 主要保存三角形几何；MSH 可保存 Gmsh 表面三角网格或体积四面体网格。
点云的 PLY 只有点，mesh 的 PLY 还包含三角形面。
生成 mesh 后检查孔洞、孤立面、自交、尺寸和法线；用于仿真还需另外配置碰撞和物理网格。

P23D 的 `[batch,x,y,z]` 体素索引需要先选 batch，再转换为普通 XYZ。
对于 64 格、归一化区间 `[-0.5,0.5]` 的输出：`points = (indices + 0.5) / 64 - 0.5`。
这会得到量化后的体素中心，不能恢复原始点云的采样精度或真实物理尺度。

## 官方资源

- [Open3D 表面重建教程](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)
- [Open3D 安装说明](https://www.open3d.org/docs/release/getting_started.html)
