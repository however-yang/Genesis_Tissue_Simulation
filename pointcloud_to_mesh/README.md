# 点云转 Mesh 工具包

读取现有表面点云，用 Open3D 的 Poisson 或 Ball Pivoting（BPA）重建三角网格。支持 Windows / Python 3.11，可在 CPU 上运行。

## 框架结构

```text
pointcloud_to_mesh/
  README.md                使用说明
  requirements.txt         Python 依赖
  check_environment.py     环境与窗口检查
  generate_demo.py         生成带法线的合成球面点云
  pointcloud_to_mesh.py    读取、预处理、法线、重建、导出、预览
  data/input/             输入点云
  data/output/            导出网格
```

数据目录默认不加入 Git，代码与说明可以提交。附带 `data/input/demo_points.ply` 和 `data/output/mesh_poisson.ply`、`mesh_bpa.ply`，均为合成球面演示数据，半径是 1 个坐标单位。

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

## 无真实数据时先跑通

生成演示点云，然后重建：

```bat
.\.venv\Scripts\python.exe generate_demo.py
.\.venv\Scripts\python.exe pointcloud_to_mesh.py --input data\input\demo_points.ply --output data\output\mesh_poisson.ply --method poisson --view
```

老师提供点云后，把文件放进 `data/input/`，替换运行命令的 `--input` 和 `--output` 路径即可。
确认坐标单位、法线和观测覆盖后，再选择方法、点间距与重建深度。

## 输入与重建

输入为表面点云，支持 `.ply`、`.pcd`、`.xyz`、`.xyzn`、`.xyzrgb`、`.pts`。
建议 PLY 保存 XYZ、可选 RGB 和法线。坐标单位由数据决定，整个流程保持原坐标和尺度。
XYZ 是空格分隔的纯数字文本，每行 `x y z`，无需表头。
XYZN 每行 `x y z nx ny nz`。CSV 需先按实际列名或列序转换为这些格式。

Poisson 适合较完整的曲面观测，可能封闭或外推缺失区域：

```bat
.\.venv\Scripts\python.exe pointcloud_to_mesh.py --input data\input\points.ply --output data\output\mesh_poisson.ply --method poisson --depth 8 --view
```

Ball Pivoting 适合较密、较均匀的表面点，通常会留下未覆盖区域的孔洞：

```bat
.\.venv\Scripts\python.exe pointcloud_to_mesh.py --input data\input\points.ply --output data\output\mesh_bpa.ply --method bpa --view
```

已有法线默认保留朝向。缺少法线时估计邻域法线并统一方向，仍需检查朝向和重建效果。
局部近似平面点云可用 `--normal-direction 0 0 1`，方向应按实际扫描表面选择。
`--voxel` 与 `--radii` 使用原始坐标单位。BPA 未指定半径时，采用点间距中位数的 1.5、3、6 倍。
`--denoise` 启用统计去噪，需检查薄结构和边界点是否被删除。
`--trim-quantile 0.01` 删除 Poisson 密度最低的 1% 顶点，可能造成孔洞，默认不裁剪。
提高 `--depth` 会增加时间和内存开销，输入缺失的几何细节不能靠提高深度恢复。

输出支持 PLY、OBJ、STL。PLY 可保存顶点颜色，STL 主要保存三角形几何。
点云的 PLY 只有点，mesh 的 PLY 还包含三角形面。
生成 mesh 后检查孔洞、孤立面、自交、尺寸和法线；用于仿真还需另外配置碰撞和物理网格。

P23D 的 `[batch,x,y,z]` 体素索引需要先选 batch，再转换为普通 XYZ。
对于 64 格、归一化区间 `[-0.5,0.5]` 的输出：`points = (indices + 0.5) / 64 - 0.5`。
这会得到量化后的体素中心，不能恢复原始点云的采样精度或真实物理尺度。

## 官方资源

- [Open3D 表面重建教程](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)
- [Open3D 安装说明](https://www.open3d.org/docs/release/getting_started.html)
