# Open3D 点云重建

在项目根目录，用 `.venv/Scripts/python.exe` 执行 `examples/open3d/pointcloud_to_mesh.py`。

## 框架结构

```text
examples/open3d/
  check_environment.py     环境与窗口检查
  generate_demo.py         生成带法线的合成球面点云
  pointcloud_to_mesh.py    读取、预处理、法线、重建、导出、预览
  data/input/             输入点云
  data/output/            导出网格
```

数据目录默认不加入 Git，代码与说明可以提交。合成球面的半径是 1 个坐标单位。

## 无真实数据时先跑通

VS Code 的“运行和调试”中，依次选择“点云框架：生成演示点云”和一种重建入口，按 F5。
生成后也可在项目根目录的 CMD 中执行：

```bat
.\.venv\Scripts\python.exe examples\open3d\generate_demo.py
.\.venv\Scripts\python.exe examples\open3d\pointcloud_to_mesh.py --input examples\open3d\data\input\demo_points.ply --output examples\open3d\data\output\mesh_poisson.ply --method poisson --view
```

老师提供点云后，把文件放进 `data/input/`，修改 `.vscode/launch.json` 中重建入口的 `--input` 路径。
确认坐标单位、法线和观测覆盖后，再选择方法、点间距与重建深度。

## 输入与重建

输入为表面点云，支持 `.ply`、`.pcd`、`.xyz`、`.xyzn`、`.xyzrgb`、`.pts`。
建议 PLY 保存 XYZ、可选 RGB 和法线。坐标单位由数据决定，整个流程保持原坐标和尺度。
XYZ 是空格分隔的纯数字文本，每行 `x y z`，无需表头。
XYZN 每行 `x y z nx ny nz`。CSV 需先按实际列名或列序转换为这些格式。

Poisson 适合较完整的曲面观测，可能封闭或外推缺失区域：

```bat
.\.venv\Scripts\python.exe examples\open3d\pointcloud_to_mesh.py --input data\points.ply --output results\mesh_poisson.ply --method poisson --depth 8 --view
```

Ball Pivoting 适合较密、较均匀的表面点，通常会留下未覆盖区域的孔洞：

```bat
.\.venv\Scripts\python.exe examples\open3d\pointcloud_to_mesh.py --input data\points.ply --output results\mesh_bpa.ply --method bpa --view
```

已有法线默认保留朝向。缺少法线时估计邻域法线并统一方向，仍需检查朝向和重建效果。
局部近似平面点云可用 `--normal-direction 0 0 1`，方向应按实际扫描表面选择。
`--voxel` 与 `--radii` 使用原始坐标单位。BPA 未指定半径时，采用点间距中位数的 1.5、3、6 倍。
`--denoise` 启用统计去噪，需检查薄结构和边界点是否被删除。
`--trim-quantile 0.01` 删除 Poisson 密度最低的 1% 顶点，可能造成孔洞，默认不裁剪。
提高 `--depth` 会增加时间和内存开销，输入缺失的几何细节不能靠提高深度恢复。

输出支持 PLY、OBJ、STL。PLY 可保存顶点颜色，STL 主要保存三角形几何。
现已支持 `.msh` 输出：用 `--msh-type volume`（默认）生成四面体体网格，用 `--msh-type surface` 只导出三角表面。需安装 `examples/open3d/requirements.txt` 中的依赖。已有 PLY/OBJ/STL mesh 会自动保留面连接关系；`--input-type pointcloud` 可强制从 PLY 顶点重新重建。具体素材路径与 MSH 运行示例见项目根目录 `pointcloud_to_mesh/README.md`。
点云的 PLY 只有点，mesh 的 PLY 还包含三角形面。
生成 mesh 后检查孔洞、孤立面、自交、尺寸和法线；用于仿真还需另外配置碰撞和物理网格。

P23D 的 `[batch,x,y,z]` 体素索引需要先选 batch，再转换为普通 XYZ。
对于 64 格、归一化区间 `[-0.5,0.5]` 的输出：`points = (indices + 0.5) / 64 - 0.5`。
这会得到量化后的体素中心，不能恢复原始点云的采样精度或真实物理尺度。
