# 隔离释放与切口重建实验

本包保存R21–R25及只读诊断源码。全部候选均未替换默认DLL，R25的局部几何修复通过本轮参考轨迹检查，但性能不退化及完整交互真实性仍未证明。详细结论见[验收报告](../../../endoscopic_tissue/docs/interaction-release-followup-2026-11-01.md)。

从固定Git提交 `38e6082edceef7898ab5ce8ca38a48ddcb175264` 提取基础源码，再逐文件校验并应用覆盖文件。DLL、构建缓存、原始数组、压缩回放和完整视频保留在本地，不加入本次Git选择。`evidence`保存构建输入/结果哈希及紧凑审计。CUDA 12.5、MSVC 2022、RTX 3090 / sm_86；构建器内记录本机工具路径。

在仓库根目录的cmd终端构建R25：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-11-01-release-next\package\build_experiment.py --variant release-r25-boundary-contact-identity
```

以下诊断保留原有材料参数、子步/迭代、接触及体积保护，不修改默认入口。`--allow-detached-flight`只允许记录被分类为游离体的出界运动，不能据此认定真实交互通过。输出目录必须是新目录，避免覆盖保留证据：

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-interaction-fixes\validate_strict.py --case kidney-reference --dll endoscopic_tissue_backups\2026-11-01-release-next\package\artifacts\release-r25-boundary-contact-identity\bin\LiverCudaSim.dll --output endoscopic_tissue\outputs\release-next-20261101\r25-kidney-reproduce --capture --archive-capture --allow-detached-flight
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-interaction-fixes\audit_release_screen.py endoscopic_tissue\outputs\release-next-20261101\r25-kidney-reproduce
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-11-01-release-next\audit_live_vertex_links.py endoscopic_tissue\outputs\release-next-20261101\r25-kidney-reproduce
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-grasp-cut-validation\render_review.py endoscopic_tissue\outputs\release-next-20261101\r25-kidney-reproduce
```

正式计时使用 `time_boundary.py`，`--boundary-graph 0/1`在初始化后设置；不启用诊断读回或视频。`run_r22_timing.py --revision r25 --include-r24-screen`记录本轮七次串行运行的进程、温度及时间戳，在输出目录已有结果时拒绝覆盖。一次完整同DLL对照包含关、开、开、关，前后各夹一个当前基线1215步；基线松夹失败，不伪造完整基线释放均值。应将新一轮输出另作隔离，再运行同一矩阵，不能与本轮结果混写。

录像30fps是仿真时间轴，不是实际显示FPS。`kidney-reference`沿用接触模式2，真正释放后锚点为0；它不替代纯摩擦模式1的深咬合、重抓或薄壁测试。R24存在接触身份混用，仅作诊断对照。R25保留真实切边作为缓存身份，表面分组只用于重建及插值；未增加切割范围、摩擦系数或辅助锚点。

`verify_experiment.py`核对源码/二进制哈希、完整录像、审计及受保护默认文件。这是实验出处和证据完整性检查，输出成功不代表候选通过手术交互验收。
