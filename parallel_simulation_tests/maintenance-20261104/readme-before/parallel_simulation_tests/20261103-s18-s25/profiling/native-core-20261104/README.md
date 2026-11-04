# 原生核心性能专项

已完成的主对照见 [REPORT.md](REPORT.md)。**稳定20Hz尚未通过，正式版本没有替换。** 最终推荐包含剪刀合批、抓钳合批和材料系数缓存；三配置交错测量与全部局部回归已完成。

本目录包含隔离原生源码、编译库、逐帧JSON、失败快照、Nsight节点时间线、GPU监控和复现工具；全部使用原项目环境，不安装依赖。

推荐配置及准确库散列以 [delivery/manifest.json](delivery/manifest.json) 为准。所有开关在原生初始化前设置，不能在episode中修改；物理快照的旧字段与实现保持原样，新增派生缓存不进入物理快照。

在本目录执行：

```bash
/srv/disk2/hryang/genesis-exec/envs/genesis-py311-cu125-v1/bin/python tools/reproduce.py --output results/recheck-rgbd-01
```

用 `--variant original` 复现同库关闭优化的原核；`--variant conservative` 只启用查询合批；`--physics` 测含全部已启用检查的无图帧；`--validate-native` 运行GPU逐项对照（诊断速度不用于FPS验收）。输出必须使用本目录results下尚不存在的新目录。

脚本核对源文件与库散列，并使用GPU资源守卫；GPU上已有其他计算进程或资源超限会停止，不会终止其他任务。默认仅用UUID `GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa` 的A100。

完整RGB-D口径为每步新1280×960图像、原始/对齐深度、有效性和独立CPU观测。显示、网络、录像、策略网络计算、训练及原基准未开启的密集S25特权标签不计入，报告没有宣称这些额外负载达到20Hz。

补丁相对上一轮native-gpu-diagnostics-v2/source，只执行 `git apply --check`，不会自动应用。正式集成时只启用报告已验证配置；实验库中其它启动调度选项不属于推荐配置。
