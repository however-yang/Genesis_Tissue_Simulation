# 腕部解析搜索实验（2026-10-31）

起点为 dev 39e5ebc655fde6f77259783fecaa43bda78de2d0。默认剪刀、单刀 DLL 均保留。两个候选分别基于同一起点，不能叠加补丁；均不作为提速优化启用。

## 候选与结果

- initial-wrist.patch：仅在剪刀非切割腕部 kind=5 的新接触查询使用现有有限线段/三角形解析搜索，退化、近乎平行、轴上法向不唯一、非有限结果回退原算法。
- dedicated-wrist.patch：同一几何替换独立编译为专用腕部核函数。调用方只在 bladeShapeKind 返回5时分派；不传刀刃 phase，不启用预载/切割逻辑。
- 原材料、摩擦、接触响应、网格、dt、substeps、迭代数、拓扑身份和阈值不变。

|正常 CUDA graphs，1110步/次|基线核心 ms|候选核心 ms|基线流水线 ms|候选流水线 ms|
|---|---:|---:|---:|---:|
|初版 ABBA 均值|43.5661|43.8216|52.4471|51.5992|
|专用版 ABBA 均值|42.9367|42.8442|52.3675|56.3511|

专用版核心仅下降0.215%，小于基线两轮波动，流水线上升7.607%；没有可确认的整体收益。不能把这项实验算作已实现加速。

编译资源：原 FindSurface 为135寄存器/1872字节栈；初版181/2104；专用版普通路径135/1872、腕部135/864。寄存器增多是初版无收益的一种解释，不视为单独证实的因果关系。

## 验证

check_wrist_geometry.py 对1698组几何运行真实CUDA helper，与独立双精度约束最小二乘参考比较。改变了重心坐标的866组案例中，最大距离误差3.50e-7 cm（3.50e-6 mm）；包含42个指定必须回退的案例，均逐位保留基线重心坐标。总计866组坐标改变、832组与基线相同，后者不等同于实际回退次数。验证容差和判据见脚本；不能把几何测试通过当作真实交互效果验证。

专用版对非腕部接触运行74次冻结输入检查：1036个shape查询、36,814,876个三角面查询，接触记录、degree、活动集合均0差异。检查刻意排除改变算法的腕部，不声称腕部与原查询逐位相同。

三个实际交互记录分别为 baseline-capture、initial-capture、dedicated-capture；记录完整求解状态，不用插值伪造高帧率。review_motion.py 比较全1111帧的组织区域运动、夹爪开度、剪刀开度及双侧载荷。视频固定30 fps仅表示模拟时间播放，不表示执行速度。最终视觉观察与失败项见主实验报告。

## 复现

所有输出使用新目录，GPU任务串行，计时期间不编译/渲染。正常计时与诊断版本分开构建，始终为 build_native.cmd 指定独立 bin 路径。

- timing-jobs.json / dedicated-timing-jobs.json：正常计时清单；用上一轮的 run_batch.py 运行。
- wrist_geometry_probe.cuh：只在几何测试副本的 xpbd_solver.cu 末尾 include，导出 LCS_ProbeWristGeometry；运行 check_wrist_geometry.py DLL 输出JSON。
- contact_equivalence.cuh：只在专用版诊断副本 include；原检查器参考来自 reference_find_surface.cuh。
- capture_motion.py：完整录像/区域运动记录；--motion-only 可省略大体积回放文件，不用于性能计时。
- run_quality_review.py：串行非腕部检查、专用版录像和三个完整视频/两组对照。

默认剪刀 SHA256：0d09f56a1f4acb813424cd1dee3233d89ce7ef5c0d45e64f079c7d7387912225。
原始单刀 SHA256：2632a926b385d9c9a81d5f7d7ebb75fe0501917a9cf83d0c19d3a29c62af961b。

最终报告：[本轮实现与交互对照](D:/Simulation/genesis-world/endoscopic_tissue/docs/contact-optimization-trials-2026-10-31.md)。
