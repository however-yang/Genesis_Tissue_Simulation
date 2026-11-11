# 接触工作量与模块优化研究诊断（2026-10-31）

对应 [分析报告](D:/Simulation/genesis-world/endoscopic_tissue/docs/module-optimization-analysis-2026-10-31.md)。

这两份补丁仅供独立诊断，均以 dev 的 fac3c7c 为基础；互为替代，不叠加。默认生产源已恢复。它们没有启用新的接触响应，插桩耗时不能用作性能比较。使用原有 kidney 开发入口记录既有拓扑问题并完成序列，最终质量失败仍保留。

## 文件

- count_contact_work.py：运行完整 1110 步并读取每步整数计数。
- contact-work-counters.patch：查询、分支、通用最近点、CCD 循环直方图。
- wrist-shadow-probe.patch：包含上述计数，另在腕部同输入计算解析最近点；原始 bary 继续驱动物理。
- counter-layout.json：96 个字段的含义；69 和 93–95 是最大值，93–95 使用 float32 位模式。
- analyze_research.py：重新生成摘要，核查计数守恒、步数、诊断 DLL 身份、生产保护和补丁适用性。
- contact-work-summary.json、wrist-shadow-summary.json：两次完整诊断的摘要。
- old-compaction-core-analysis.json：旧四次完整运行的核心/管线重算结果。

原始数据在 D:/Simulation/genesis-world/endoscopic_tissue/outputs/module-research-20261031/；论文检索在 sources/。大体积输出、DLL 和生产源备份不加入提交。

## 构建

从工作区根目录运行。先确认待修改 CUDA 源没有未保存改动。必须显式指定独立输出目录，不覆盖 native/bin 或 native/bin-scissors。

```bat
git apply endoscopic_tissue_backups/2026-10-31-module-research/contact-work-counters.patch
endoscopic_tissue\native\build_native.cmd D:\Simulation\genesis-world\endoscopic_tissue\native\bin-contact-work31
git apply -R endoscopic_tissue_backups/2026-10-31-module-research/contact-work-counters.patch
```

第二份改用 wrist-shadow-probe.patch 和 bin-wrist-probe31。使用反向补丁恢复，不使用覆盖用户改动的整文件 Git 回退。构建日志应包含 xpbd_solver.cu 的 CUDA 编译，不能只链接旧对象。

## 运行

选择新的输出目录，不覆盖已有结果。需要正常 CUDA 运行环境。

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-module-research\count_contact_work.py endoscopic_tissue/outputs/module-research-20261031/contact-work-new --tool scissors --dll endoscopic_tissue/native/bin-contact-work31/LiverCudaSim.dll
```

旁路比较使用 bin-wrist-probe31，输出另设 wrist-shadow-new。GPU 任务串行运行。两次诊断均以既有质量验收失败退出 1，已完成完整 1110 步；退出码不等于未完成记录，也不能将记录完成当成交互验收通过。

## 复核本轮证据

```bat
D:\develop\anaconda3\envs\genesis_world\python.exe -X utf8 endoscopic_tissue_backups\2026-10-31-module-research\analyze_research.py
```

脚本针对本轮 contact-work 与 wrist-shadow 原始目录，验证：

- 每次记录步骤 1..1110，报告完成，既有质量验收仍 false；
- 字段非负整数，查询/有效几何/粗筛/有效输出的包含关系；
- 四种窄阶段路径之和等于通过粗筛数；
- CCD 直方图之和等于 traceTriangleBox 调用数；
- 诊断 DLL 与对应报告身份一致；
- 生产 CUDA 源相对 fac3c7c 无差异，两份默认 DLL 哈希不变；
- 两个独立补丁均适用于保护源。

这些检查验证计数与保护完整性，不能证明达到 33 Hz 或手术真实性。正式优化候选须另行启用、正常 Graph 计时及完整视频/多场景对照。
