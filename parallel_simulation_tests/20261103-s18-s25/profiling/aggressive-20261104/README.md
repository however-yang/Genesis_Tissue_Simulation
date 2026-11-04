> GitHub 归档说明：本页结论保持原样。部分逐帧 JSON 链接已改为带原文件 SHA256 的指标摘要；原始 trace、编译库、环境和大型快照仅保留在服务器。详见 [归档范围](../../GITHUB_ARCHIVE.md)。

# 实时性专项独立实验

已完成结果见 [REPORT.md](REPORT.md)，实现与失败取舍见 [EXPERIMENTS.md](EXPERIMENTS.md)。

最终最小源码位于 final-source，补丁与散列位于 delivery；原正式代码未替换。复现入口 tools/reproduce_final.py，必须选择本目录中的新输出目录。仅使用原Python环境，不安装依赖。

历史候选candidate-r1到r8、原生特化/压紧/预计算及GPU只读诊断全部独立归档。失败实验不表示已采用；结果中profiled、topology_validation、native_validation及不完整场景不能计作完整性能验收。全部命令见各轮commands.jsonl，原始逐帧结果见results，日志见logs。

不要在等待/执行中的队列外另起GPU基准。资源守卫会在外来任务出现时停止本轮，绝不终止外来作业。物理域原有失败和密集S25标签问题仍未解除，正式训练域未批准。
