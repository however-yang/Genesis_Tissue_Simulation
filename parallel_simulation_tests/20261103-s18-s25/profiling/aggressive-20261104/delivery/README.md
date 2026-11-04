> GitHub 归档说明：本页结论保持原样。部分逐帧 JSON 链接已改为带原文件 SHA256 的指标摘要；原始 trace、编译库、环境和大型快照仅保留在服务器。详见 [归档范围](../../../GITHUB_ARCHIVE.md)。

# 最终最小候选

参见 ../REPORT.md 和 manifest.json。正式代码未替换，稳定20Hz状态按最终报告。

Python补丁的相对根是 profiling/candidate-v4-source/endoscopic_tissue，原生补丁的相对根是 native-baseline/source。只进行了 git apply --check，未实际应用。编译原生候选时使用已有CMake、nvcc 12.5和g++-12，Release/sm80，与本目录编译日志相同。

复现：用现有Python运行 ../tools/reproduce_final.py --output results/一个新目录。脚本检查源文件/库散列，仍受GPU资源守卫保护。
