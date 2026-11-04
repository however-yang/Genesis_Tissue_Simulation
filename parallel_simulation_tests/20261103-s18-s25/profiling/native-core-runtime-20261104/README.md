# 本轮唯一候选：原生核心运行优化

**先看 [完整报告](REPORT.md)。普通完整 1280×960 RGB-D 为 24.84 Hz（V8 同条件 24.01 Hz），P95 46.24 ms。稳定 25 Hz、所有慢窗口 20 Hz 和全域交互无损均未获批。**

|用途|唯一入口|
|---|---|
|本轮候选 CUDA/C++ 源码|[native/source/](native/source/)|
|本轮冻结候选库|`native/binaries/runtime-core.so`|
|本轮检查/复测|[tools/reproduce.py](tools/reproduce.py)|
|参数、SHA、验收范围|[delivery/manifest.json](delivery/manifest.json)|
|真实原始数据与时间线|[results/](results/)|
|探索过程与淘汰原因|[EXPERIMENTS.md](EXPERIMENTS.md)|

本目录只保留一个原生候选源码树，Python 借用冻结 V8，不再复制 Genesis。`native/binaries/` 的其他库及 delivery 中标记 experimental 的补丁仅为历史证据；`native/build/` 是构建工作区。复测脚本依据清单加载上述唯一选定库，默认不启动 GPU。

```bash
P=/srv/disk2/hryang/genesis-exec
E="$P/parallel_simulation_tests/20261103-s18-s25/profiling/native-core-runtime-20261104"
PY="$P/envs/genesis-py311-cu125-v1/bin/python"
"$PY" "$E/tools/reproduce.py"
"$PY" "$E/tools/reproduce.py" --run benchmark --name new-full-frame-run
"$PY" "$E/tools/reproduce.py" --run validate --name new-numerical-proof
"$PY" "$E/tools/reproduce.py" --run rgbd-validated --name new-rgbd-check
"$PY" "$E/tools/reproduce.py" --run snapshot --name new-snapshot-check
```

每次名称必须未使用；仍有原 GPU 资源守卫，只使用 GPU1，顺序执行。core mode `43264`、64 query threads、1280×960、原 dt/子步/迭代/质量阈值由工具固定。初始化、计时方式及 dense S25/桌面显示等未覆盖范围见报告。

正式部署与项目根 `current/` 仍为 V8，未替换；本轮候选不是已经通过全部交互验收的新正式版。已测牵拉/实切、接触系数、RGB-D 和快照回归通过，136 项 CPU 回归通过；补充自由组织的严格跨轨迹 1 μm 复现未通过，参考版自身重跑也未通过，完整记录未删除。新几何、纯摩擦、wall 器械碰撞、真实撕裂分离等既有失败仍保留。
