> GitHub 归档说明：本页结论保持原样。部分逐帧 JSON 链接已改为带原文件 SHA256 的指标摘要；原始 trace、编译库、环境和大型快照仅保留在服务器。详见 [归档范围](../../GITHUB_ARCHIVE.md)。

# 实验索引与取舍

主数据与结论见 [REPORT.md](REPORT.md)。`quality_passed`只代表当前已启用局部指标；物理域与训练域保持未批准。

|实验|目录/结果|结论|
|---|---|---|
|新鲜完整RGB-D参考|results/baseline-rgbd-initial|21.38Hz、P95 56.61ms|
|剪刀合批初测|results/batch-rgbd-initial|22.16Hz、P95 53.95ms；未减少查询|
|剪刀完整GPU配对|results/batch-captured-validation|190350个原查询等价调用、约100万有效接触记录，差异0，覆盖图节点|
|活动工作量采样|results/workload-original-grid|只作诊断；约76050节点、205800四面体、15k表面三角形；接触采样远少于固定启动网格|
|网格扫描|results/grid-selection.json|全部结果保留；256/256第三刀失败；较快方案不能直接批准|
|查询线程数32/64/128|results/querythreads-selection.json|较小差异受状态/时段影响；最终回退64|
|最终激进组合|results/final-captured-validation|第三刀第1094帧触发位置非流形边；即使两轮RGB-D较快仍排除|
|只合批保守回退|results/conservative-suite-status.json|两轮完整GPU对照、两轮RGB-D、真实切割与快照通过|
|零贡献快捷返回|results/coarse-selection.json|5927万项无差异，但44.82→45.67ms；排除|
|已有宽相位组压紧|results/compact-32-rgbd|第二刀第793帧非流形边；失败，排除；局部对照成功不能覆盖完整失败|
|原包围盒预过滤|results/prefilter-selection.json|完整数值对照通过，44.98→45.74ms，无收益，排除|
|材料J/gJ复用|results/bound-cache-selection.json|两轮平均分别净省0.612/0.638ms；完整生产核对照约209亿项差异0|
|最终静默同库对照|results/delivery-suite-status.json|原核与剪刀合批+缓存完整对照；无并发编译；平均22.61Hz，P95 51.42ms，稳定20Hz未过|
|抓钳扩展V1|results/all-queries-full-validation|接触字节/度数/集合/材料无差异，但诊断错误使用6槽解释3槽；仍保留为验证失败|
|诊断槽步长GPU正反例|results/query-order-stride-probes|5例通过；正确组序及组内重排接受，跨组乱序拒绝；没有放宽组序要求|
|抓钳扩展V2|results/all-queries-v2-full-validation|完整GPU对照、六轮交错测量与RGB-D/快照通过；最终约23.03Hz、P95 50.42ms，稳定20Hz仍未通过|

## 诊断器修正的边界

V1的原分组检查器只面向剪刀，`slot % 6`用于提取器械kind。抓钳编码是`slot=3*triangle+kind`。例如合法的`[0,3,1,4,2,5]`按3解码是`[0,0,1,1,2,2]`，按6错误解码为`[0,3,1,4,2,5]`，产生假乱序。

V2保留原6槽检查器逐字不动，只为其他槽布局增加按实际stride及packed-kind次序的诊断。生产核未改；329个父版设备函数逐指令一致，唯一新增核是诊断。V1原库、源码、失败JSON和计数都保留；V2重新跑完整验证，不改写V1为通过。

## 原核与快照完整性

材料缓存初稿复用了同一共享归约模板，CUDA编译器因此将原CoarseAccumulate的共享内存地址重新排列。离线SASS审计发现后，将新核的共享工作区独立实例化，保留原归约运算、顺序和全部屏障；正式测试库恢复全部313个原设备函数指令一致。中间编译及差异保留在results/bound-cache-accumulate-sass-diff.json等文件，不作为测试成功版本。

所有新派生CUDA工作区使用`(::cudaMalloc)`/`(::cudaFree)`，不被旧快照分配宏拦截。没有排除或删减任何旧物理字段，没有修改state_archive或xpbd_state实现。真实快照回归检查完整原生字节恢复及恢复后下一步。

## 数据与资源口径

诊断、Nsight、短窗口、提前停止及失败运行不进入完整帧验收。完整RGB-D性能必须同时满足1410步完成和已启用局部质量检查通过。日志中的frame、GPU节点、CPU API包络与CPU线程时钟是不同层，不能加总父子范围或等待与计算。

所有GPU测试串行，仅用指定A100，启用外来进程/内存/温度守卫。探索期编译固定CPU30/31；最终交错验收在编译结束后进行。GPU0及其它任务未中断，不改系统、网络或依赖。
