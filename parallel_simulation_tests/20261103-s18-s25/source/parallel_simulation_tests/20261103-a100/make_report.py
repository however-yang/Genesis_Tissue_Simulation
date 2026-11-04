"""Compose the final report using measured artifacts; untested values stay unfilled."""
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    rows=json.loads((ROOT/'performance.json').read_text())
    full=[r for r in rows if r['complete'] and r['stop_after']=='hold' and not r['instrumented']]
    valid=[r for r in full if r['passed']==r['episodes']]
    best=max(valid,key=lambda r:r['aggregate_steps_s']) if valid else None
    lines=['# A100 软组织并行仿真实测报告','',
        '完整代码级分析、三方案比较、架构图、双 GPU/训练衔接与改造顺序见 [DESIGN.md](DESIGN.md)；可复现命令见 [README.md](README.md)。本报告只把已完成运行写为实测。','',
        '## 结论','',
        '当前可落地路线为 **C：每个独立进程持有一个原生求解器会话，优先使用本轮已验证的私有用户 MPS 配置**。A/B 需要原生状态/ABI 改造，Genesis 的 n_envs 不能直接复制本项目切割求解器。',
        f"本轮完成质量检查的最大并行数量为 **{max((r['workers'] for r in valid),default=0)}**；当前实测最高吞吐批次为 **{best['name']}，{best['aggregate_steps_s']:.2f} steps/s**。这是固定墙面场景的短批次验证下界，不是 A100 的硬件最大容量或小时级稳定性认证。" if best else '尚无完整通过批次。','',
        f"推荐先以 8 workers / 8 CPU 核部署采集试运行；需要更高吞吐且能保障 CPU 时，采用已测 {best['workers']} workers / {best['cpu_count']} CPU 核配置。若新网格、渲染或训练模型加入，重新测容量。" if best else '',
        '正式 release、CUDA 求解器、物理参数与验收阈值未改；依赖未安装/升级；网络、MIG 和 GPU compute mode 未改。始终只使用空闲 A100 UUID `GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa`，另一张繁忙 A100 未使用。','',
        '## 完整回合实测','',
        '固定 76,050 总节点 / 30,417 材料节点的 wall-traction；一夹钳一剪刀末端控制、辅助夹持、每回合 1,440 步，三次切割，dt=1/30、3 子步、完整质量检查、无渲染。没有测试完整关节机械臂动力学。','',
        '| 批次 | 环境数 | CPU 核数 | 完整通过 | 平均单环境 steps/s | 总 steps/s | 有效 episodes/h | 显存峰值 GiB | GPU 利用率均值 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(full,key=lambda r:(r['mps'],r['workers'],r['name'])):
        lines.append(f"| {r['name']} | {r['workers']} | {r['cpu_count']} | {r['passed']}/{r['episodes']} | {statistics.mean(r['per_env_steps_s']):.2f} | {r['aggregate_steps_s']:.2f} | {r['valid_episodes_hour']:.1f} | {r['peak_gpu_mib']/1024:.2f} | {r['mean_gpu_percent']:.1f}% |")
    lines += ['',
        '总吞吐使用整批墙钟时间，包含准备、原检查和输出，不把各 worker 的速度简单相加。episodes/h 是短批次折算，失败/部分回合不计为有效完整产出。显存峰值为约 2 秒间隔采样。原始记录全部保存在 runs/；完整指标含 P95/max，见各批次 summary.json。基准保存逐步质量元数据和最终网格，不逐帧落盘完整网格/图像；因此这是仿真验证吞吐，不是完整世界模型数据落盘或 RL 训练吞吐。','',
        '![实测性能](performance.png)','',
        '最初使用最多 8 个物理核对应逻辑 CPU，单线程 BLAS；每 worker 绑定其中一个逻辑 CPU。16-worker/8-CPU 实测用满 8 核后，在整机 CPU 约 21% 时另做 16-CPU 对照；上表明确列出实际核数，不能将增加 CPU 配额的收益归为 GPU 算力收益。其他用户的整机 CPU 负载变化明显。普通/MPS 对照不是严格 ABBA，不能将所有速度差都归因于 MPS。推荐配置需在实际训练负载下复测。','',
        '## 大规模准入与中断记录','']
    for count in (16,32,64):
        measured=[r for r in full if r['workers']==count]
        rejections=[]
        for f in (ROOT/'runs').glob('*/status.json'):
            d=json.loads(f.read_text())
            if d.get('phase')=='rejected_preflight' and d.get('workers')==count:rejections.append((f,d))
        if measured:
            lines.append(f"- {count} 环境：有完整回合实测，见上表。")
        elif rejections:
            d=rejections[-1][1]
            lines.append(f"- {count} 环境：启动前拒绝；每 worker 预留 {d['per_worker_reserve_mib']:.0f} MiB，合计约 {d['predicted_reserve_mib']/1024:.1f} GiB，超过本轮 {d['gpu_budget_mib']/1024:.1f} GiB 预算。没有启动 worker，不能称为运行失败或硬件容量实测。")
        else:lines.append(f'- {count} 环境：没有完整已完成结果，未验证。')
    lines += ['',
        '32 环境的裸显存线性外推可能放得下，但加上逐环境余量和本轮预算后不放行；64 环境的裸显存外推已超过单卡 80GB。外推仅针对同一无渲染墙面库，不能代替真实分配或运行。双 A100 未实测，因为另一张卡持续有其他任务。','',
        '`plain-4-r0` 在检测到外来 CUDA 进程后仅停止了自己的四个 worker；保留所有日志，不纳入性能结论。启动前拒绝和监测中断均不伪装成通过。','',
        '## 正确性与接口验证','',
        '串行重复与并行分布的数值范围见 [correctness-summary.md](correctness-summary.md)。完整逐步检查通过与严格物理等价是不同层次，本轮只宣称前者及已执行的状态隔离测试。','',
        '- 完整通过 63 回合（串行 3、并行 60）。串行两两材料节点 RMS 差 0.0229–0.0303 mm，并行对首个串行 0.0148–0.0653 mm；最大局部差分别为 2.61 mm 和 4.46 mm。并行范围更宽，且样本量不同，尚未完成统计等价检验。',
        '- 最大局部差所在回合中，30,417 材料节点有 4 个差异超过 1 mm；仍不能用较小全局 RMS 排除局部问题。节点定位见 largest-local-difference.json，后续应检查切口局部力与形变轨迹。',
        '- 短测两个进程各重置两次，共四个 reset 均通过；最终邻接一致，最大节点差约 5.06e-6 mm。见 reset-comparisons.json。',
        '- 所有完整通过批次沿用原逐步质量、工作区、表面与材料切割检查，没有降低子步/迭代或放宽阈值。通过只覆盖当前辅助夹持墙面动作域。',
        '- 完整切割后的节点/断边并非逐位一致。state-comparisons.json 和 cut-repeatability.json 记录 RMS/最大差、起切步和材料断边数；串行重复与并行分布须共同查看。不能将单一最终差值直接解释为污染，也不能仅凭同样通过就宣称统计等价。',
        '- 两材料的接口测试只覆盖短动作及独立重置；不同材料/几何同时完整切割、不同抓取点和大幅牵拉方向组合尚未验收，不能由相同墙面脚本外推。',
        '- 原生会话使用独立进程地址空间和 CUDA 客户端；未复用/覆盖对方的原生全局变量。CPU-only 统计回归检查确认部分回合不计有效 episode，吞吐分母使用整批时间。','']
    for f in sorted((ROOT/'runs').glob('*/integration-result.json')):
        d=json.loads(f.read_text())
        lines.append(f"- `{f.parent.name}`：passed={d['passed']}，{d['recorded_valid_windows']} 条有效数据窗口通过读取及哈希验证；独立材料、不同动作和单环境重置测试（16-worker 批次另显式验证第二个在途请求被拒绝）。详情见对应 integration-result.json。")
    lines += ['',
        '## 耗时与瓶颈','',
        '接口短测：16 workers / 8 CPU 核、开启完整 EpisodeWriter 后，128 个采样动作响应平均约 715 ms，其中 worker 步进及记录约 650 ms；响应约 3.49 KB。差值约 65 ms 含收集端调度、监控和 IPC。该短测没有剪切，不能与完整剪切基准直接相减来估算写盘成本。',
        '紧凑观测回传、worker 内记录与端到端响应的短测统计见 integration-summary.json。往返减去 worker 时长仍包含接收端调度与资源监控，不把它误称为纯 IPC 延迟。完整帧数据传输及实际训练是否成为瓶颈，尚需带模型和渲染负载测试。','',
        '| 批次 | 原生步进及读回 ms/步 | 质量检查+表面导出 ms/步 | 完整环境 step ms/步 |',
        '|---|---:|---:|---:|']
    for r in sorted(full,key=lambda r:(r['mps'],r['workers'],r['name'])):
        lines.append(f"| {r['name']} | {r['native_step_ms']:.2f} | {r['quality_and_export_ms']:.2f} | {r['total_step_ms']:.2f} |")
    lines += ['',
        'quality_and_export 包含 surface_export，不能重复相加。原生步进是主机观察到的延迟，含上下文调度与读回，不是纯 kernel 执行时间。普通进程数增加后该延迟明显增长；MPS 可改善跨进程调度，但 CPU 几何检查仍占重要部分。NVML memory utilization 不是 GB/s，本轮未测 Nsight SM occupancy 或有效 HBM 带宽。','',
        '独立 section profiling 见 [profile-breakdown.md](profile-breakdown.md) 和原始 profile-breakdown.json。材料与接触采用耦合迭代，现有计时点把 cut tick 与子步间拓扑重建合并；无法诚实拆成完全独立的四个纯 kernel 时间。开启 section profiling 会绕过部分 Graph 并插入事件，故这些值仅用于定位，不能混入常规吞吐。','',
        '## 推荐实施顺序','',
        '1. 在本隔离目录使用已通过批次的 worker 数进行采集试运行；配置固定库 SHA、GPU UUID、质量规则与唯一输出目录。并行池 `reset/step`、有界 IPC 和每 worker EpisodeWriter 已实现。',
        '2. 在空闲资源保障下重复普通/MPS 配对及小时级压力回归，补充纯摩擦重抓、真正松夹、肾脏大网格、不同几何和 RGB-D；这些不能由墙面三剪代替。',
        '3. 接入实际策略推理，记录策略版本、episode/step/request ID、实际执行动作、失败原因和 bootstrap_allowed；按 reset lineage/几何/材料划分数据，避免分支泄漏。当前未执行 RL 优化器或训练模型，推理/训练吞吐没有实测数字。',
        '4. 两卡空闲时可按 episode 分片各跑一个采样池，或一张采样一张训练；当前 PIX 拓扑没有显示 NVLink，不依赖跨卡共享原生状态，也不承诺共享 CPU 下 2 倍吞吐。',
        '5. 原生改造先做同一步导出/几何计算复用的逐项等价验证，再考虑显式 context/stream 所有权，最后才做 env-major CUDA batching。现在已有进程/MPS 路线的实测收益，不建议先大规模重写跨多个 CUDA 模块的静态状态与动态切割缓存。批量内核的额外收益尚未验证。','',
        '内核优化的收益判断：当前已用不改物理核心的进程/MPS 方案取得实测吞吐提升；在 CPU 表面检查仍占较大比例时，重写全部 CUDA 状态的收益上界与成本均未验证。优先做可逐项比较的重复导出/CPU 计算复用，保留验收，再决定 context 化或 batching。',
        '所有本次工具、日志、结果与图都在本目录。MPS 生命周期和退出记录在 mps/*/lifecycle.json；最终清理和源码状态证据见 final-verification.json。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT.md generated from',len(full),'completed full batches')

if __name__=='__main__':main()
