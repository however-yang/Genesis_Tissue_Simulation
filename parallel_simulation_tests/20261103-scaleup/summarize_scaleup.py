"""Keep attempted capacity, accepted physics and raw throughput separate."""
import collections
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent

def main():
    d=json.loads((ROOT/'runs/m32a/summary.json').read_text());duration=d['duration_s_including_reset_checks_and_output']
    failed=[]
    for ep in (ROOT/'runs/m32a').glob('worker-*/episode-*/result.json'):
        r=json.loads(ep.read_text())
        if not r['passed']:
            rows=json.loads((ep.parent/'steps.json').read_text())
            bad=[{'tick':s['tick'],'stage':s['stage'],'quality':s['metadata']['quality']} for s in rows if not s['metadata']['quality']['checked_metrics_passed']]
            failed.append({'path':str(ep.relative_to(ROOT)),'failure_steps':bad})
    s=json.loads((ROOT/'runs/m64a/summary.json').read_text());status=json.loads((ROOT/'runs/m64a/status.json').read_text())
    errors=collections.defaultdict(list)
    for f in sorted((ROOT/'runs/m64a').glob('worker-*/stdout.log')):
        for line in f.read_text(errors='replace').splitlines():
            if line.startswith('RuntimeError: LCS_'):errors[line].append(f.parent.name)
    result={'32':{'attempted':32,'completed_full_episodes':32,'passed_full_episodes':d['passed_full_episodes'],
        'aggregate_all_steps_s':d['aggregate_steps_s'],'aggregate_steps_from_accepted_episodes_s':d['passed_full_episodes']*1440/duration,
        'effective_episodes_hour':d['valid_episodes_per_hour'],'mean_per_env_steps_s':statistics.mean(r['steps_s'] for r in d['worker_rates']),
        'peak_gpu_mib':d['gpu_memory_mib']['max'],'mean_gpu_percent':d['gpu_utilization_percent']['mean'],'duration_s':duration,
        'failed_episodes':failed,'long_duration_stability_certified':False},
        '64':{'attempted':64,'imports_ready':len(list((ROOT/'runs/m64a').glob('worker-*/ready.json'))),
              'full_episodes':0,'complete':False,'peak_gpu_mib':s['gpu_memory_mib']['max'],
              'worker_error_groups':dict(errors),'observed_mps_client_limit':48,
              'mps_limit_evidence':'mps/m64a/l/server.log: Maximum number of clients (48) reached.',
              'cuda_status_interpretation':{'-2':'negative cudaErrorMemoryAllocation; allocation failure','-46':'negative cudaErrorDevicesUnavailable; MPS server rejected extra clients'},
              'status':status['error'],'result_files':s['recorded_episodes'],'completed_short_settles':sum(r['passed'] and not r['full_script'] for r in s['results']),
              'not_a_64_environment_throughput_measurement':True},
        'recovery':json.loads((ROOT/'runs/post64-reset/summary.json').read_text()),
        'scope':'Same pinned wall/three-snips physics and acceptance as prior study; CPU/shared-server load not controlled; 32 workers had one rejected episode.'}
    (ROOT/'scaleup-summary.json').write_text(json.dumps(result,indent=2))
    r=result['32'];q=result['64']
    text=f'''# 32 / 64 环境实际扩容结果\n\n用户在上一轮后授权越过保守预算，实际启动 32 与 64 个进程；本轮没有停在显存预测阶段。正式求解器和验收规则未修改。\n\n| 配置 | 完整通过 | 总 steps/s（含失败回合成本） | 合格回合 steps/s | 合格 episodes/h | 显存峰值 GiB |\n|---|---:|---:|---:|---:|---:|\n| 32 worker / 32 CPU，私有 MPS | {r['passed_full_episodes']}/32 | {r['aggregate_all_steps_s']:.2f} | {r['aggregate_steps_from_accepted_episodes_s']:.2f} | {r['effective_episodes_hour']:.2f} | {r['peak_gpu_mib']/1024:.2f} |\n| 64 worker / 32 CPU，私有 MPS | 初始化/准备阶段失败 | 未测 | 未测 | 0 | {q['peak_gpu_mib']/1024:.2f} |\n\n32 环境运行 {r['duration_s']:.1f} 秒，平均单环境 {r['mean_per_env_steps_s']:.3f} steps/s，GPU 利用率均值 {r['mean_gpu_percent']:.1f}%。第 14 号 worker 在第三剪 tick 1124 出现一条 positional_nonmanifold_edges，只有该步质量失败；非法四面体、工具穿入和范围检查均未失败。该回合不计入合格产出，不能将 32 环境称为完整物理验收通过。\n\n64 个 worker 均通过 Python 导入屏障并实际进入原生初始化。16 个进程报告 LCS_Init -46，MPS 服务端直接记录最多 48 个客户端；另 14 个进程报告 LCS_Step -2（CUDA 显存分配失败）。34 个进程完成各自短 settle，不代表 64 环境共同成功运行。初始化时既有 MPS 连接数量限制，也遇到了显存容量限制。\n\nCUDA 状态码核对本机 targets/x86_64-linux/include/driver_types.h；MPS 限制以本次 server.log 为证。NVIDIA [MPS 文档](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)亦说明客户端上限及版本差异，不能把新驱动限制套用本机。未升级驱动或修改 MPS 全局配置。\n\n32/64 批次的所有进程及私有 MPS 均已退出，GPU 随后回到 15 MiB；两次单环境 reset/settle 通过。没有 GPU reset，没有终止其他任务。\n\n上一轮 16 环境 / 16 CPU 的 57.07 steps/s 对比本轮 32 环境原始 71.67 steps/s，仅是共享服务器上的观测，非严格配对因果比较。本轮合格回合折算为 69.43 steps/s。32 环境增量收益有限且已暴露拓扑失败，优先完成清单中的物理修复，再签署更大规模稳定性。\n\n原始证据：runs/m32a、runs/m64a、runs/post64-reset、mps/m64a/l/server.log、scaleup-summary.json。\n'''
    (ROOT/'SCALEUP_REPORT.md').write_text(text)
    print(json.dumps({k:{x:y for x,y in v.items() if x not in ('failed_episodes','worker_error_groups')} for k,v in result.items() if k in ('32','64')},indent=2))
if __name__=='__main__':main()
