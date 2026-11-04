from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent;project=h.parents[2]
def read(p):return json.loads((h/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
profile=read('recovery-run/r3-complete-frame-profile/attempt-000/profile/result.json')
full=read('recovery-run/r3-complete-frame-profile/attempt-000/profile/full-run/report.json')
hot=read('training-adapter-run/prepared-restore-abba/attempt-000/diagnostic/result.json')
raw=read('diagnostic-followup-run/r3-restore-profile/attempt-000/diagnostic/result.json')
tree=read('diagnostic-followup-run/fragment-tree-equivalence/attempt-000/probe/result.json')
r9=read('r9-run/r9-regrasp-matrix/attempt-000/matrix/result.json')
policy_path=h/'training-adapter-run/two-env-policy-loop/attempt-000/pilot/result.json'
policy=json.loads(policy_path.read_text()) if policy_path.exists() else None
cases=[]
for path in sorted((h/'r9-run/r9-regrasp-matrix/attempt-000/matrix').glob('regrasp-r*/result.json')):
 d=json.loads(path.read_text());item=dict(case=path.parent.name,passed=d['passed'],restore_ms=d['restore_ms'])
 for branch in ['original','restored']:
  rows=json.loads((path.parent/(branch+'-steps.json')).read_text());item[branch]=dict(failed_quality_steps=sum(not x['quality']['checked_metrics_passed'] for x in rows),max_blade_overlap_mm=1000*max(x['quality']['metrics']['blade_body_overlap_m'] for x in rows))
 cases.append(item)
summary=dict(complete_checklist=False,default_solver_replaced=False,queue_tests_passed=5,prepared_snapshot_cpu_tests_passed=4,recovered_video_frames=466,profiled_ticks=[f['native_tick'] for f in profile['frames']],r9_candidate_adopted=False,r9_matrix=dict(completed_pairs=r9['completed_pairs'],required_pairs=20,passed=r9['passed'],cases=cases),snapshot_abba=hot,fragment_tree=dict(tests=tree['tests'],passed=tree['passed'],integrated=False),topology_reproduction=read('topology-interruption.json'),ccd=read('ccd-endpoint-witness/result.json'),policy_loop=({k:v for k,v in policy.items() if k!='events'} if policy else {'status':'pending'}))
(h/'continuation-final-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 剩余清单续做报告（2026-11-03）','', '**清单尚未全部完成。** 新增可复用的监督/恢复工具、热快照缓存和最小策略管线；物理阻塞原样保留，默认求解器未替换。','', '## 版本边界','', '| 实验 | 原生库 | 结论适用范围 |','|---|---|---|','| 32/64 与策略管线 | 原始 contact-groups31，SHA a12b75cd…89cf2 | 原壁面三剪、局部微小牵拉分别验收 |','| 完整帧剖析、失败快照、热缓存 | R3，SHA 5f89e18a…448b | 诊断及适配器正确性，完整任务仍失败 |','| 顶点可行方向实验 | R9，SHA a8f33b88…6ec5d | 第二对换抓失败，不采用 |','', '## 已完成的工程修复','', '- `acceptance_queue.py` 使用子日志/产物进度、不可覆盖 attempt、同计划恢复、输入和完成证据哈希、进程身份校验、GPU 资源保护。5 项真实子进程回归通过。原错误尝试仍留在历史中。','- 最后中断视频恢复完成：466 帧全部解码通过，3 张 storyboard 已抽帧审阅，原始文件哈希不变。','- 完整肾脏轨迹 1320 步及七个完整原生帧的 CPU/CUDA 关联追踪完成。执行完成与物理通过分开记录。','', '## 32/64 及拓扑重复','', '首次 32 环境完整完成，31/32 通过；原始总吞吐 71.67 steps/s、合格回合折算 69.43 steps/s、173.58 episodes/h、峰值 57.84 GiB。第三剪 tick 1124 的一条非流形边仍是有效失败。64 个 worker 确实进入初始化；本机 MPS 日志给出 48 客户端上限，16 个初始化失败，另 14 个 CUDA 显存失败，只有 34 个短 settle 完成，不能称 64 环境运行成功。详见 `../SCALEUP_REPORT.md`。','', '为捕获拓扑问题新增 `topology_capture_worker.py`，在原质量判定后保存首失败完整网格。t32 诊断重复到约 tick 1110–1200 时出现未知 PID 3203533（12 MiB CUDA 占用），资源保护只停止自有 worker；私有 MPS 清理完成，后续链正确停止。该进程随后消失，身份无法追溯，不将它归因于某个用户。这个中断批次没有完整合格回合，也未捕获失败网格。','', '## 换抓：R9 实验拒绝','', '两份 R3 原始/恢复失败快照按原库逐字节导入；额外八次接触修复分别将刀体穿入从 0.204→0.217 mm、0.216→0.196 mm。仅增加修复次数不足以解决。','', 'R9 只在重建接触修复中试验受体积约束的顶点下降方向：原二次模型、力律、材料、时间步与 J 下界保持；所有相邻活四面体重新约束，模型能量不优于原步时回退。72 例 GPU 代数测试通过，但不能推出物理正确。','', '| 对号 | 原始分支失败步 / 最大穿入 mm | 恢复分支失败步 / 最大穿入 mm | 结果 |','|---|---|---|---|']
for c in cases:lines.append(f"| {c['case']} | {c['original']['failed_quality_steps']} / {c['original']['max_blade_overlap_mm']:.6f} | {c['restored']['failed_quality_steps']} / {c['restored']['max_blade_overlap_mm']:.6f} | {'通过' if c['passed'] else '拒绝'} |")
lines+=['','矩阵仅完成 2/20 对，第二对拒绝后停止；剩余 18 对没有声称已测。失败状态保存在 r9-run 中，原 0.2 mm 门限未变。R9 未采用，不能将第一对成功拼接为修复完成。','', '## 快照：显式热缓存','', '`PreparedSnapshot` 是隔离的可选适配器，保留原压缩文件和 SHA、原版本绑定及解压边界检查；缓存为不可变 bytes，构造前检查显式主机内存限额。4 项 CPU 合同测试通过。','', f"同库 ABBA 共 12 次恢复均与完整原生状态逐字节一致。冷路径指内存中的压缩快照即时解压，并非磁盘冷启动；热路径提前保留原生解压字节。一次准备 {hot['prepare_ms']:.1f} ms，额外原生状态约 {hot['native_bytes']/2**30:.3f} GiB，原快照和缓存合计保留 {hot['retained_host_bytes']/2**30:.3f} GiB；缓存释放后 RSS 从约 {hot['rss_after_prepare']/2**30:.3f} 回落至 {hot['rss_after_cache_release']/2**30:.3f} GiB。",'', '| 路径 | 数量 | 中位数 ms | P95 ms | 最大 ms |','|---|---:|---:|---:|---:|']
for mode,label in [('cold','即时解压'),('hot','显式热缓存')]:v=hot[mode];lines.append(f"| {label} | {v['count']} | {v['median_ms']:.2f} | {v['p95_ms']:.2f} | {v['max_ms']:.2f} |")
lines+=['','热缓存降低了重复分叉耗时，但 P95 仍超过一秒；冷恢复目标也未关闭。不能省略准备时间或内存成本后声称原快照全面达标。','', '## 帧性能与等价优化探针','', '| native tick | 剖析中原生步进墙钟 ms | GPU kernel 活跃并集 ms | 无 API 归属 kernel 数 |','|---:|---:|---:|---:|']
for f in profile['frames']:lines.append(f"| {f['native_tick']} | {f['profiled_frame_wall_ms']:.2f} | {f['kernel_union_active_ms']:.2f} | {f['unattributed_kernel_launches']} |")
lines+=['','CPU/GPU 两列不能相加。CUPTI、状态导出及共享 CPU 负载使这些值不能用于无记录 FPS 签署；部分条件图 kernel 的 API 关联缺失已保留。33 Hz 未达到。','', '针对追踪中显眼的游离块接触树更新，写了保持相同子节点顺序和浮点运算的顶层合并 kernel。16 组冻结输入（含无查询）逐字节相同。CUDA event、图执行 ABBA 中，1024 叶树从 0.02184 降至 0.01520 ms，仅节省约 0.00665 ms/次；说明不能直接将剖析的热点时间解释成可消除的整步耗时。收益不足以据此整合新的正式候选。','', '## 纯摩擦和连续碰撞诊断','', '- 原 R3 cut_ready 失败并非从未夹住：close 阶段已有双侧压力，最终四个见证点仍全部保留，切向滑移约 0.0238 mm；进刀后某侧载荷 0.0486 N 低于原 0.05 N 判据。失败原样保留，后续应区分建立、滑移和载荷不足。','- `ccd-endpoint-witness/result.json` 给出运动三角形反例：顶点线性移动最多 0.835 mm；点在 t≈0.2023、0.8946 两次穿过三角形内部，两端距平面均大于 0.03 mm 皮肤层，源码端点筛选会跳过。已保存距离图。它是代数反例，不是声称肾脏轨迹实际发生漏碰；全求解器还有终态包含修复，通用 CCD 保证仍不能成立。','', '## 两环境采样—更新—回评','']
if policy and policy.get('passed'):
 lines += [f"已完成 {policy['validated_recorded_windows']} 个记录后校验的转移，4 个独立回合：前 2 个回合采集 64 转移，CPU 上做 250 次小型模仿更新，数字数组检查点重载推理完全一致，再由后 2 个新回合回评 32 转移。每个动作绑定 policy_version、episode_id、worker、上下文 SHA；沿用公共 env.step、原质量终止和有界 IPC。",'',f"训练 MSE {policy['training']['loss_before']:.6g}→{policy['training']['loss_after']:.6g}；这些只是管线证据。动作是已持有状态下最大 20 μm 的 x 指令增量，目标包络 ±0.2 mm、无切割。它不是完整切割策略、PPO 收益、世界模型或训练域认证。"]
elif policy:lines += ['管线实际尝试未通过，保留 result.json 与异常；不能称闭环验证成功。']
else:lines += ['代码和计划已准备，执行状态以 training-adapter-run/state.json 为准；尚未计入完成。']
lines+=['','## 未完成与实施次序','', '1. 修复换抓后接触/体积可行性，以及切缘/辅助壳层拓扑；基于捕获的真实失败状态验证，不能靠改门限或增加隐藏锚点。','2. 在同一最终候选上完成 20 对换抓、纯摩擦准备重复、释放扰动/连续碰撞、完整视频及无录像交替性能；当前 R9 拒绝，不能移植其单对成功。','3. 再完成 S16 tear/force 与纯摩擦各 900 次矩阵；本轮没有启动这 1800 次，也没有把已排队当作完成。','4. 完整任务质量可靠后再作 33 Hz 等价优化和 32 环境长时间稳定性回归；64 单卡当前受客户端上限与显存双重限制。','5. 当前实际可用的方案仍是多进程隔离。A/B 原生批量和双 GPU 联合吞吐未实现/验证；GPU0 既有任务未动。S14 真实双臂仍按原决定排除。钳口切向载荷导出仍为可选缺口。','', '可复用命令和所有队列入口见 `CONTINUATION_README.md`。所有新代码、日志、录像和结果均在专用测试目录内；默认源码/库、网络配置和其他进程未改。']
(h/'CONTINUATION_REPORT.md').write_text('\n'.join(lines)+'\n')
# The user checklist preserves the original audit and failed candidates.
p=project/'docs/remaining-work-2026-11-03.md';s=p.read_text()
s=s.replace('72 例 GPU 代数核验通过，20 对完整换抓矩阵执行中，未采用。','72 例 GPU 代数核验通过；换抓矩阵第 1 对通过，第 2 对原始分支 58 步失败、最大穿入 0.23934 mm，2/20 对后拒绝停止，R9 不采用。')
s=s.replace('新队列的 R9 换抓矩阵有独立进度','新队列的 R9 换抓矩阵已拒绝停止')
s=s.replace('三类 T4、T7 和 T8 已有通过记录 | 主要缺换抓后 T4 的合格分布；所有证据须绑定最终同一候选。恢复目标 ≤1 秒还未达到，详见下表','三类 T4、T7 和 T8 已有通过记录；新增显式热快照缓存，12 次同库恢复逐字节一致 | 原库 ABBA 即时解压/热缓存中位数 1.722/0.905 秒，热缓存 P95 1.140 秒，≤1 秒仍未达标；换抓后 T4 分布仍缺')
if policy and policy.get('passed'):
 s=s.replace('现有外部策略入口、回合记录和读取器可复用；本轮未完成 RL/VLP/WAM 并行采样→推理→更新→回评闭环验证','已完成局部两环境采样→CPU 模仿更新→新回合回评的管线冒烟测试：4 回合、96 个校验转移；完整切割 RL/VLP/WAM 尚未验证')
 s=s.replace('先小规模、固定可用任务，明确策略版本、回合/分支/几何谱系、异常样本隔离、队列限流与中断恢复；本次不启动训练','局部管线已记录策略版本、回合、上下文哈希，保持原质量终止与队列限流；下一步是最终可靠任务上的正式学习，不能把这次小型模仿更新称作完整策略训练成功')
marker='## 本轮补充边界\n'
if marker in s:s=s.split(marker)[0]
s+='\n'+marker+'\n- t32 的完整网格复现批次因临时额外 CUDA 进程被资源保护中止，未算作通过，尚未捕获前次非流形边。\n- 连续碰撞已取得运动三角形端点筛选漏过两次中间穿越的代数反例；未宣称实际肾脏轨迹发生该反例，通用 CCD 尚未完成。\n- 接触树合并 16 组冻结输入逐字节一致，但 CUDA event 实测收益很小，未整合；原 33 Hz 缺口保留。\n- 当前所有完成数量、适用库、失败和热缓存内存成本见 [续做报告](../parallel_simulation_tests/20261103-scaleup/continuation/CONTINUATION_REPORT.md)。\n'
p.write_text(s)
print('Reports synchronized; complete_checklist=False; R9 rejected, no default adoption')
