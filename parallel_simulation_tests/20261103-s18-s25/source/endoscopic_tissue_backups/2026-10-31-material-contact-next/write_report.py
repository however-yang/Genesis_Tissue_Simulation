"""Generate a source-backed Chinese experiment report after artifact verification."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
DATA=ROOT/'endoscopic_tissue/outputs/material-contact-next-20261031'
e=json.loads((WORK/'evidence-summary.json').read_text(encoding='utf8'))
series={s['name']:s for s in e['series']};caps={c['name']:c for c in e['captures']}
def path(p):return str(p).replace(chr(92),'/')
def link(title,p):return '['+title+']('+path(p)+')'
lines=['# 材料求解与接触分组实验：2026-10-31','',
'本轮完成两个独立算法实验，均未替换默认版本；超过 10 万节点接近 33 Hz 的目标尚未达成。上一轮联合粗模态的两个版本继续不采用。本轮两种算法也没有叠加。','',
'起点为 dev fbcaf71dc0eec30e00b55d46bf5ae0eddac271b3。默认夹爪/剪刀和单刀 DLL、生产 CUDA 源保持原版。实验补丁、数值探针、失败输入小型夹具、测量脚本及汇总保存于 '+link('实验目录',WORK/'README.md')+'。','',
'## 模型与测量口径','',
'RTX 3090，肾脏场景，1 mm 网格；354,200 个网格槽，121,260 个活动节点，102,710 个材料节点，118,660 个可动活动节点。每次运行 1,110 步，保持 dt=1/30 s、3 子步、Young=15,000 Pa、Poisson=0.25、strain damping=0.04 s，受载/自由 PCG 预算保持 3/7 次。没有减少细网格自由度、降低接触精度或调整材料参数。','',
'每项按 A1–B1–B2–A2 顺序串行测量普通 CUDA Graph 版本；诊断、实际状态捕获、渲染、模块计时分开。核心是 LCS_Step；pipeline 还包括控制、读回、记录提交及可能的背压，不含显示窗口、阶段边界检查或最终排空。33 Hz 的单步预算约 30.30 ms。以下不是 GUI 帧率。','',
'|独立候选|核心：基线→候选 ms|变化|pipeline：基线→候选 ms|候选核心/pipeline Hz|','|---|---:|---:|---:|---:|']
for name,title in [('additive','材料加性多层'),('groups','接触分组 AABB')]:
    s=series[name];a=s['means']['core_ms'];b=s['means']['pipeline_ms']
    lines.append(f"|{title}|{a['baseline']:.3f} → {a['candidate']:.3f}|{a['change_percent']:+.2f}%|{b['baseline']:.3f} → {b['candidate']:.3f}|{a['candidate_hz']:.2f} / {b['candidate_hz']:.2f}|")
lines+=['','负值表示耗时减少。每侧只有两次完整计时，不把小于运行波动的差别宣称为稳定加速；详细逐阶段均值、P95、最大值及原始文件哈希见 '+link('evidence-summary.json',WORK/'evidence-summary.json')+'。','',
'|实验/运行|核心均值 ms|核心 P95 ms|pipeline 均值 ms|交互验收|','|---|---:|---:|---:|---|']
for s in e['series']:
    for r in s['runs']:lines.append(f"|{s['name']}/{r['name']}|{r['core_ms']['mean']:.3f}|{r['core_ms']['p95']:.3f}|{r['pipeline_ms']['mean']:.3f}|{'passed' if r['quality_passed'] else 'failed'}|")
lines+=['','接触 B1 从第 748 步起到 1110 步连续 363 步超过 100 ms，第二剪后常达 210–230 ms。B2 没有复现，核心均值为 44.70 ms；相对该组基线均值 44.35 ms，也没有加速依据。上表原样保留异常运行，未删除异常值或只选快的一次。', '', '## 材料实验：加性多层预条件器','',
'把每次 PCG 中的乘性 V(1,1) 改为加性多层递推：B_l = 0.7 D_l^-1 + P_l B_(l+1) P_l^T。保留原 Galerkin 算子、按切割连通分量建立的层次、最粗层逆及细层自由度；各层限制原残差并累加修正，省去 V-cycle 中反复遍历 A*x。原材料目标、工具耦合、可行性检查和线搜索保留。公式参考 [NGSolve 官方教程](https://docu.ngsolve.org/latest/i-tutorials/unit-2.1.3-multigrid/multigrid.html)，其椭圆问题收敛结论不能直接套用到本系统。','',
'初版在第二剪后的第 800 步冻结输入检查失败：相对 L2 误差 1.2003e-4，超过事先设定的 1e-4。主要误差集中于切口附近一个条件数约 17,634 的 3×3 细块；保留初版和失败证据。候选局部求逆改用 double 中间量，未修改全局材料矩阵。1,038 个实际 GPU 块测试对照独立 double 求解，新版最大相对误差 5.60e-8；原求逆仍在该回归中暴露误差。阈值没有放宽。','',
'修正后的完整轨迹在十个冻结状态与 CPU 独立递归一致：最大相对误差 '+format(e['additive_frozen_checks']['max_relative_error'],'.3e')+'，最大归一化对称误差 '+format(e['additive_frozen_checks']['max_symmetry_error'],'.3e')+'，被测随机输入的二次型为正，固定节点输出为零。这证明抽样预条件器计算正确，不证明有限次 PCG 的动态效果等价。','',
'完整视频中夹持、提拉、两次剪切和取出均完成；第二剪附近局部连接/分离几何出现差别。夹持点 55→59；首剪和第二剪的刀刃载荷曲线改变。最后保持阶段肿瘤区域 RMS 速度峰值 0.1596→1.0536 mm/s，均值 0.0600→0.2431 mm/s；非切割器械最大重叠 0.0250→0.0784 mm。这些是需要解释的变化，不能用第二剪速度峰值下降或切痕数量减少抵消。材料候选暂不采用。','',
'## 接触实验：保守三角形分组','',
'每 64 个连续三角形建立当前/上一位置的合并 AABB；六个表面点更新位置均同步重建。每次夹爪或剪刀查询重新计算分组排除标志。盒形器械保留端点 AABB，并额外合并两端包围球与浮点外扩，覆盖旋转扫掠。旧刀刃预载绕过早退。原窄阶段函数体仅增加可选分组早退，没有候选截断、接触简化、摩擦替换或跨步缓存。','']
c=e['contact_frozen_checks']
lines += [f"270 个实际冻结状态（600–810 步逐步检查，其余每 15 步）累计 {c['triangle_queries']:,} 次三角形查询，分组排除 {c['group_rejected_queries']:,} 次（{100*c['group_rejected_queries']/c['triangle_queries']:.2f}%）；有效接触记录、数量、节点 degree、活动成员及原逐三角形 AABB 保守性均为 0 差异。比较使用独立冻结参考查询及私有 scratch。",
'',f"预载绕过计数为 {c['preload_bypasses']}：本场景没有触发旧预载落在被排除整组之外的情况，不能声称该边界已被动态测试覆盖。同输入一致也不代表不同运行的原子累加顺序和完整轨迹逐位一致。",'','## 完整交互录像与质量','',
'每项各新录制基线和候选，共四份 1,111 帧实际状态录像和两份完整同步对照。30 fps 表示模拟时间播放速度；不是执行速度。观察基于夹持/提拉、第一次剪切、第二次剪切/撤离等匹配帧组，以及覆盖所有步的运动与载荷曲线。全部视频另做逐帧解码检查，未把关键帧代替完整视频。','',
'|捕获|提拉 mm|夹持点|切断边|额外皮肤切痕|最大非切割重叠 mm|验收|','|---|---:|---:|---:|---:|---:|---|']
for name,item in caps.items():
    m=item['metrics'];lines.append(f"|{name}|{1000*m['lift_m']:.3f}|{m['final_grip_points']}|{m['cut_edges']}|{m['stray_skin_cuts']}|{1000*m['max_blade_body_overlap_m']:.4f}|{'passed' if item['quality_passed'] else 'failed'}|")
am=json.loads((DATA/'additive/additive-review/motion-comparison.json').read_text(encoding='utf8'))
gm=json.loads((DATA/'groups/groups-review/motion-comparison.json').read_text(encoding='utf8'))
a_gap=am['stages']['incise_1']['jaw_gap_mm'][0]['mean'];g_gap=gm['stages']['incise_1']['jaw_gap_mm'][0]['mean']
a_hold=am['stages']['hold']['tumor_rms_speed_mm_s'][0]['maximum'];g_hold=gm['stages']['hold']['tumor_rms_speed_mm_s'][0]['maximum']
lines+=['',f'两份新基线录像之间也有运行差异：首剪夹爪平均开度 {a_gap:.3f} vs {g_gap:.3f} mm，保持阶段肿瘤 RMS 速度峰值 {a_hold:.4f} vs {g_hold:.4f} mm/s。两次基线不足以给出完整波动范围，但足以说明不能将单次轨迹的每个差异都归因于候选。当前没有证明这种变化具体来自哪一个并行累加或分支；不能把这个解释当成已定位的根因。',
'',f"接触录像首剪夹爪平均开度为 {gm['stages']['incise_1']['jaw_gap_mm'][0]['mean']:.3f} → {gm['stages']['incise_1']['jaw_gap_mm'][1]['mean']:.3f} mm；第二剪肿瘤 RMS 速度峰值为 {gm['stages']['incise_2']['tumor_rms_speed_mm_s'][0]['maximum']:.3f} → {gm['stages']['incise_2']['tumor_rms_speed_mm_s'][1]['maximum']:.3f} mm/s。只记录差别，不把速度峰值下降直接判为更真实。"]
for name,title in [('additive','材料'),('groups','接触')]:
    folder=DATA/name/(name+'-review')
    lines+=['',link(title+'完整并排视频（H.264）',folder/'comparison-h264.mp4')+' · '+link('原始编码',folder/'comparison.mp4')+' · '+link('运动与载荷曲线',folder/'motion-comparison.png')+' · '+link('逐阶段运动指标',folder/'motion-comparison.json')]
lines+=['','基线本身的完整质量验收也未通过，不能将此作为放宽候选门槛的理由。沿用现有诊断入口，原 non-manifold 计数另存 recorded_nonmanifold_edges 后继续收集完整流程；这不构成拓扑通过。kidney 的 release 只是剪刀张开，夹爪仍保持夹持；真正松夹回弹、不同进刀角度和多场景重复尚未完成。本轮没有证明与真实手术视频等价，也没有把明显能形变当成真实性验收。','',
'## 模块诊断','',
'下表来自单独运行的 section profiler。关闭 Graph 后包含发射间隙，只用于定位；cg assembly 子项与外项重叠，不能把所有条目相加。','',
'|模块 ms/step|基线|材料候选|接触候选|','|---|---:|---:|---:|']
sections=e['section_diagnostics']['runs']
for key in ['cg: assembly','cg: iterations','loop: local sweeps','loop: coarse modes','loop: contact discovery','loop: manifold, contact marks','pre: tool, contact discovery, manifold']:
    lines.append('|'+key+'|'+ '|'.join(f"{sections[n]['sections_ms_per_step'][key]:.3f}" for n in ('baseline','additive','groups'))+'|')
lines+=['','## 第二剪后慢路径的独立复现','',
'另构建只增加 CPU 执行图选择计数的诊断 DLL，记录 loaded/free/recovery。若第 680 步后连续三步超过 100 ms，保存 GPU 状态与进程清单，并将后续四步切换到 section profiling。此跟踪会改变诊断期间的执行路径，只用于定位，不计入正常 ABBA。','',
'|复现|loaded/free/recovery 次数|未开 profiler 最大 ms|触发四步采集|','|---|---|---:|---|']
for trial in e['slow_graph_diagnostics']:
    g=trial['graphs'];lines.append(f"|{trial['repeat']}|{g['loaded']}/{g['free']}/{g['recovery']}|{trial['maximum_unprofiled_ms']:.3f}|{trial['profile_triggered']}|")
if not any(t['profile_triggered'] for t in e['slow_graph_diagnostics']):
    lines+=['','这些有限次诊断没有复现连续慢段，原始 B1 异常原因仍未确定，不能据此声称已修复。']
else:
    lines+=['','触发时的原始图计数、GPU 状态、四步模块耗时均保留在 graph-diagnostics-summary.json 所指目录；是否能定位根因须结合这些证据，不能仅凭触发恢复图就断言因果。']
lines+=['','## 决策与后续边界','',
'两项保留为可复现实验，不替换默认版本。下一步应优先在同一冻结系统上衡量预条件器的残差衰减和局部接触位移误差，解释载荷/保持阶段变化，再考虑材料算法的组合；接触粗筛需要补齐分组外持续预载的专门场景。达到 33 Hz 必须同时压低材料、接触及重建开销，不能把模块耗时下降直接宣称为整帧达标。任何采用仍需真实松夹回弹、不同器械姿态及多场景重复交互验证。','',
'默认剪刀 DLL SHA256：'+e['protected_dll_sha256']['bin-scissors']+'。默认单刀 DLL SHA256：'+e['protected_dll_sha256']['bin']+'。完整证据核验脚本见 '+link('verify_artifacts.py',WORK/'verify_artifacts.py')+'。','']
(ROOT/'endoscopic_tissue/docs/material-contact-experiments-2026-10-31.md').write_text(chr(10).join(lines),encoding='utf8')
print('Wrote verified experiment report')
