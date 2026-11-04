"""Finalize navigational notes and provenance after successful offline delivery."""
from pathlib import Path
import json,time,hashlib,re
p=Path(__file__).resolve().parents[1];f=p/'REPORT.md';text=f.read_text();text=text.replace('`void k_UnifiedVertex<`','`k_UnifiedVertex`').replace('`void k_MaterialStencilAssemble<`','`k_MaterialStencilAssemble`')
needle='CPU与GPU时间线：'
paragraph='调用点对照捕获确认：每帧162次FindSurface中，135次来自剪刀路径，27次来自抓钳；相关实现见[当前原生接触核](native-gpu-diagnostics-v2/source/vendor/cuda/contact/surface_contact.cuh)。单独提前计算器械形状未带来整帧收益，不能将其视为这个热点已经解决。\n\n'
assert text.count(needle)==1;text=text.replace(needle,paragraph+needle)
needle='on-CPU包含驱动忙等，不等于纯CPU数学运算。'
text=text.replace(needle,'OpenGL工作记录存在未覆盖的帧区间，图中的GPU并集表示已记录活动的覆盖；不能据此推算精确硬件空闲时间或SM占用率。“无CUDA活动”也可能包含GL工作，不能直接称为CPU计算时间。\n\n'+needle)
needle='GPU拓扑库保留原308个设备函数SASS逐位一致；'
text=text.replace(needle,'真实切割状态的完整环境快照/恢复额外通过：重复GPU检查不改快照，继续执行后可逐位恢复原生状态，恢复后检查及下一步正常；新Python辅助模块均绑定散列。见[真实快照回归](results/snapshot-roundtrip-final/result.json)。这不证明未来轨迹逐位确定。\n\n'+needle)
needle='CPU负载在本轮从约60%降到约12%'
text=text.replace(needle,'建议的下一轮顺序是：先对FindSurface的多种器械查询合并调度做最小原型，保留全部查询并继续逐记录对照；再分析CoarseAccumulate/UnifiedInteractions的数据访问及融合空间；最后评估动态表面数量导致的图重建。当前图销毁5.075ms中4.631ms与CUDA活动重叠，单看API时长会高估可省预算。以上尚未实施，不作为本轮收益。\n\n'+needle)
f.write_text(text)
e=p/'EXPERIMENTS.md';text=e.read_text().replace('逐像素及完整场景对照正在排队，最终结论见报告','修正OWNDATA后4组RGB-D逐位一致，完整场景及最终两次1410帧回归通过；最终收益见报告')
text='最终状态：完整验收及报告已完成；最终代码见 final-source。稳定20Hz仍未通过。以下保留实验动机与过程，最终取舍以 REPORT.md 为准。\n\n'+text
text+='\n私有缓冲快照问题已单独定位并修复：初版只有5个新增拓扑派生缓冲改变，旧物理字段不变；修正版不把新临时工作区纳入物理快照，不改原快照实现/ABI，并通过完整快照/恢复和1410帧双算。最终物理核308个均与原库SASS一致。形状预计算虽960次零差异，但完整ABBA无收益，未保留。\n';e.write_text(text)
manifest=p/'delivery/manifest.json';d=json.loads(manifest.read_text());d['python_patch_apply_check_passed']=True;d['native_patch_apply_check_passed']=True;d['native_patch_sha256']=hashlib.sha256((p/'delivery/native-from-baseline.patch').read_bytes()).hexdigest();d['native_vendor_line_endings_note']='Exact patch retains original CRLF in deletion/context lines; final compiled source uses its recorded LF bytes.';d['snapshot_roundtrip_passed']=json.loads((p/'results/snapshot-roundtrip-final/result.json').read_text())['passed'];manifest.write_text(json.dumps(d,indent=2))
# Keep the previously opened profiling report intact, with a link to this turn.
prior=p.parent/'REPORT.md';old=prior.read_text();note='> 2026-11-04 后续激进优化已完成：同条件原版15.57/15.66Hz，候选21.45/21.42Hz，P95约57.1ms，稳定20Hz仍未通过。新报告与最终独立代码见[实时性专项报告](aggressive-20261104/REPORT.md)。正式版本未替换。\n\n'
if '后续激进优化已完成' not in old:
 title,sep,body=old.partition('\n');prior.write_text(title+'\n\n'+note+body.lstrip('\n'))
final=json.loads((p/'results/final-selection.json').read_text());integrity=json.loads((p/'results/source-integrity.json').read_text());assert final['completed'] and final['real_snapshot_roundtrip_passed'] and integrity['passed']
(p/'results/completion-status.json').write_text(json.dumps(dict(completed=True,finished_unix_s=time.time(),source=final['source'],production_replaced=False,all_requested_artifacts_written=True,source_integrity_passed=True,stable_20hz_passed=False,formal_physics_domain_approved=False),indent=2));print('Documentation, reviewable patches and completion status finalized.')
