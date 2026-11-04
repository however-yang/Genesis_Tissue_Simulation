from pathlib import Path
import json,hashlib,time,subprocess,shutil,tempfile
ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
b=read('baseline.json');metrics=read('results/final-comparison.json');baseline=Path(b['baseline_experiment'])/'native-readonly-v8/source'
# Verify the supplied source patch reconstructs every retained source file.
with tempfile.TemporaryDirectory(prefix='patch-verify-',dir=ROOT/'tmp') as temp:
 target=Path(temp)/'source';shutil.copytree(baseline,target)
 patch=ROOT/'delivery/native-from-v8.patch'
 check=subprocess.run(['git','apply','--check',str(patch)],cwd=target,text=True,capture_output=True)
 assert check.returncode==0,check.stderr
 apply=subprocess.run(['git','apply',str(patch)],cwd=target,text=True,capture_output=True)
 assert apply.returncode==0,apply.stderr
 actual={str(x.relative_to(target)):sha(x) for x in target.rglob('*') if x.is_file()}
 expected={str(x.relative_to(ROOT/'native/source')):sha(x) for x in (ROOT/'native/source').rglob('*') if x.is_file()}
 assert actual==expected
(ROOT/'results/patch-reconstruction.json').write_text(json.dumps(dict(passed=True,files=len(expected),patch_sha256=sha(patch)),indent=2)+'\n')
assert all(sha(Path(b['borrowed_python_source'])/name)==digest for name,digest in b['baseline_python_files'].items())
assert all(sha(baseline/name)==digest for name,digest in b['baseline_native_files'].items())
assert sha(Path(b['baseline_library']))==b['baseline_library_sha256']
assert read('results/final-queue.json').get('finished_unix')
assert read('results/final-proof/result.json')['native_validation']['passed']
assert read('results/final-proof/result.json')['coarse_bound_validation']['passed']
assert read('results/final-rgbd-proof/result.json')['passed'] and read('results/final-snapshot/result.json')['passed']
assert read('results/cpu-regression.json')['returncode']==0
free=read('results/free-component-repeat-control/result.json')
assert free['physical_quality_passed'] and free['fallback_coverage_passed']
assert not free['strict_cross_run_determinism_passed']
readme='''# 本轮唯一候选：原生核心运行优化

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
'''
(ROOT/'README.md').write_text(readme)
(ROOT/'EXPERIMENTS.md').write_text('''# 实验状态：最终代码只看 native/source

最终选定 `native/binaries/runtime-core.so`，SHA 8b24b3f7…；保留图生命周期更新、80-register 原融合粗层核、原守卫禁止修正的门控、并行几何/有序 double 只读反馈。模式 43264。完整条件/结果见 [REPORT](REPORT.md)。

|尝试|状态与证据|
|---|---|
|图复用|保留；V1 原 332 设备函数机器码不变，最终有正常更新拒绝回退|
|直接接触记录构造|淘汰；4.743+0.117 ms，比 4.415 ms 查询更慢|
|粗层材料/接触拆核|淘汰；完整 native 29.115 ms 更慢|
|合批观察读回|淘汰；498 次字段字节一致但未证明收益，最终 native 接口已删除|
|80-register 粗层 + 有序反馈|保留；完整系数/反馈逐位对照通过，64-register 较慢分支已删除|
|零修正门控|保留为独立候选；最终完整原步骤 4,190,202,900 坐标值零位差异；自由小块 63 子步全部走原回退|
|32/128 query threads|不采用；均比 64 的完整运行更慢|
|夹爪/刀刃查询编译期特化|淘汰；完整 native 26.981 ms 未优于 V6 26.889 ms|
|最终 ABBA|4×1410 帧均过原局部质量；24.01→24.84 Hz，native 28.274→26.837 ms|
|最终实切 RGB-D、快照、CPU 回归|当前状态像素/深度、恢复存档及后续一步通过；CPU 136 项通过|
|补充自由组织重复性|未通过严格 1 μm 跨轨迹对照；V8 自重跑也失败，保留控制和失败轨迹，不批准全域无损|

最终稳定 25 Hz、慢窗口全部 20 Hz、完整物理域、未来轨迹确定性及训练域均未获批。上述淘汰方向的源码不在保留 native/source 中；历史生成器、补丁、冻结库、日志仍可审计。首次自由测试的 Python 字段错误与随后真实轨迹阈值失败在报告中分别说明。
''')
manifest=dict(schema='native_runtime_candidate_v1',created_unix=time.time(),status='independent measured candidate; local held-task regressions pass; performance floor and full physics not approved',selected_library='native/binaries/runtime-core.so',library_sha256=sha(ROOT/'native/binaries/runtime-core.so'),native_source='native/source',core_mode=43264,core_bits={'graph_reuse':256,'ordered_parallel_grip_diagnostic':2048,'coarse_80_register_budget':8192,'original_disabled_modes_gate':32768},query_threads=64,query_mode=3,assembly_blocks=512,manifold_blocks=1024,coarse_bound_cache=True,gpu_topology_checks=True,gpu_body_checks=True,batch_observation=False,render={'width':1280,'height':960,'fresh_cpu_rgbd_every_step':True,'dense_s25_labels_included':False,'desktop_transport_included':False},stable_25hz_approved=False,all_20frame_windows_above_20hz=False,formal_physics_domain_approved=False,training_domain_approved=False,strict_free_trajectory_reproduction_approved=False,formal_version_changed=False,current_navigation_changed=False,baseline_library=b['baseline_library'],baseline_library_sha256=b['baseline_library_sha256'],borrowed_python_source=b['borrowed_python_source'],native_source_sha256={'native/source/'+name:digest for name,digest in sorted(expected.items())},tool_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'tools').glob('*.py'))},regression={'full_unprofiled_abba_frames':5640,'local_quality_checks_passed':True,'full_numeric_proof':'results/final-proof/result.json','rgbd':'results/final-rgbd-proof/result.json','snapshot':'results/final-snapshot/result.json','cpu_tests':136,'free_repeat_failed':'results/free-component-repeat-control/result.json','patch_reconstruction':'results/patch-reconstruction.json'},performance=metrics,build='results/build-retained-runtime.json',native_patch_sha256=sha(ROOT/'delivery/native-from-v8.patch'),report_sha256=sha(ROOT/'REPORT.md'))
(ROOT/'delivery/manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print('Delivery manifest written. Source patch reconstructs',len(expected),'files; protected baseline unchanged. Known free determinism failure retained.')
