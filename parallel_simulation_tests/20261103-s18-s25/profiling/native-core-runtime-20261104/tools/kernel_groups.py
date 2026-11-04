"""Disjoint measured CUDA-kernel sums classified by explicit defining source.

A fused material/contact response belongs to the mixed solve bucket. The cut
bucket is direct cutting/reconstruction work, not all downstream cutting cost.
"""
import argparse,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'native/source/vendor/cuda'

def names(paths):
 found=set()
 for path in paths:
  text=(SOURCE/path).read_text()
  found.update(re.findall(r'__global__\s+(?:__launch_bounds__\([^\n]*?\)\s+)?void\s+(k_\w+)\s*\(',text))
 return found

def classify(summary):
 groups=[('direct_cut_topology_reconstruction',names(['cut.cu','dc_recon.cu','surface_cleanup.cuh'])),('quality_diagnostics',names(['gpu_body_check.cuh','topology_checks.cuh'])),('observation_diagnostics',names(['grasp/grip_observation.cuh','grasp/runtime_grip_feedback.cuh'])),('runtime_mode_decisions',names(['zero_mode_gates.cuh'])),('surface_contact_discovery_manifold',names(['contact/surface_contact.cuh','contact/contact_groups.cuh','contact/friction_contact.cuh'])),('fused_fem_constraint_solve',names(['solver/material/unified_vertex.cuh','solver/material/coupled_material.cuh','solver/material/material_stencil.cuh','solver/material/material_line_search.cuh','solver/material/material_multigrid.cuh','solver/material/pressure_patch_line.cuh','contact/coarse_bound_cache.cuh','contact/coarse_occupancy.cuh','grasp/tool_coupling.cuh','grasp/gaussian_grasp.cuh']))]
 result={key:dict(kernel_sum_ms_per_frame=0,calls_per_frame=0,members=[]) for key,_ in groups};result['other_kernels']=dict(kernel_sum_ms_per_frame=0,calls_per_frame=0,members=[])
 for kernel in summary['kernels']:
  match=re.search(r'\b(k_\w+)(?:<|\()',kernel['name']);name=match[1] if match else ''
  category=next((label for label,functions in groups if name in functions),'other_kernels')
  entry=result[category];entry['kernel_sum_ms_per_frame']+=kernel['sum_ms_per_frame'];entry['calls_per_frame']+=kernel['calls_per_frame'];entry['members'].append(kernel)
 total=sum(x['kernel_sum_ms_per_frame'] for x in result.values())
 for x in result.values():x['share_of_kernel_sum_pct']=100*x['kernel_sum_ms_per_frame']/total if total else 0
 return dict(kernel_sum_ms_per_frame=total,groups=result,meaning='Disjoint sums of measured kernel durations, not summed CPU/GPU critical path; direct_cut excludes downstream contact/repair costs caused by cutting')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('summary',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data=classify(json.loads(a.summary.read_text()));a.output.write_text(json.dumps(data,indent=2)+'\n')
 for k,v in data['groups'].items():print(k,round(v['kernel_sum_ms_per_frame'],4),round(v['share_of_kernel_sum_pct'],1))
