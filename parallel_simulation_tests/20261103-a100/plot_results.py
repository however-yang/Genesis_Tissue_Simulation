"""Standalone plots of completed full runs only; no inferred untested points."""
import json
from pathlib import Path
import os
os.environ["MPLCONFIGDIR"]=str(Path(__file__).resolve().parent/"cache/matplotlib")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
rows=[r for r in json.loads((ROOT/'performance.json').read_text())
      if r['complete'] and r['passed']==r['episodes'] and r['stop_after']=='hold' and not r['instrumented']]
fig,axes=plt.subplots(1,3,figsize=(13,4))
for mps,label,color in [(False,'Separate CUDA contexts','#2563eb'),(True,'Private MPS','#ea580c')]:
    group=[r for r in rows if r['mps']==mps]
    if not group:continue
    x=[r['workers'] for r in group]
    axes[0].scatter(x,[r['aggregate_steps_s'] for r in group],label=label,color=color,s=55)
    axes[1].scatter(x,[np.mean(r['per_env_steps_s']) for r in group],label=label,color=color,s=55)
    axes[2].scatter(x,[r['peak_gpu_mib']/1024 for r in group],label=label,color=color,s=55)
for r in rows:
    if r['workers']==16:
        for ax,y in [(axes[0],r['aggregate_steps_s']),(axes[1],np.mean(r['per_env_steps_s']))]:
            ax.annotate(str(r['cpu_count'])+' CPU', (16,y), xytext=(5,5), textcoords='offset points',fontsize=7)
for ax,title,ylabel in zip(axes,['Aggregate throughput','Per-environment throughput','Sampled peak GPU memory'],
                          ['Simulation steps / second','Simulation steps / second','GiB']):
    ax.set(title=title,xlabel='Concurrent environments',ylabel=ylabel)
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32,64]);ax.set_xticklabels([1,2,4,8,16,32,64])
    ax.grid(alpha=.2);ax.set_ylim(0,ax.get_ylim()[1]*1.12)
axes[0].legend(fontsize=8)
fig.suptitle('A100 80GB PCIe: full wall-traction checks, shared-server CPU load',fontsize=12)
fig.tight_layout();fig.savefig(ROOT/'performance.png',dpi=180);fig.savefig(ROOT/'performance.svg')
