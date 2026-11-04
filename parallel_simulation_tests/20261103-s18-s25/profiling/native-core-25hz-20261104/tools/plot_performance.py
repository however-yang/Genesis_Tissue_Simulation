import os,json
from pathlib import Path
os.sched_setaffinity(0,[30,31]);p=Path(__file__).resolve().parents[1];out=p/'results/figures';out.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(out/'matplotlib-cache')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
s=json.loads((p/'results/final-summary.json').read_text());labels=['Mean','P50','P95','P99'];keys=['mean_ms','p50_ms','p95_ms','p99_ms'];x=np.arange(4);fig,ax=plt.subplots(figsize=(8,4.2))
for shift,color,kind in [(-.18,'#7b8795','reference'),(.18,'#16867c','readonly')]:ax.bar(x+shift,[s[kind][k] for k in keys],width=.35,label=kind,color=color)
ax.axhline(40,color='#b63c3c',linestyle='--',label='25 Hz = 40 ms');ax.axhline(50,color='#c99733',linestyle=':',label='20 Hz = 50 ms');ax.set_xticks(x,labels);ax.set_ylabel('Full RGB-D frame latency (ms)');ax.set_title('ABBA: 2 x 1410 frames per variant, no slow frames removed');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'latency-comparison.svg');fig.savefig(out/'latency-comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4.5))
for name,color in [('readonly-abba-0-reference-a','#999999'),('readonly-abba-3-reference-b','#697582'),('readonly-abba-1-readonly-a','#2d9b84'),('readonly-abba-2-readonly-b','#147568')]:
 r=json.loads((p/'results'/name/'result.json').read_text());ax.plot([f['frame_id'] for f in r['frames']],[f['frame_ms'] for f in r['frames']],lw=.6,alpha=.65,label=name.split('abba-')[1],color=color)
ax.axhline(40,color='#b63c3c',ls='--',label='25 Hz budget');ax.axhline(50,color='#c99733',ls=':');ax.set_ylim(bottom=0);ax.set_xlabel('Frame index (all 1410 frames)');ax.set_ylabel('Wall-clock ms');ax.set_title('Unfiltered full-frame timings, including initialization-related slow frames');ax.legend(fontsize=8,ncol=3);fig.tight_layout();fig.savefig(out/'all-frame-times.svg');fig.savefig(out/'all-frame-times.png',dpi=160);plt.close(fig)
print(out)
