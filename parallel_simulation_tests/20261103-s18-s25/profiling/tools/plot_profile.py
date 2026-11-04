import json,sqlite3,os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parents[1]/'tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
r=Path(__file__).resolve().parents[1];d=r/'results';out=r/'figures';out.mkdir(exist_ok=True)
for key,db in [('baseline','baseline-nodes.sqlite'),('labels','baseline-labels-trace.sqlite')]:
 c=sqlite3.connect(d/db);c.row_factory=sqlite3.Row
 frames=c.execute("select start,end from NVTX_EVENTS where text='frame' order by start").fetchall();lo=frames[0]['start'];hi=frames[min(len(frames)-1,2)]['end']
 colors=['#2070b4','#358c71','#b163bc','#c68620','#8c6358','#348dac','#6c69ad','#d65846']
 lanes=[('environment.step','CPU API: environment (inclusive)'),('native.step','CPU API: native (includes waits)'),('quality.all','CPU quality checks'),('render.update_mesh','CPU mesh update + GL calls'),('render.rgbd','CPU render API (includes waits)'),('labels.compose_pixel_supports','CPU label composition')]
 fig,ax=plt.subplots(figsize=(14,6));labels=[];idx=0
 for text,label in lanes:
  ranges=[((max(x['start'],lo)-lo)/1e6,(min(x['end'],hi)-max(x['start'],lo))/1e6) for x in c.execute('select start,end from NVTX_EVENTS where text=? and start<? and end>?',(text,hi,lo))]
  if not ranges:continue
  ax.broken_barh(ranges,(idx-.32,.64),facecolors=colors[idx%len(colors)],edgecolors='none');labels.append(label);idx+=1
 for table,label in [('CUPTI_ACTIVITY_KIND_KERNEL','GPU: CUDA kernels'),('CUPTI_ACTIVITY_KIND_GRAPH_TRACE','GPU: CUDA graph envelope'),('CUPTI_ACTIVITY_KIND_MEMCPY','GPU: CUDA transfers'),('OPENGL_WORKLOAD','GPU: OpenGL workloads')]:
  if not c.execute("select 1 from sqlite_master where name=?",(table,)).fetchone():continue
  ranges=[((max(x['start'],lo)-lo)/1e6,(min(x['end'],hi)-max(x['start'],lo))/1e6) for x in c.execute(f'select start,end from {table} where start<? and end>?',(hi,lo))]
  if not ranges:continue
  ax.broken_barh(ranges,(idx-.32,.64),facecolors=colors[idx%len(colors)],edgecolors='none');labels.append(label);idx+=1
 for i,f in enumerate(frames[:3]):
  ax.axvline((f['start']-lo)/1e6,color='grey',lw=.6);ax.text((f['start']-lo)/1e6,-.6,f'frame {470+i}',ha='left',fontsize=9)
 ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,(hi-lo)/1e6);ax.set_xlabel('Milliseconds from capture start');ax.set_title('Measured timeline: '+key+' (CPU API ranges overlap GPU activity; do not sum lanes)');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/(key+'-timeline.svg'));fig.savefig(out/(key+'-timeline.png'),dpi=160);plt.close(fig)
print(out)
