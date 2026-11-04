"""A fail-closed adoption gate; throughput never substitutes for interactive FPS."""
import argparse,json
from pathlib import Path

def evaluate(baseline,candidate):
 reasons=[]
 for role,value in [('baseline',baseline),('candidate',candidate)]:
  if not value.get('completed') or not value.get('quality_passed'):reasons.append(role+': incomplete or rejected physics')
  render=value.get('render',{})
  if not (render.get('depth') is True and render.get('render_every')==1 and render.get('width',0)>0 and render.get('height',0)>0):
   reasons.append(role+': complete fresh RGB-D rendering not demonstrated')
  binding=value.get('render_device_binding',{})
  if binding.get('gpu_uuid')!=value.get('gpu') or 'egl_index' not in binding:
   reasons.append(role+': EGL/CUDA same-GPU binding missing')
  frames=value.get('frames',[])
  ticks=[f.get('native_tick') for f in frames]
  if not ticks or any(type(t) is not int for t in ticks) or len(set(ticks))!=len(ticks):
   reasons.append(role+': unique physical frames not demonstrated')
  perf=value.get('performance',{})
  if perf.get('frames')!=len(frames):reasons.append(role+': inconsistent measured frame count')
  if perf.get('unique_frames_hz',0)<20:reasons.append(role+': mean below 20 Hz')
  if perf.get('p95_ms',float('inf'))>50:reasons.append(role+': P95 slower than 50 ms')
  if (perf.get('rolling_20_frame_min_hz') or 0)<20:reasons.append(role+': sustained 20-frame window below 20 Hz')
 for key in ('scene','render','gpu','render_device_binding','supervision','opencv_threads'):
  if baseline.get(key)!=candidate.get(key):reasons.append('comparison mismatch: '+key)
 b,c=baseline.get('performance',{}),candidate.get('performance',{})
 if b.get('frames')!=c.get('frames'):reasons.append('different completed frame counts')
 if c.get('mean_ms',float('inf'))>b.get('mean_ms',0):reasons.append('candidate mean regressed')
 if c.get('p95_ms',float('inf'))>b.get('p95_ms',0):reasons.append('candidate slow-frame latency regressed')
 return dict(passed=not reasons,reasons=reasons,gate='20 Hz complete unique RGB-D frames, mean/P95/sustained windows and paired nonregression',
             note='A paired result is preliminary; ABBA repetition and interaction review still required before adoption')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
 print(json.dumps(evaluate(json.loads(a.baseline.read_text()),json.loads(a.candidate.read_text())),indent=2))
