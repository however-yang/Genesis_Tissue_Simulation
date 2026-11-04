"""Assemble a minimal serial candidate; do not carry rejected experiments forward.

Run only after measurement chooses the exact optional RGB/topology configuration.
The newly assembled source requires its own regressions and full benchmarks.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parents[1];a=argparse.ArgumentParser();a.add_argument('--rgb',action='store_true');a.add_argument('--gpu-topology',action='store_true');a.add_argument('--output',type=Path,required=True);args=a.parse_args()
base=p.parent/'candidate-v4-source';experimental=p/'candidate-r8-source';out=args.output;out.mkdir(exist_ok=False)
shutil.copytree(base/'endoscopic_tissue',out/'endoscopic_tissue',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(out/'genesis').symlink_to(p.parent/'baseline-source/genesis',target_is_directory=True)
def read(root,name):return (root/'endoscopic_tissue'/name).read_text()
def write(name,text):(out/'endoscopic_tissue'/name).write_text(text)
def replace(text,old,new,count=1):
 assert text.count(old)==count,(old,text.count(old),count)
 return text.replace(old,new)
for name in ['training/distance_kernels.py','training/instrument_collision.py','training/grasp.py','training/mesh_kernels.py']:
 shutil.copy2(experimental/'endoscopic_tissue'/name,out/'endoscopic_tissue'/name)
r=read(experimental,'rendering.py')
r=replace(r,', pbo_readback=False, render_nogil=False, persistent_soup=False, static_visuals=False',', persistent_soup=False')
r=replace(r,'        self.static_visuals=bool(static_visuals)\n','')
r=replace(r,',pbo_readback=pbo_readback','')
begin=r.index('        self._render_scheduling=None\n');end=r.index('\n    def add_mesh',begin);r=r[:begin]+r[end:]
begin=r.index('    def _force_visual_update(self):\n');end=r.index('    def render(',begin);r=r[:begin]+r[end:]
r=replace(r,'force_render=self._force_visual_update()','force_render=True',count=2)
if not args.rgb:
 r=replace(r,'    def render(self, depth=False, *, output_rgb=False):','    def render(self, depth=False):')
 r=replace(r,'            self._gpu_post.output_rgb=bool(output_rgb)\n','')
 r=replace(r,'        frame=cv2.multiply(frame,self.vignette,dtype=cv2.CV_8U)\n        return (cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) if output_rgb else frame),depth_m','        return cv2.multiply(frame,self.vignette,dtype=cv2.CV_8U),depth_m')
write('rendering.py',r)
v=read(base,'training/vision.py')
v=replace(v,',gpu_aligned_depth=config.get(\'gpu_aligned_depth\',False),**extra)',',gpu_aligned_depth=config.get(\'gpu_aligned_depth\',False),fused_mesh=config.get(\'fused_mesh\',False),persistent_soup=config.get(\'persistent_soup\',False),**extra)')
v=replace(v,'gpu_aligned_depth=self.gpu_aligned_depth)','gpu_aligned_depth=self.gpu_aligned_depth,fused_mesh=bool(config.get(\'fused_mesh\',False)),persistent_soup=bool(config.get(\'persistent_soup\',False))'+(',direct_rgb=self.direct_rgb)' if args.rgb else ')'))
if args.rgb:
 v=replace(v,"        self.gpu_aligned_depth=bool(config.get('gpu_aligned_depth',False))","        self.gpu_aligned_depth=bool(config.get('gpu_aligned_depth',False))\n        self.direct_rgb=bool(config.get('direct_rgb',False))")
 v=replace(v,'        frame_bgr, pinhole = self.view.render(depth=True)','        if self.direct_rgb:frame_bgr,pinhole=self.view.render(depth=True,output_rgb=True)\n        else:frame_bgr,pinhole=self.view.render(depth=True)')
 v=replace(v,'        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)','        if self.direct_rgb:\n            from .color_kernels import pack_rgb\n            rgb=pack_rgb(frame_bgr)\n        else:rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)')
 shutil.copy2(experimental/'endoscopic_tissue/training/color_kernels.py',out/'endoscopic_tissue/training/color_kernels.py')
 g=read(base,'training/gpu_postprocess.py')
 g=replace(g,'uniform bool emit_aligned;','uniform bool emit_aligned;\nuniform bool emit_rgb;')
 g=replace(g,'roundEven(shaded.bgr)','roundEven(emit_rgb?shaded.rgb:shaded.bgr)')
 g=replace(g,'        self.active = False','        self.active = False\n        self.output_rgb=False')
 g=replace(g,"self.aligned_location=gl.glGetUniformLocation(self.program,'emit_aligned')","self.aligned_location=gl.glGetUniformLocation(self.program,'emit_aligned');self.rgb_location=gl.glGetUniformLocation(self.program,'emit_rgb')")
 g=replace(g,'            gl.glUniform1i(self.aligned_location,int(emit));','            gl.glUniform1i(self.rgb_location,int(self.output_rgb))\n            gl.glUniform1i(self.aligned_location,int(emit));')
 write('training/gpu_postprocess.py',g)
write('training/vision.py',v)
if args.gpu_topology:
 e=read(base,'training/env.py')
 e=replace(e,'    fast_topology_checks: bool = False','    fast_topology_checks: bool = False\n    gpu_topology_checks: bool = False')
 e=replace(e,'        if self.max_steps < 1:',"        if type(self.gpu_topology_checks) is not bool or (self.gpu_topology_checks and not self.full_surface_checks):\n            raise ValueError('GPU topology checks require full surface checks and explicit bool')\n        if self.max_steps < 1:")
 e=replace(e,'        if self.config.fast_topology_checks:',"        if self.config.gpu_topology_checks:\n            from .gpu_topology import topology_statistics\n            self.quality=QualityMonitor(self.grid,self.config.limits,topology_function=lambda surface:topology_statistics(self.native,surface))\n        elif self.config.fast_topology_checks:")
 e=replace(e,'                                fast_topology_checks=self.config.fast_topology_checks,','                                fast_topology_checks=self.config.fast_topology_checks,\n                                gpu_topology_checks=self.config.gpu_topology_checks,')
 write('training/env.py',e);shutil.copy2(experimental/'endoscopic_tissue/training/gpu_topology.py',out/'endoscopic_tissue/training/gpu_topology.py')
# Extend provenance with only the helpers actually retained in the minimal source.
q=read(base,'training/snapshot.py');helpers=['distance_kernels.py','mesh_kernels.py']
if args.rgb:helpers.append('color_kernels.py')
if args.gpu_topology:
 helpers.append('gpu_topology.py')
 q=replace(q,"'fast_topology_checks': env.config.fast_topology_checks}","'fast_topology_checks': env.config.fast_topology_checks, 'gpu_topology_checks': getattr(env.config,'gpu_topology_checks',False)}")
q=replace(q,"'pixel_compose_fast.py')]}","'pixel_compose_fast.py', "+', '.join(repr(x) for x in helpers)+")]}")
write('training/snapshot.py',q)
# Compile syntax in memory; do not write outside the isolated output.
for f in out.rglob('*.py'):compile(f.read_text(),str(f),'exec')
manifest=dict(baseline=str(base),source=str(out),rgb=args.rgb,gpu_topology=args.gpu_topology,rejected_options_present=False,production_replaced=False,source_sha256={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*.py')})
(out/'ASSEMBLY.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},indent=2))
