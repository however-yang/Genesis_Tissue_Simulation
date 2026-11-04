from pathlib import Path
import hashlib,json,shutil
here=Path(__file__).resolve().parent
project=here.parents[2]
source=project/'runs/a100-functional-fixes-20261102-r1/repair_convergence_probe.py'
shutil.copyfile(source,here/'repair_convergence_probe.py')
plan=json.loads((here/'recovery-plan.json').read_text())
plan['purpose']='R3 retained original/restored failure convergence diagnostics; unchanged objective, no time advancement'
plan['inputs_sha256']={str(here/'repair_convergence_probe.py'):hashlib.sha256((here/'repair_convergence_probe.py').read_bytes()).hexdigest()}
lib=project/'builds/57f184c-release-training-r3-sm80/lib/libLiverCudaSim.so'
plan['inputs_sha256'][str(lib)]=hashlib.sha256(lib.read_bytes()).hexdigest()
plan['jobs']=[]
for name in ['01-tick-975','02-tick-941']:
    snapshot=project/'releases/0643e10991cf/endoscopic_tissue/outputs/acceptance-fixes-20261102/a100-combined-T4-regrasp-r1/regrasp-r00/first-failures'/name/'state.lcssnap'
    plan['inputs_sha256'][str(snapshot)]=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    plan['jobs'].append(dict(id='repair-'+name,args=[str(project/'envs/genesis-py311-cu125-v1/bin/python'),'-B','-u',str(here/'repair_convergence_probe.py'),'--release',str(project/'releases/0643e10991cf'),'--library-dir',str(lib.parent),'--snapshot',str(snapshot),'--output','{attempt}/diagnostic'],cwd=str(here),max_idle=900,max_wall=1200,progress_globs=['{attempt}/diagnostic/result.json'],results=[dict(path='{attempt}/diagnostic/result.json',require={'completed':True,'native_import_exact':True})]))
(here/'repair-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
review=dict(frames=466,automated_full_decode=True,visual_review='Reviewed all three storyboards: preparation contact, jaw closure, tissue traction ramp and hold; no black/missing frames in inspected thumbnails. This is sparse visual inspection, not independent per-frame physical certification.',physical_acceptance_changed=False)
(here/'recovered-video-review.json').write_text(json.dumps(review,indent=2)+'\n')
print('Prepared two unchanged R3 failure-state diagnostics')
