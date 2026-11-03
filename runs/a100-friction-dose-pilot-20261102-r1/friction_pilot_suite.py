"""Compare BASE/R3 pure-friction held/cut-ready probes; preserve all failures."""
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root=Path('/srv/disk2/hryang/genesis-exec')
    here=Path(__file__).resolve().parent
    assert here==root/'runs/a100-friction-dose-pilot-20261102-r1'
    release=root/'releases/0643e10991cf'
    output=release/'endoscopic_tissue/outputs/acceptance-fixes-20261102'
    python=root/'envs/genesis-py311-cu125-v1/bin/python'
    renderer=release/'endoscopic_tissue_backups/2026-11-02-acceptance-fixes/render_capture.py'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-c1c4c0d5-5b39-9b0e-84a0-a667e65484fa',
        EGL_DEVICE_ID='1',PYOPENGL_PLATFORM='egl',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    record=dict(started=time.time(),completed=False,candidate_adopted=False,stages=[],
        full_dose_matrix=False,scope='Two preparations, +z 4 mm single pilot; not the 900-trial S16d matrix')
    def run(label,args,allowed):
        stage=dict(name=label,command=[str(python),'-B','-u',*map(str,args)],started=time.time())
        record['stages'].append(stage)
        (here/'suite-progress.json').write_text(json.dumps(record,indent=2))
        with (here/(label+'.log')).open('xb') as log:
            result=subprocess.run(stage['command'],cwd=release,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,timeout=600)
        stage.update(exit_code=result.returncode,wall_s=time.time()-stage['started'])
        if result.returncode not in allowed:
            raise RuntimeError(f'{label}: unexpected exit {result.returncode}')
        return result.returncode
    variants=[('base','0643e10991cf-contact-groups31-sm80'),('r3','57f184c-release-training-r3-sm80')]
    try:
        for capture in [False,True]:
            # Alternate order across uncaptured/captured diagnostics.
            for label,build in (variants if not capture else list(reversed(variants))):
                name=f'a100-s16d-{label}-pilot'+('-video-r1' if capture else '-r1')
                assert not (output/name).exists()
                args=[here/'friction_dose.py','--release',release,'--library-dir',root/'builds'/build/'lib',
                      '--output',output/name,'--pilot']
                if capture:
                    args+=['--capture']
                run(name,args,(0,2))
                if capture:
                    for folder in sorted((output/name).iterdir()):
                        if folder.is_dir() and (folder/'replay-archive.json').exists():
                            run(name+'-'+folder.name,[renderer,folder],(0,))
        record['completed']=True
        record['physical_passed']=all(s['exit_code']==0 for s in record['stages'])
        return 0 if record['physical_passed'] else 2
    except BaseException as error:
        record['error']=repr(error)
        raise
    finally:
        record['wall_s']=time.time()-record['started']
        with (here/'suite-result.json').open('x') as stream:json.dump(record,stream,indent=2)
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    raise SystemExit(main())
