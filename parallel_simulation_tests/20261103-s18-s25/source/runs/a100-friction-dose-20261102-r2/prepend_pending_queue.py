"""Replace only an owned, childless dependency waiter, preserving its queue.

No GPU worker is stopped. A fresh queue carries the exact old jobs after the
explicitly listed new diagnostics. Existing sources, outputs and logs remain.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    root=Path('/srv/disk2/hryang/genesis-exec')
    old=root/'runs/a100-friction-dose-20261102-r1'
    new=Path(__file__).resolve().parent
    assert new==root/'runs/a100-friction-dose-20261102-r2'
    previous=root/'runs/a100-functional-remaining-20261102-r3'
    pid=2481102
    proc=Path('/proc')/str(pid)
    expected=str(old/'after_queue.py').encode()
    def ready():
        assert proc.stat().st_uid==os.getuid()
        assert expected in (proc/'cmdline').read_bytes().split(b'\0')
        assert not (proc/'task'/str(pid)/'children').read_text().strip()
        assert json.loads((old/'dependency-status.json').read_text())['phase']=='waiting_for_predecessor'
        assert json.loads((previous/'status.json').read_text())['phase']=='running'
        assert not (old/'status.json').exists() and not (old/'supervisor.json').exists()
    ready()
    birth=(proc/'stat').read_text().split()[21]
    old_bytes=(old/'jobs.json').read_bytes()
    old_jobs=json.loads(old_bytes)
    assert len(old_jobs)==34
    assert not (new/'jobs.json').exists()
    py=root/'envs/genesis-py311-cu125-v1/bin/python'
    release=root/'releases/0643e10991cf'
    output=release/'endoscopic_tissue/outputs/acceptance-fixes-20261102'
    inserted=[
        dict(name='a100-s16d-public-policy-pilot-r2',args=[str(py),'-B','-u',str(new/'friction_pilot_suite.py')],max_wall=2400,max_idle=900),
        dict(name='a100-r3-frame-profile-r1',args=[str(py),'-B','-u',str(new/'profile_complete_frames.py'),
             '--release',str(release),'--library-dir',str(root/'builds/57f184c-release-training-r3-sm80/lib'),
             '--output',str(output/'a100-r3-frame-profile-r1')],max_wall=900,max_idle=600)]
    for job in inserted:
        assert not (output/job['name']).exists()
    for name in ['run_batch.py','after_queue.py']:
        data=(old/name).read_bytes()
        compile(data,str(new/name),'exec')
        with (new/name).open('xb') as f:f.write(data)
    with (new/'jobs.json').open('x') as f:json.dump(inserted+old_jobs,f,indent=2)
    record=dict(previous_queue=str(old),new_queue=str(new),waiting_for=str(previous),
        previous_jobs_sha256=hashlib.sha256(old_bytes).hexdigest(),retained_jobs=len(old_jobs),
        inserted_jobs=inserted,old_waiter_pid=pid,no_gpu_worker_stopped=True,created=time.time(),
        reason='S16d raw replay correctly rejected outside diagnostic mode; corrected to public policy step. '
               'Frame profiling dependency stopped on that script error and is explicitly requeued.')
    with (new/'queue-provenance.json').open('x') as f:json.dump(record,f,indent=2)
    ready()
    assert (proc/'stat').read_text().split()[21]==birth
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        if not proc.exists() or (proc/'stat').read_text().split()[2]=='Z':break
        time.sleep(.05)
    else:raise RuntimeError('Owned waiter did not exit; no replacement started')
    with (old/'superseded-before-launch.json').open('x') as f:json.dump(record,f,indent=2)
    with (new/'dependency.log').open('xb') as log:
        child=subprocess.Popen([str(py),'-B','-u',str(new/'after_queue.py'),'--previous',str(previous),'--queue',str(new)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    with (new/'dependency-pid.json').open('x') as f:json.dump(dict(pid=child.pid),f)
    print(json.dumps(dict(queue=str(new),pid=child.pid,retained_jobs=len(old_jobs),new_jobs=len(inserted))))


if __name__=='__main__':
    main()
