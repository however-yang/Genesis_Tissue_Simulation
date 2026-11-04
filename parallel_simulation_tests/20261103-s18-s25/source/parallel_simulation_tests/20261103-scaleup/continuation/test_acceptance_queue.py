"""Real subprocess checks for nested progress, interrupted-job resume, and rejection semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from acceptance_queue import HERE,Runner

class QueueTests(unittest.TestCase):
    def job(self,name,code,idle=1.):
        return {'id':name,'args':[sys.executable,'-B','-u','-c',code,'{attempt}'],
                'max_idle':idle,'max_wall':5,'progress_globs':['{attempt}/children/*'],
                'results':[{'path':'{attempt}/result.json','require':{'completed':True}}]}
    def test_silent_parent_child_file_progress_is_live(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-queue-') as temp:
            child="import pathlib,sys,time;p=pathlib.Path(sys.argv[1]);p.mkdir();[(time.sleep(.15),(p/str(i)).write_text(str(i))) for i in range(12)]"
            code="import subprocess,sys,pathlib,json;p=pathlib.Path(sys.argv[1]);subprocess.run([sys.executable,'-c',"+repr(child)+",str(p/'children')],check=True);(p/'result.json').write_text(json.dumps({'completed':True}))"
            runner=Runner(Path(temp)/'queue',{'jobs':[self.job('child-progress',code)]},poll=.02)
            self.assertEqual(runner.run(),0)
            self.assertGreater(runner.state['jobs']['child-progress']['attempts'][0]['wall_s'],1.5)
    def test_resume_skips_completed_and_retries_only_interrupted_boundary(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-queue-') as temp:
            marker=Path(temp)/'ready'
            okay="import sys,pathlib,json;(pathlib.Path(sys.argv[1])/'result.json').write_text(json.dumps({'completed':True}))"
            maybe="import time,pathlib;time.sleep(0 if pathlib.Path("+repr(str(marker))+").exists() else 2);"+okay
            plan={'jobs':[self.job('one',okay),self.job('two',maybe)]};out=Path(temp)/'queue'
            with self.assertRaises(TimeoutError):Runner(out,plan,poll=.02).run()
            first=json.loads((out/'state.json').read_text())['jobs']['one']['artifact_sha256']
            marker.write_text('retry');runner=Runner(out,plan,resume=True,poll=.02)
            self.assertEqual(runner.run(),0)
            self.assertEqual(len(runner.state['jobs']['one']['attempts']),1)
            self.assertEqual(len(runner.state['jobs']['two']['attempts']),2)
            self.assertEqual(first,runner.state['jobs']['one']['artifact_sha256'])
    def test_physical_failure_is_completed_evidence_not_pass(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-queue-') as temp:
            code="import pathlib,sys,json;(pathlib.Path(sys.argv[1])/'result.json').write_text(json.dumps({'completed':True,'passed':False}));sys.exit(2)"
            job=self.job('rejected',code);job['allowed_exit_codes']=[0,2];job['results'][0]['acceptance_field']='passed'
            runner=Runner(Path(temp)/'queue',{'jobs':[job]},poll=.02)
            self.assertEqual(runner.run(),2);self.assertFalse(runner.state['jobs']['rejected']['physical_passed'])
            resumed=Runner(Path(temp)/'queue',{'jobs':[job]},resume=True,poll=.02)
            self.assertEqual(resumed.run(),2)
    def test_relative_queue_with_different_child_cwd(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-queue-') as temp:
            code="import sys,pathlib,json;(pathlib.Path(sys.argv[1])/'result.json').write_text(json.dumps({'completed':True}))"
            job=self.job('relative',code);job['cwd']=temp
            directory=os.path.relpath(Path(temp)/'queue',Path.cwd())
            runner=Runner(directory,{'jobs':[job]},poll=.02)
            self.assertEqual(runner.run(),0)
            self.assertTrue(Path(runner.state['jobs']['relative']['attempts'][0]['directory']).is_absolute())

    def test_supervisor_own_state_is_not_progress(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='test-queue-') as temp:
            job=self.job('stalled','import time;time.sleep(3)');job['progress_globs']=[str(Path(temp)/'queue/state.json')]
            with self.assertRaises(TimeoutError):Runner(Path(temp)/'queue',{'jobs':[job]},poll=.02).run()
if __name__=='__main__':unittest.main()
