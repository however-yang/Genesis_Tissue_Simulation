"""CPU-only regression checks for evidence accounting and invalid launch requests."""
import json
from pathlib import Path
import tempfile
import unittest
from benchmark import ROOT,summarize

class Accounting(unittest.TestCase):
    def test_partial_or_crashed_work_cannot_be_counted_as_valid_episode(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='accounting-') as temp:
            out=Path(temp);wd=out/'worker-000';ep=wd/'episode-000';ep.mkdir(parents=True)
            (ep/'result.json').write_text(json.dumps({'passed':True,'full_script':False,'transitions':2,
                'quality_failures':0,'scope_failures':0,'snips':{}}))
            (ep/'steps.json').write_text('[]')
            (ep/'preparation.json').write_text('[{}]')
            (wd/'worker-result.json').write_text(json.dumps({'started':1,'finished':3,'completed':True}))
            result=summarize(out,[{'name':'worker-000','episodes':1,'exit_code':0}],[],1,3)
            self.assertTrue(result['complete'])
            self.assertEqual(result['aggregate_steps_s'],1.5)
            self.assertEqual(result['valid_episodes_per_hour'],0)
            self.assertEqual(result['passed_full_episodes'],0)
            result=summarize(out,[{'name':'worker-000','episodes':2,'exit_code':1}],[],1,3)
            self.assertFalse(result['complete'])

    def test_batch_makespan_is_used_instead_of_sum_of_worker_rates(self):
        with tempfile.TemporaryDirectory(dir=ROOT,prefix='accounting-') as temp:
            out=Path(temp);records=[]
            for i in range(2):
                name=f'worker-{i:03d}';wd=out/name;ep=wd/'episode-000';ep.mkdir(parents=True)
                (ep/'result.json').write_text(json.dumps({'passed':i==0,'full_script':True,'transitions':9,
                    'quality_failures':i,'scope_failures':0,'snips':{}}))
                (ep/'steps.json').write_text('[]');(ep/'preparation.json').write_text('[{}]')
                (wd/'worker-result.json').write_text(json.dumps({'started':1,'finished':2+i,'completed':True}))
                records.append({'name':name,'episodes':1,'exit_code':0})
            result=summarize(out,records,[],1,3)
            self.assertEqual(result['aggregate_steps_s'],10)
            self.assertEqual(result['passed_full_episodes'],1)
            self.assertEqual(result['valid_episodes_per_hour'],1800)
            self.assertNotEqual(result['aggregate_steps_s'],sum(r['steps_s'] for r in result['worker_rates']))

if __name__=='__main__':unittest.main()
