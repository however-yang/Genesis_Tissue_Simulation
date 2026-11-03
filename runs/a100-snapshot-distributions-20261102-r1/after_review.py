import pathlib,json,time,subprocess,sys
b=pathlib.Path(__file__).resolve().parent;r=b.parents[1]
while True:
 f=r/'runs/a100-combined-release-review-20261102-r1/status.json'
 if f.exists():
  s=json.loads(f.read_text())
  if s['phase'] in ('complete','failed','blocked'):break
 time.sleep(5)
if s['phase']!='complete':
 (b/'status.json').write_text(json.dumps({'phase':'blocked','reason':'Prior review process failed; inspect rather than masking it'}));raise SystemExit(2)
f=r/'releases/0643e10991cf/endoscopic_tissue/outputs/acceptance-fixes-20261102/a100-combined-regrasp-r1/result.json'
assert json.loads(f.read_text())['passed'], 'Initial paired mechanism/quality screen failed'
raise SystemExit(subprocess.call([sys.executable,'-B',str(b/'run_batch.py')]))
