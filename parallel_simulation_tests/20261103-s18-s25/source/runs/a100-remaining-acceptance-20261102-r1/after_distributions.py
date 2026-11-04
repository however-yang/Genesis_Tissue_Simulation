import pathlib,json,time,subprocess,sys
b=pathlib.Path(__file__).resolve().parent;r=b.parents[1]
while True:
 f=r/'runs/a100-snapshot-distributions-20261102-r1/status.json'
 if f.exists():
  s=json.loads(f.read_text())
  if s['phase'] in ('complete','failed','blocked'):break
 time.sleep(5)
if s['phase']!='complete':
 (b/'status.json').write_text(json.dumps({'phase':'blocked','reason':'Prior serial queue failed; inspect its logs before resuming'}));raise SystemExit(2)
raise SystemExit(subprocess.call([sys.executable,'-B',str(b/'run_batch.py')]))
