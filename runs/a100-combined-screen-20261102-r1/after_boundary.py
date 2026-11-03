import pathlib,json,time,subprocess,sys
b=pathlib.Path(__file__).resolve().parent;r=b.parents[1]
while True:
 s=json.loads((r/'runs/a100-boundary-screen-20261102-r1/status.json').read_text())
 if s['phase'] in ('complete','failed'):break
 time.sleep(5)
build=json.loads((r/'runs/a100-functional-fixes-20261102-r1/combined-build.json').read_text());assert len(build)==2 and all(x['exit_code']==0 for x in build)
raise SystemExit(subprocess.call([sys.executable,'-B',str(b/'run_batch.py')]))
