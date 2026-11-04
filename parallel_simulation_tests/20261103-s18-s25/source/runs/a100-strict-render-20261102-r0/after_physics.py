import pathlib,json,time,subprocess,sys
b=pathlib.Path(__file__).resolve().parent
state=b.parent/'a100-kidney-snapshot-20261102-r0/status.json'
while True:
 try:s=json.loads(state.read_text())
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
 if s.get('phase') in ('complete','failed'):break
 time.sleep(5)
raise SystemExit(subprocess.call([sys.executable,'-B',str(b/'run_batch.py')]))
