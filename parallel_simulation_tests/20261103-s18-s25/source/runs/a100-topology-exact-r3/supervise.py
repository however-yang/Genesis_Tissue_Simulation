import pathlib,json,subprocess,time,sys
p=pathlib.Path(__file__).resolve().parent;start=time.time();code=1
try:
 code=subprocess.run(json.loads((p/'command.json').read_text()),timeout=3600).returncode
finally:
 (p/'status.json').write_text(json.dumps({'exit_code':code,'wall_s':time.time()-start,'offline_replay_only':True,'gpu_simulation_started':False},indent=2)+'\n')
raise SystemExit(code)
