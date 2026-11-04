import json,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(r.parent/'source'))
from endoscopic_tissue.training.acceptance_queue import Runner
raise SystemExit(Runner(r/'results/comparison-v1',json.loads((r/'comparison-v1-plan.json').read_text())).run())
