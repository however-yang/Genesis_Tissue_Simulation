"""Check every accepted joint correction; diagnostic timing is not a speed result."""
import ctypes as ct
import json
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from endoscopic_tissue import native_backend,run
output=Path(sys.argv[1]).resolve();arguments=sys.argv[2:]
opened,checked=run.Flow.open,native_backend.NativeTissue.check
rows=[];ready=False
names=('roots','accepted','zero','fallback','disabled','negative_margin','volume_violation','closing_violation','energy_violation')
def read_stats(tissue):
    values=(ct.c_ulonglong*9)();margin=ct.c_float()
    assert tissue.dll.LCS_ReadJointDiagnostics(values,ct.byref(margin))==0
    return dict(zip(names,map(int,values)),minimum_negative_margin=float(margin.value))
def open_probe(self,tissue):
    global ready
    tissue.dll.LCS_ReadJointDiagnostics.argtypes=(ct.POINTER(ct.c_ulonglong),ct.POINTER(ct.c_float))
    tissue.dll.LCS_ReadJointDiagnostics.restype=ct.c_int
    result=opened(self,tissue);read_stats(tissue);ready=True;return result
def check_probe(self,name,*args):
    result=checked(self,name,*args)
    if ready and name=='LCS_Step':
        row=dict(step=self.tick+1,**read_stats(self));rows.append(row)
        if row['step']%100==0:print('[JOINT-DIAGNOSTIC] '+json.dumps(row),flush=True)
        assert row['roots']==row['accepted']+row['zero']+row['fallback'],row
        assert not any(row[k] for k in ('negative_margin','volume_violation','closing_violation','energy_violation')),row
    return result
run.Flow.open=open_probe;native_backend.NativeTissue.check=check_probe
sys.argv=['dev_run.py','--output',str(output),'--native-only','--scene','kidney',*arguments]
try:runpy.run_path(str(ROOT/'endoscopic_tissue_backups/2026-10-28-kidney/dev_run.py'),run_name='__main__')
finally:
    if output.is_dir():
        (output/'joint-diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        totals={k:sum(r[k] for r in rows) for k in names}
        totals.update(steps=len(rows),minimum_negative_margin=min((r['minimum_negative_margin'] for r in rows),default=0),note='Diagnostic DLL only; duration not a performance benchmark. This regression requires zero negative determinant margins. Legacy diagnostic violation counters are retained separately.')
        (output/'joint-diagnostics-summary.json').write_text(json.dumps(totals,indent=2),encoding='utf8')
