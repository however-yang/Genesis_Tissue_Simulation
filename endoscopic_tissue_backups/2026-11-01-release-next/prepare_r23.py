"""R23: reuse the existing immutable mode-2 query proof; no new GPU readback."""
import hashlib
import json
from pathlib import Path
import shutil

P = Path(__file__).resolve().parent
PACKAGE = P/'package'
PARENT = 'release-r22-repair-empty-manifold'
VARIANT = 'release-r23-repair-prelude-proof'
manifest = json.loads((PACKAGE/(PARENT+'.json')).read_text())
target = PACKAGE/'overrides'/VARIANT
if target.exists():
    raise FileExistsError(target)
for item in manifest['files']:
    source = PACKAGE/'overrides'/PARENT/item['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    dest = target/item['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
path = target/'vendor/cuda/xpbd/xpbd_solver.cu'
text = path.read_text()


def once(old, new):
    global text
    assert text.count(old) == 1, old[:120]
    text = text.replace(old, new)


once('static bool g_repairEmptySpecialization=true;', '''static bool g_repairEmptySpecialization=true;
// Conservative until the first successful immutable mode-2 prelude read.
// The mutable d_fragmentQueryCount is later reused by mode 1, and is NOT proof.
static bool g_lastMaterialFragmentQueries=true;
static bool g_repairProofAudit=false;''')
start = text.index('        // The paired constraints survive mode-1 fragment queries.')
end = text.index('        if(g_repairFragmentMode!=', start)
text = text[:start]+'''        // launchPredictedFragmentContact clears the retained count, then
        // creates the immutable mode-2 list. Its existing host query proves
        // empty => no paired constraint can be produced in this substep.
        // Later mode-1 passes never add paired constraints. A topology rebuild
        // can invalidate old witnesses but cannot create a paired constraint.
        // Possible queries conservatively keep the complete original path,
        // even when none actually made a contact. No additional CPU/GPU sync.
        const bool repairFragments=g_lastMaterialFragmentQueries || !g_repairEmptySpecialization;
        if(g_repairProofAudit){
            int retained=0;
            if(d_fragmentConstraintCount){
                cudaError_t read=cudaMemcpy(&retained,d_fragmentConstraintCount,sizeof(int),cudaMemcpyDeviceToHost);
                if(read!=cudaSuccess)return -(int)read;
            }
            g_repairDispatchStats[3]++;
            if(retained<0 || retained>g_fragmentConstraintCapacity || (!repairFragments && retained!=0))return XPBD_ERR_BAD_ARG;
        }
        g_repairDispatchStats[g_lastMaterialFragmentQueries?1:0]++;
'''+text[end:]
once('    const bool anyFragmentQueries=query[2]>0;',
     '    const bool anyFragmentQueries=query[2]>0;\n    g_lastMaterialFragmentQueries=anyFragmentQueries;')
once('    cudaFree(d_fragmentContacted);d_fragmentContacted=nullptr;',
     '    cudaFree(d_fragmentContacted);d_fragmentContacted=nullptr;\n    g_lastMaterialFragmentQueries=true;g_repairFragmentMode=-1;')
text += '''
// Readback audit and explicit repair are opt-in diagnostics only. Neither is
// used by the ordinary application or independent frame timing.
extern "C" __declspec(dllexport) void LCS_SetRepairProofAudit(int enabled){g_repairProofAudit=enabled!=0;}
extern "C" __declspec(dllexport) int LCS_ProbeReconstructedRepair(){return xpbd_relax_reconstructed_contact();}
'''
path.write_text(text, encoding='utf8', newline='\n')
manifest.update(variant=VARIANT, parent_variant=PARENT,
    status='isolated experiment; existing prelude empty-query proof; pending validation; never default')
for item in manifest['files']:
    content = (target/item['path']).read_bytes()
    item.update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
(PACKAGE/(VARIANT+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
builder = (PACKAGE/'build_experiment.py').read_text()
assert builder.count("'"+PARENT+"']") == 1
assert builder.count("'"+PARENT+"')") == 1
builder = builder.replace("'"+PARENT+"']", "'"+PARENT+"', '"+VARIANT+"']")
builder = builder.replace("'"+PARENT+"')", "'"+PARENT+"', '"+VARIANT+"')")
(PACKAGE/'build_experiment.py').write_text(builder, encoding='utf8', newline='\n')
print(json.dumps(dict(variant=VARIANT, physical_kernels_changed=False,
                     mode1_query_count_not_used=True, default_changed=False)))
