"""R22: exact empty retained-manifold specialization in reconstruction repair."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).resolve().parent
PACKAGE = P/'package'
OLD = ROOT/'endoscopic_tissue_backups/2026-10-31-interaction-fixes/package'
VARIANT = 'release-r22-repair-empty-manifold'
manifest = json.loads((OLD/'release-r19-volume-gate.json').read_text())
target = PACKAGE/'overrides'/VARIANT
if target.exists():
    if (PACKAGE/(VARIANT+'.json')).exists():
        raise RuntimeError('Preserve existing prepared source')
    for item in manifest['files']:
        existing = target/item['path']
        if not existing.exists() or hashlib.sha256(existing.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Partial preparation contains an edit; preserve it')
for item in manifest['files']:
    source = OLD/'overrides'/item.get('source_variant', 'release-r19-volume-gate')/item['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    dest = target/item['path']
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    item.pop('source_variant', None)
path = target/'vendor/cuda/xpbd/xpbd_solver.cu'
text = path.read_text()


def once(old, new):
    global text
    if text.count(old) != 1:
        raise RuntimeError('Expected one occurrence: '+old[:90])
    text = text.replace(old, new)


once('static bool g_unifiedContactGraphDirty=false;', '''static bool g_unifiedContactGraphDirty=false;
static int g_repairFragmentMode=-1;
static bool g_repairEmptySpecialization=true;
static int g_repairDispatchStats[4]={};''')
start = text.index('extern "C" int xpbd_relax_reconstructed_contact(){')
end = text.index('\n    // Continue the SAME final-substep constraints', start)
old = text[start:end]
block = old
needle = '        if(!g_unifiedContactGraph || g_unifiedContactGraphDirty){'
assert block.count(needle) == 1
block = block.replace(needle, '''        // The paired constraints survive mode-1 fragment queries. Test the
        // retained manifold count itself, never the last query-list count.
        int retained=0;
        if(d_fragmentConstraintCount){
            cudaError_t read=cudaMemcpy(&retained,d_fragmentConstraintCount,sizeof(int),cudaMemcpyDeviceToHost);
            if(read!=cudaSuccess)return -(int)read;
        }
        if(retained<0 || retained>g_fragmentConstraintCapacity)return XPBD_ERR_BAD_ARG;
        const bool repairFragments=retained>0 || !g_repairEmptySpecialization;
        g_repairDispatchStats[retained>0?1:0]++;
        if(g_repairFragmentMode!=(int)repairFragments)g_unifiedContactGraphDirty=true;
'''+needle)
lines = []
specialized = {'k_CoarseWeights', 'k_CoarseApply', 'k_UnifiedVertex'}
guarded = {'k_FragmentCoarseTerms', 'launchFragmentCoarseBounds', 'k_FragmentLineClear',
           'k_FragmentLineSum', 'k_FragmentLineSet'}
for line in block.splitlines():
    stripped = line.strip()
    name = stripped.split('<<<')[0].split('(')[0]
    if name in specialized:
        lines.append(' '*(len(line)-len(line.lstrip()))+'if(repairFragments)'+stripped.replace(name+'<<<', name+'<true><<<', 1)+
                     'else '+stripped.replace(name+'<<<', name+'<false><<<', 1))
    elif name in guarded:
        lines.append(' '*(len(line)-len(line.lstrip()))+'if(repairFragments)'+stripped)
    elif 'k_AssembleFragmentInteractions<<<' in line:
        lines.append(line.replace('k_AssembleFragmentInteractions<<<', 'if(repairFragments)k_AssembleFragmentInteractions<<<'))
    else:
        lines.append(line)
block = '\n'.join(lines)
assert block.count('if(repairFragments)') == 13, block.count('if(repairFragments)')
needle = 'cudaGraphDestroy(graph);if(capture!=cudaSuccess)return -(int)capture;g_unifiedContactGraphDirty=false;'
assert block.count(needle) == 1
block = block.replace(needle, needle+'g_repairFragmentMode=(int)repairFragments;g_repairDispatchStats[2]++;')
text = text[:start]+block+text[end:]
text += '''
// Diagnostic switch changes only empty-work dispatch. Nonempty retains the
// original kernels, manifold, quadrature, reactions and iteration budgets.
extern "C" __declspec(dllexport) void LCS_SetRepairEmptySpecialization(int enabled){g_repairEmptySpecialization=enabled!=0;}
extern "C" __declspec(dllexport) int LCS_ReadRepairDispatchStats(int* output,int capacity,int reset){
    if(!output || capacity<4)return -1;
    for(int i=0;i<4;i++){output[i]=g_repairDispatchStats[i];if(reset)g_repairDispatchStats[i]=0;}
    return 4;
}
'''
path.write_text(text, encoding='utf8', newline='\n')
manifest.update(variant=VARIANT, parent_variant='release-r19-volume-gate',
    status='isolated experiment; exact empty retained-contact dispatch in reconstruction repair; pending validation; never default')
for item in manifest['files']:
    content = (target/item['path']).read_bytes()
    item.update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
(PACKAGE/(VARIANT+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
builder = (PACKAGE/'build_experiment.py').read_text()
builder = builder.replace("'release-r21-topology-probe']", "'release-r21-topology-probe', '"+VARIANT+"']")
builder = builder.replace("'release-r21-topology-probe')", "'release-r21-topology-probe', '"+VARIANT+"')")
(PACKAGE/'build_experiment.py').write_text(builder, encoding='utf8', newline='\n')
(P/'r22-original-repair.txt').write_text(old, encoding='utf8')
print(json.dumps(dict(variant=VARIANT, guarded_sites=13, physical_iterations_changed=False, default_changed=False)))
