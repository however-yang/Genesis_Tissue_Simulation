"""R25: preserve true cut identity independently of reconstructed boundary groups."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

P = Path(__file__).resolve().parent/'package'
ROOT = P.parents[2]
PARENT = 'release-r24-material-boundary'
VARIANT = 'release-r25-boundary-contact-identity'
manifest = json.loads((P/(PARENT+'.json')).read_text())
destination = P/'overrides'/VARIANT
if destination.exists() and (sys.argv[1:] != ['--resume'] or (P/(VARIANT+'.json')).exists()):
    raise FileExistsError(destination)
for item in manifest['files']:
    source = P/'overrides'/PARENT/item['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    target = destination/item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def once(text, old, new):
    assert text.count(old) == 1, old[:160]
    return text.replace(old, new)


relative = 'vendor/cuda/contact/surface_contact.cuh'
path = destination/relative
text = path.read_text()
text = once(text, '    const float3* rest=nullptr,const float4* materialFP=nullptr){',
            '    const float3* rest=nullptr,const float4* materialFP=nullptr,const unsigned* boundaryMasks=nullptr){')
text = once(text, '    float3 center=mkf3(0,0,0);\n    for(int k=0;k<8;k++){\n        if(cut && conn && conn[9*v.mask+1+k]!=index%8)continue;',
'''    // The cache key remains the ACTUAL severed-edge mask. Distinct true cuts
    // can have the same surface grouping after air paths are excluded.
    // Only the interpolation group lookup uses the reconstruction mask.
    const unsigned boundary=boundaryMasks?boundaryMasks[v.voxel]&4095u:v.mask;
    float3 center=mkf3(0,0,0);
    for(int k=0;k<8;k++){
        if(cut && conn && conn[9*boundary+1+k]!=index%8)continue;''')
path.write_text(text, encoding='utf8', newline='\n')

relative = 'vendor/cuda/xpbd/xpbd_solver.cu'
path = destination/relative
text = path.read_text()
text = once(text, 'static const unsigned* g_surfaceMasks=nullptr;',
            'static const unsigned* g_surfaceMasks=nullptr;\nstatic const unsigned* g_surfaceBoundaryMasks=nullptr;')
text = once(text, '''extern "C" int xpbd_set_contact_surface(int count,const int* tags,const float3* expanded,
    const int* corners,const unsigned* masks,const int* conn) {''',
'''extern "C" int xpbd_set_contact_surface(int count,const int* tags,const float3* expanded,
    const int* corners,const unsigned* masks,const int* conn,const unsigned* boundaryMasks) {''')
text = once(text, '    g_surfaceMasks=masks;g_surfaceConn=conn;return 0;',
            '    g_surfaceMasks=masks;g_surfaceConn=conn;g_surfaceBoundaryMasks=boundaryMasks;return 0;')
assert text.count('d_surfaceVertices,g_surfaceRest,g_surfaceMaterialFP);') == 4
text = text.replace('d_surfaceVertices,g_surfaceRest,g_surfaceMaterialFP);',
                    'd_surfaceVertices,g_surfaceRest,g_surfaceMaterialFP,g_surfaceBoundaryMasks);')
text = once(text, '    g_surfaceRest=nullptr;g_surfaceMaterialFP=nullptr;',
            '    g_surfaceRest=nullptr;g_surfaceMaterialFP=nullptr;g_surfaceBoundaryMasks=nullptr;')
path.write_text(text, encoding='utf8', newline='\n')

relative = 'vendor/cuda/dc_recon.cu'
path = destination/relative
text = path.read_text()
text = once(text, 'g_cutActive?cut_surface_mask():nullptr,g_cutActive?(const int*)d_cut_conn4096:nullptr);',
            'g_cutActive?d_cut_voxelCutMask:nullptr,g_cutActive?(const int*)d_cut_conn4096:nullptr,\n        g_cutActive?cut_surface_mask():nullptr);')
path.write_text(text, encoding='utf8', newline='\n')

relative = 'vendor/cuda/xpbd/xpbd_solver.h'
path = destination/relative
assert not path.exists()
text = subprocess.check_output(['git', 'show', manifest['base_commit']+':'+manifest['base_source_prefix']+relative], cwd=ROOT).decode().replace('\r\n', '\n')
start = text.index('int xpbd_set_contact_surface(')
end = text.index(';', start)
old = text[start:end+1]
assert old.endswith('const int* connectivity);'), old
text = once(text, old, old[:-2]+', const unsigned* boundaryMasks=nullptr);')
path.write_text(text, encoding='utf8', newline='\n')
manifest['files'].append(dict(path=relative))
for item in manifest['files']:
    content = (destination/item['path']).read_bytes()
    item.update(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
manifest.update(variant=VARIANT, parent_variant=PARENT,
    status='isolated boundary grouping with actual-cut cache identity; pending full interaction and performance validation; never default')
(P/(VARIANT+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')
builder = (P/'build_experiment.py').read_text()
assert builder.count("'"+PARENT+"']") == 1
assert builder.count("'"+PARENT+"')") == 1
builder = builder.replace("'"+PARENT+"']", "'"+PARENT+"', '"+VARIANT+"']")
builder = builder.replace("'"+PARENT+"')", "'"+PARENT+"', '"+VARIANT+"')")
(P/'build_experiment.py').write_text(builder, encoding='utf8', newline='\n')
print(VARIANT)
