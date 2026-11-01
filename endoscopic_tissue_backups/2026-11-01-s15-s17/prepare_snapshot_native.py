"""Generate a separate snapshot build and a reviewed, explicit state catalog.

All live CUDA allocations are saved in this first version. Pointer-bearing
records use declared layouts; no guessing from integer bit patterns is allowed.
The module variable catalog is generated from the frozen inventory, with each
omission explicitly classified below. Compilation rejects nontrivial PODs.
"""
from pathlib import Path
import json
import re
import shutil

HERE = Path(__file__).parent
SOURCE = HERE/'artifacts/training-state1/source'
TARGET = HERE/'artifacts/training-snapshot1/source'
CATALOG = HERE/'snapshot-native'

MODULES = {
    'recon': ['cuda/dc_recon.cu', 'cuda/render_buffer.cuh', 'cuda/surface_cleanup.cuh'],
    'physics': ['cuda/physics.cu'],
    'cut': ['cuda/cut.cu', 'cuda/training/cut_training.cuh'],
    'haptics': ['cuda/haptics.cu'],
    'xpbd': ['cuda/xpbd/xpbd_solver.cu', 'cuda/contact/friction_contact.cuh',
             'cuda/cutting/pressure_blade.cuh', 'cuda/training/xpbd_training.cuh',
             'cuda/xpbd/tool_feedback.cuh', 'cuda/solver/material/coupled_material.cuh',
             'cuda/solver/material/material_line_search.cuh', 'cuda/solver/material/unified_vertex.cuh'],
}
PROFILES = {'g_profile', 'g_profileEvents', 'g_sectionProfile', 'g_sectionMarks', 'g_sectionPool', 'g_sectionTotals'}
DERIVED = {'g_unifiedGraphDirty', 'g_unifiedRecoveryGraphDirty', 'g_unifiedLoadedGraphDirty',
           'g_unifiedContactGraphDirty', 'g_fragmentPassKey'}
SYMBOLS = {'edgeCutTDevice', 'edgeSkinTDevice', 'edgeInsideDevice', 'materialRestDevice', 'c_jawMeshBounds'}


def clean(text):
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/', '', text)


def pieces(text):
    """Split a declaration at commas outside initializers/templates/arrays."""
    depth = 0
    start = 0
    for i, c in enumerate(text):
        if c in '({[<':
            depth += 1
        elif c in ')}]>':
            depth -= 1
        elif c == ',' and depth == 0:
            yield text[start:i]
            start = i+1
    yield text[start:]


def names(declaration):
    s = clean(declaration).rstrip(';').strip()
    return [re.search(r'(\w+)\s*(?:\[[^\]]*\]\s*)*$', p.split('=')[0].strip())[1] for p in pieces(s)]


def struct_pointer_fields(path, name):
    s = clean(path.read_text(encoding='utf-8-sig'))
    body = re.search(r'\bstruct\s+'+name+r'\s*\{([\s\S]*?)\};', s)[1]
    result = []
    for statement in body.split(';'):
        if '*' not in statement:
            continue
        if '(' in statement:
            raise ValueError('Review member functions in pointer-bearing layout: '+name)
        result.extend(names(statement))
    return result


def layout_code(name, fields):
    return 'lcs_state::Layout{sizeof('+name+'),{'+','.join('offsetof('+name+','+p+')' for p in fields)+'}}'


def generate():
    inventory = json.loads((HERE/'state-inventory.json').read_text(encoding='utf8'))
    by_file = {f['file']: f for f in inventory['files']}
    assert set(by_file) == {f for members in MODULES.values() for f in members}
    reviewed = []
    graphs = []
    for module, members in MODULES.items():
        lines = ['// Generated from the reviewed inventory. Do not hand edit.',
                 '#include "state_archive.h"', '#include <cstddef>']
        if module == 'xpbd':
            # These are the pointer-bearing device structs; all other
            # allocated POD layouts are audited in state-layout-audit.json.
            unified = struct_pointer_fields(SOURCE/'vendor/cuda/solver/material/unified_vertex.cuh', 'UnifiedState')
            mg = struct_pointer_fields(SOURCE/'vendor/cuda/solver/material/material_multigrid.cuh', 'MultigridLevel')
            tool = struct_pointer_fields(SOURCE/'vendor/cuda/grasp/tool_coupling.cuh', 'CoupledTool')
            lines += ['static lcs_state::Layout snapshotUnifiedLayout(){return '+layout_code('UnifiedState', unified)+';}',
                      'static lcs_state::Layout snapshotMultigridLayout(){return '+layout_code('MultigridLevel', mg)+';}',
                      'static lcs_state::Layout snapshotToolLayout(){return '+layout_code('CoupledTool', tool)+';}']
        lines += ['namespace lcs_state {', 'void collect_'+module+'(Fields& f){']
        if module == 'xpbd':
            lines += ['    if(g_profile || g_sectionProfile)throw std::runtime_error("Stop timing-event profiling before snapshot");',
                      '    allocation_layout(d_unified,snapshotUnifiedLayout());',
                      '    allocation_layout(d_bladeState,snapshotUnifiedLayout());',
                      '    allocation_layout(d_coupledTool,snapshotToolLayout());',
                      '    allocation_layout(d_bladeBody,snapshotToolLayout());']
        seen = set()
        for file in members:
            for declaration in by_file[file]['declarations']:
                d = declaration['declaration']
                if declaration['brace_depth'] or '__device__ void operator=' in d:
                    reviewed.append(dict(file=file, line=declaration['line'], declaration=d,
                                         category='immutable_or_function', reason='local constexpr tet table or operator method; pinned by DLL SHA'))
                    continue
                for name in names(d):
                    if name in seen:
                        raise ValueError('Duplicate module variable: '+name)
                    seen.add(name)
                    record = dict(file=file, line=declaration['line'], name=name, declaration=d)
                    if name in PROFILES:
                        record.update(category='recompute', reason='profiling events/statistics do not enter physical equations; snapshot rejects active profiling')
                    elif 'cudaGraphExec_t' in d:
                        record.update(category='recompute', reason='destroy on restore; next use recaptures graphs with restored pointers and parameters')
                        graphs.append((name, '[8]' in d or '[2]' in d))
                    elif 'cudaStream_t' in d:
                        record.update(category='recompute', reason='idle CUDA scheduling resource, no physical state; all work synchronized at boundary')
                    elif name in DERIVED:
                        record.update(category='recompute', reason='graph cache metadata; all associated graphs invalidated on restore')
                    elif name == 'STENCIL_OFF' or 'static constexpr' in d or re.search(r'static\s+const\s+(?:int|float)\s', d):
                        record.update(category='immutable', reason='compile-time constants pinned by DLL SHA')
                    else:
                        record.update(category='save', reason='conservative complete host/device-symbol state')
                        label = module+'.'+name
                        if name in SYMBOLS:
                            layout = 'Layout{sizeof(void*),{0}}' if name != 'c_jawMeshBounds' else 'Layout{}'
                            lines += ['    f.symbol("'+label+'",sizeof('+name+'),'+layout+',',
                                      '        [](void* out){check(cudaMemcpyFromSymbol(out,'+name+',sizeof('+name+')));},',
                                      '        [](const void* in,size_t bytes){check(cudaMemcpyToSymbol('+name+',in,bytes));});']
                        elif name == 'g_multigrid':
                            lines += ['    f.pod("'+label+'",'+name+',snapshotMultigridLayout());']
                        elif 'std::vector<' in d:
                            if 'std::vector<int>' not in d:
                                raise ValueError('Unreviewed vector codec: '+d)
                            lines += ['    f.vector("'+label+'",'+name+');']
                        else:
                            lines += ['    f.pod("'+label+'",'+name+');']
                    reviewed.append(record)
        lines += ['}', '}']
        if module == 'xpbd':
            lines += ['namespace lcs_state { void invalidate_xpbd_graphs(){']
            for name, array in graphs:
                if array:
                    lines += ['    for(auto& graph:'+name+'){if(graph)check(cudaGraphExecDestroy(graph));graph=nullptr;}']
                else:
                    lines += ['    if('+name+')check(cudaGraphExecDestroy('+name+'));'+name+'=nullptr;']
            for name in sorted(DERIVED-{'g_fragmentPassKey'}):
                lines += ['    '+name+'=false;']
            lines += ['    for(auto& key:g_fragmentPassKey)key={};', '}}']
        (CATALOG/(module+'_state.cuh')).write_text('\n'.join(lines)+'\n', encoding='utf8')
    (HERE/'state-reviewed-host.json').write_text(json.dumps({'schema': 'native_state_catalog_v1',
        'review_complete': False, 'note': 'Host classifications generated/reviewed; GPU layout audit and runtime T1-T8 pending',
        'fields': reviewed}, ensure_ascii=False, indent=2), encoding='utf8')


def prepare(refresh=False):
    if TARGET.exists() and not refresh:
        raise FileExistsError(TARGET)
    generate()
    assert TARGET.resolve().is_relative_to(HERE.resolve()/'artifacts')
    shutil.copytree(SOURCE, TARGET, dirs_exist_ok=True)
    dest = TARGET/'vendor/cuda/training'
    for file in CATALOG.glob('*'):
        shutil.copyfile(file, dest/file.name)
    for module, members in MODULES.items():
        file = TARGET/'vendor'/members[0]
        text = file.read_text(encoding='utf-8-sig')
        relative = '../training/' if module == 'xpbd' else 'training/'
        text = '#define LCS_STATE_TRACK_ALLOCATIONS\n#include "'+relative+'state_archive.h"\n'+text
        text += '\n#include "'+relative+module+'_state.cuh"\n'
        file.write_text(text, encoding='utf8')
    api = TARGET/'vendor/src/plugin_api.cpp'
    text = api.read_text(encoding='utf-8-sig')
    text = '#include "training/state_archive.h"\n'+text
    # Explicit frame boundary, rather than guessing from a substep index.
    old = 'LCS_API int LCS_Step(float dt)'
    index = text.index(old)
    brace = text.index('{', index)
    text = text[:brace+1]+'\n    lcs_state::boundary(false);'+text[brace+1:]
    index = text.index('LCS_API int LCS_Finalize(')
    brace = text.index('{', index)
    # Wrap the original complete body, preserving all existing error returns.
    end = brace+1
    depth = 1
    while depth:
        depth += (text[end] == '{')-(text[end] == '}')
        end += 1
    body = text[brace+1:end-1]
    text = text[:brace+1]+'\n    int result=[]()->int {'+body+'\n    }();\n    lcs_state::boundary(result==0);return result;\n'+text[end-1:]
    text += '''
LCS_API int LCS_StateAbiVersion(){return 1;}
LCS_API int LCS_StateSize(size_t* bytes){return lcs_state::state_size(bytes);}
LCS_API int LCS_SaveState(void* data,size_t bytes){return lcs_state::save_state(data,bytes);}
LCS_API int LCS_LoadState(const void* data,size_t bytes){return lcs_state::load_state(data,bytes);}
LCS_API const char* LCS_StateLastError(){return lcs_state::last_error();}
LCS_API int LCS_StateInventory(char* text,size_t capacity){return lcs_state::allocation_report(text,capacity);}
'''
    api.write_text(text, encoding='utf8')
    cmake = TARGET/'CMakeLists.txt'
    text = cmake.read_text(encoding='utf8')
    text = text.replace('    vendor/src/plugin_api.cpp', '    vendor/src/plugin_api.cpp\n    vendor/cuda/training/state_archive.cpp')
    cmake.write_text(text, encoding='utf8')
    print('Prepared independent snapshot source:', TARGET)


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['--refresh-catalog']:
        generate()
    elif sys.argv[1:] == ['--refresh']:
        prepare(refresh=True)
    elif sys.argv[1:]:
        raise SystemExit('Only --refresh-catalog is supported')
    else:
        prepare()
