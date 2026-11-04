"""Read-only source/ABI audit of the pinned native build and Genesis batching path."""
import hashlib,json,re
from pathlib import Path
from benchmark import ROOT,RELEASE,LIBRARY

def main():
    source=RELEASE/'endoscopic_tissue/outputs/a100-migration/contact-groups31/source'
    rows=[]
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.suffix not in ('.cu','.cuh','.cpp','.h'):continue
        text=p.read_text(errors='replace')
        rows.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'lines':text.count('\n'),'global_kernels':len(re.findall(r'__global__',text)),
            'static_line_declarations':len(re.findall(r'^static ',text,re.M)),
            'device_sync_calls':text.count('cudaDeviceSynchronize('),
            'sync_memcpy_calls':text.count('cudaMemcpy('),'async_memcpy_calls':text.count('cudaMemcpyAsync('),
            'graph_launch_calls':text.count('cudaGraphLaunch('),
            'tensor_core_api_hits':len(re.findall(r'\b(wmma|cublasGemmEx|mma\.sync)\b',text))})
    record={'library':str(LIBRARY),'library_sha256':hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
        'source_root_from_CMakeCache':str(source),'files':rows,
        'note':'Static counts are source occurrences, not per-step runtime counts; kernel definitions may live in included .cuh files.',
        'abi_has_context_or_environment_id':False,'native_session_process_global':True,
        'genesis_native_fem_has_batch_dimension':True,'custom_tissue_uses_genesis_fem':False,
        'full_articulated_dual_robot_arms_in_test':False}
    (ROOT/'source-audit.json').write_text(json.dumps(record,indent=2))
    print('audited',len(rows),'source files')

if __name__=='__main__':main()
