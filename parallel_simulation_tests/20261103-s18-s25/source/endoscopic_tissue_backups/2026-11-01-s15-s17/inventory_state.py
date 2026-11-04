"""Enumerate state declarations/allocation sites for explicit review, not a proof."""
from pathlib import Path
import json
import re

HERE=Path(__file__).parent
SOURCE=HERE/'artifacts/training-state1/source/vendor'


def clean(text):
    # Preserve offsets and line numbers while hiding comments and literals.
    pattern=r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern,lambda m: ''.join('\n' if c=='\n' else ' ' for c in m[0]),text)


records=[]
for file in sorted(SOURCE.rglob('*')):
    if file.suffix not in ('.cpp','.cu','.h','.cuh'):continue
    text=file.read_text(encoding='utf-8-sig'); stripped=clean(text)
    depth=[];level=0
    for ch in stripped:
        depth.append(level)
        level+=int(ch=='{')-int(ch=='}')
    declarations=[]
    for m in re.finditer(r'\b(?:static|__constant__|__device__)\b',stripped):
        end=stripped.find(';',m.start())
        if end<0:continue
        snippet=text[m.start():end+1]
        code=stripped[m.start():end+1]
        # Global functions and inline member functions are not storage objects.
        prefix=code.split('=',1)[0]
        if '(' in prefix or '__global__' in code or '__forceinline__' in code or '__device__ inline' in code:
            continue
        declarations.append({'line':text.count('\n',0,m.start())+1,'brace_depth':depth[m.start()],
                             'declaration':snippet.strip(), 'classification':'review_required'})
    allocations=[]
    for m in re.finditer(r'\bcudaMalloc\s*\(',stripped):
        start=m.end();paren=1;end=start
        while end<len(stripped) and paren:
            paren+=int(stripped[end]=='(')-int(stripped[end]==')');end+=1
        allocations.append({'line':text.count('\n',0,m.start())+1,'call':text[m.start():end],
                            'classification':'save_all_first_version_pending_pointer_layout_audit'})
    if declarations or allocations:
        records.append({'file':file.relative_to(SOURCE).as_posix(),'declarations':declarations,'allocations':allocations})
result={'schema':'native_state_inventory_review_v1','source':str(SOURCE),
        'allocation_sites':sum(len(r['allocations']) for r in records),
        'storage_candidates':sum(len(r['declarations']) for r in records),'review_complete':False,'files':records}
(HERE/'state-inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
for row in records:
    locals=[d for d in row['declarations'] if d['brace_depth']>0]
    if locals:print(row['file'],'NON_GLOBAL:',json.dumps(locals,ensure_ascii=False))
