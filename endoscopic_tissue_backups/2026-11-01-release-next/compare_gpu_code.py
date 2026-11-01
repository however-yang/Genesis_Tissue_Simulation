"""Compare generated GPU instructions without retaining large disassembly files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

TOOL = Path(r'D:\develop\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\cuobjdump.exe')


def functions(dll):
    result = subprocess.run([str(TOOL), '--dump-sass', str(dll)], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = result.stdout.decode('utf8', errors='replace')
    blocks = re.split(r'\s*Function\s*:\s*([^\r\n]+)', text)
    answer = {}
    for i in range(1, len(blocks), 2):
        # Instruction addresses and scheduling encodings are part of the comparison.
        lines = [line.strip() for line in blocks[i+1].splitlines()
                 if re.search(r'/\*[0-9a-fA-F]+\*/', line)]
        answer[blocks[i].strip()] = hashlib.sha256('\n'.join(lines).encode()).hexdigest()
    if not answer:
        raise RuntimeError('No GPU functions found')
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    a, b = functions(args.parent), functions(args.candidate)
    common = sorted(a.keys() & b.keys())
    report = dict(parent=str(args.parent), candidate=str(args.candidate),
        parent_sha256=hashlib.sha256(args.parent.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        parent_functions=len(a), candidate_functions=len(b),
        changed=[n for n in common if a[n] != b[n]],
        removed=sorted(a.keys()-b.keys()), added=sorted(b.keys()-a.keys()),
        parent_instruction_hashes=a, candidate_instruction_hashes=b)
    report['all_gpu_functions_identical'] = a == b
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_hashes')}, indent=2))
