"""CPU-only independent enumeration of every fine/coarse Galerkin contribution."""
import json
from pathlib import Path
import re

COUPLED = 0x6c3761b


def offset(x, y, z):
    return (x + 1) + 3 * (y + 1 + 3 * (z + 1))


def coarse_step(corner, delta):
    return -1 if corner + delta < 0 else 1 if corner + delta > 1 else 0


def main():
    source = (Path(__file__).resolve().parent /
              'native/vendor/cuda/solver/material/multigrid_gather.cuh').read_text()
    outputs = [i for i in range(27) if COUPLED >> i & 1]
    fine, coarse = [], []
    for corner in range(8):
        position = (corner & 1, corner >> 1 & 1, corner >> 2)
        for table, size in ((fine, 15), (coarse, 27)):
            groups = {o: 0 for o in outputs}
            for slot in range(size):
                if size == 15:
                    sign, bits = (-1 if slot < 7 else 1), abs(slot - 7)
                    delta = tuple(sign * (bits >> axis & 1) for axis in range(3))
                else:
                    if not COUPLED >> slot & 1:
                        continue
                    delta = (slot % 3 - 1, slot // 3 % 3 - 1, slot // 9 - 1)
                output = offset(*(coarse_step(p, d) for p, d in zip(position, delta)))
                assert output in groups, (corner, slot, output)
                groups[output] |= 1 << slot
            # Each surviving original input is present exactly once.
            assert sum(v.bit_count() for v in groups.values()) == 15
            table.extend(groups[o] for o in outputs)
    expected = {'g_mgGatherOffsets': outputs, 'g_mgGatherFineMasks': fine,
                'g_mgGatherCoarseMasks': coarse}
    for name, values in expected.items():
        match = re.search(r'\b' + name + r'\[\d+\]=\{([^}]+)\}', source)
        assert match is not None, name
        actual = [int(v.strip().rstrip('u'), 0) for v in match[1].split(',')]
        assert actual == values, name
    print(json.dumps(dict(passed=True, outputs=len(outputs), corners=8,
                          fine_inputs_checked=120, coarse_inputs_checked=120,
                          scope='Index/mask mapping only; not floating-point or physics certification')))


if __name__ == '__main__':
    main()
