"""Small native integration probe; run as a module from the repository root."""
import argparse

import numpy as np

from .fixation import add_fixation_arguments, use_fixation
from .grid import build_grid
from .native_backend import Command, NativeTissue

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--spacing',type=float,default=.6)
    add_fixation_arguments(parser)
    args=parser.parse_args()
    grid=build_grid(args.spacing)
    use_fixation(parser, args, grid)
    print(f'Grid: {len(grid.positions)} nodes, {len(grid.intersections)} intersections',flush=True)
    with NativeTissue(grid) as tissue:
        obs=tissue.observe()
        surface=tissue.surface()
        print(f'Initialized: {len(surface.triangles)} triangles',flush=True)
        for i in range(args.steps):
            obs=tissue.step(Command())
            print(f'Step {i+1}: max displacement {np.max(np.linalg.norm(obs.nodes-grid.positions,axis=1)):.5f} cm',flush=True)
        print('NATIVE_SMOKE_OK',flush=True)

if __name__=='__main__':
    main()
