"""Small native integration probe; run as a module from the repository root."""
import argparse
import numpy as np
from .grid import build_grid
from .native_backend import NativeTissue, Command

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--spacing',type=float,default=.6)
    args=parser.parse_args()
    grid=build_grid(args.spacing)
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
