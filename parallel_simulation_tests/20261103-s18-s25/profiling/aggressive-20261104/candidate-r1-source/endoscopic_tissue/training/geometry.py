"""Explicit wall-family geometry and one shared geometric preparation recipe.

The reference instrument clearances come from the accepted wall demonstration.
Instrument dimensions, mechanics, cutting criteria and step budgets are fixed.
`fixed_edge` names the supporting block's clamped boundary face; it does not
add a hidden grasp or pin the mobile wall. Ranges are candidates, not approval.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np

from ..grid import BLOCK_SIZE, Tab, BoxUnion, build_grid
from ..native_backend import Command
from ..scenario import (GRASP_ABOVE, SCISSORS_FACE_WIDTH, SCISSORS_SPACING, Stage,
    WALL_FACE_OFFSET, WALL_HOLD_BITE, WALL_HOLD_LIFT, WALL_HOLD_READY, WALL_HOLD_RAISE,
    WALL_ORIENTATION, WALL_PARK, WALL_BITES, WALL_TAB, WALL_TRACTION_SCENE)


@dataclass(frozen=True)
class WallGeometry:
    length_m: float = .025
    height_m: float = .007
    thickness_m: float = .004
    fixed_edge: str = 'y_min'
    cut_height_m: float = .0035
    support_length_m: float | None = None

    def __post_init__(self):
        if self.support_length_m is not None:
            if (type(self.support_length_m) not in (int,float) or not np.isfinite(self.support_length_m)
                    or not .002 <= self.support_length_m <= .008 or self.fixed_edge != "y_min"):
                raise ValueError("Unsupported-wall candidate requires 2..8 mm basal support and y_min clamp")
        for name, lo, hi in (('length_m',.022,.028),('height_m',.006,.009),
                             ('thickness_m',.003,.005),('cut_height_m',.003,.006)):
            value = getattr(self,name)
            if type(value) not in (int,float) or not np.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{name} must be finite and within [{lo}, {hi}] m')
        if self.cut_height_m > self.height_m-.002:
            raise ValueError('Cut plane must leave at least 2 mm of wall above it')
        if self.fixed_edge not in ('y_min','x_min','x_max','z_min','z_max'):
            raise ValueError('fixed_edge must name a supported boundary face of the block')

    @property
    def tab(self):
        top = BLOCK_SIZE[1]/2
        front = BLOCK_SIZE[2]/2
        return Tab((0.,top,front-self.thickness_m*100),
                   (self.length_m*100,top+self.height_m*100,front),1)

    def recipe(self):
        # A changed clamp boundary does not create held-out anatomy. Likewise
        # the procedural default must group with the identical legacy wall.
        if self.support_length_m is not None:
            return {'kind':'unsupported_wall_candidate_v1','tab':asdict(self.tab),
                    'support_length_m':self.support_length_m,'physical_acceptance':'pending'}
        return {'kind':'block_with_tab','tab':asdict(self.tab)}

    def build_grid(self,density,spacing_scale=1.):
        spacing=SCISSORS_SPACING*spacing_scale
        if spacing < .1:
            raise ValueError('Parametric wall spacing cannot be below 1 mm')
        if self.support_length_m is None:
            grid=build_grid(spacing,vertical_spacing=spacing,density=density,tab=self.tab)
        else:
            # Remove the block under the cutting side; retain the far support.
            # Same material, tool poses, resolution, native contact and cut law.
            half=np.asarray(BLOCK_SIZE)*.5
            block_high=half.copy();block_high[0]=self.support_length_m*100
            shape=BoxUnion(np.array([-half,self.tab.low]),np.array([block_high,self.tab.high]),half)
            grid=build_grid(spacing,vertical_spacing=spacing,density=density,shape=shape)
        if self.fixed_edge!='y_min':
            axis='xyz'.index(self.fixed_edge[0])
            bound=BLOCK_SIZE[axis]/2
            p=grid.positions[:,axis]
            bed=(p <= -bound) if self.fixed_edge.endswith('min') else (p >= bound)
            grid.pinned=(bed & (grid.active!=0)).astype(np.int32)
        return grid

    def scene(self):
        tab=self.tab
        low,high=np.asarray(tab.low),np.asarray(tab.high)
        center_z=(low[2]+high[2])/2
        height=high[1]-low[1]
        reference_height=WALL_TAB.high[1]-WALL_TAB.low[1]
        shift=np.array([high[0]-WALL_TAB.high[0],high[1]-WALL_TAB.high[1],center_z-1.8])
        # Translate the same pad engagement from the reference wall's free top
        # corner. Approach clearance is an instrument length, not tissue scale.
        bite=np.asarray(WALL_HOLD_BITE)+shift
        ready=bite+(np.asarray(WALL_HOLD_READY)-WALL_HOLD_BITE)
        lift=bite+(np.asarray(WALL_HOLD_LIFT)-WALL_HOLD_BITE)*(height/reference_height)
        plane=low[1]+self.cut_height_m*100
        hinge_y=plane+WALL_FACE_OFFSET-SCISSORS_FACE_WIDTH
        stride=(high[0]-low[0])/5
        bites=[(high[0]+.05-i*stride,hinge_y,high[2]+.6) for i in range(3)]
        clearance=np.asarray(WALL_PARK)-WALL_BITES[0]
        park=tuple(np.asarray(bites[0])+clearance)
        exit_=tuple(np.asarray(bites[-1])+clearance)
        opened=Command(tuple(ready),0,park,**WALL_ORIENTATION)
        held=replace(opened,hinge=tuple(lift),closure=1)
        stages=[Stage('settle',30,replace(opened,hinge=tuple(np.asarray(GRASP_ABOVE)+shift))),
                Stage('approach',165,opened),Stage('bite',45,replace(opened,hinge=tuple(bite))),
                Stage('close',45,replace(opened,hinge=tuple(bite),closure=1)),Stage('lift',60,held),
                Stage('blade_approach',120,replace(held,blade_center=bites[0]))]
        for i,point in enumerate(bites,1):
            if i>1:
                stages.append(Stage(f'reposition_{i}',90,replace(held,blade_center=point)))
            stages.append(Stage(f'incise_{i}',150,replace(held,blade_center=point,scissors_closure=1,
                                                       scissors_cutting=True),ramp=.5))
            if i<3:
                stages.append(Stage(f'open_{i}',30,replace(held,blade_center=point)))
            lift=lift+(0,WALL_HOLD_RAISE*height/reference_height,0)
            held=replace(held,hinge=tuple(lift))
            stages.append(Stage(f'raise_{i}',30,replace(held,blade_center=point,scissors_closure=float(i==3))))
        stages += [Stage('withdraw',120,replace(held,blade_center=exit_,scissors_closure=1)),
                   Stage('release',45,replace(held,blade_center=exit_)),
                   Stage('hold',30,replace(held,blade_center=exit_))]
        return replace(WALL_TRACTION_SCENE,tab=tab,vertical_spacing=.1,plane=plane,stages=tuple(stages),
            cut_low=(bites[-1][0]-.1,low[1]+.05,low[2]-.1),
            cut_high=(high[0]+.1,high[1]-.05,high[2]+.1))


def scene_for_config(config):
    from ..scenario import SCISSORS_SCENES
    return config.geometry.scene() if config.geometry is not None else SCISSORS_SCENES[config.scene]
