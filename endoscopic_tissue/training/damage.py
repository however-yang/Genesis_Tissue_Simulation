"""Stepwise damage observations; no inferred EMA or blade/tear relabelling."""
import numpy as np


class DamageMonitor:
    COUNTERS=('brokenCoreCorners','brokenGripPoints','tearEdges','materialTearEdges',
              'rimTearEdges','interiorTearEdges','nearGraspTearEdges')

    def __init__(self,grid):
        self.previous_neighbors=grid.neighbors
        target=grid.neighbors[:,1::2]
        material=np.asarray(grid.inside,bool)
        self.material_edges=(target>=0)&material[:,None]&material[np.maximum(target,0)]
        self.previous_counts={name:0 for name in self.COUNTERS}

    def assess(self,observed,metrics):
        if metrics is None:
            raise RuntimeError('Native damage metrics unavailable; cannot invent failure-law evidence')
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError('Nonfinite native damage metrics')
        delta={name:int(metrics[name])-self.previous_counts[name] for name in self.COUNTERS}
        if min(delta.values())<0:
            raise ValueError('Damage counters rolled backwards outside snapshot restore')
        total=material=0
        if observed.neighbors is not self.previous_neighbors:
            new=(self.previous_neighbors[:,1::2]>=0)&(observed.neighbors[:,1::2]<0)
            total=int(np.count_nonzero(new))
            material=int(np.count_nonzero(new&self.material_edges))
        if (delta['tearEdges']>total or delta['materialTearEdges']>material or
                delta['rimTearEdges']+delta['interiorTearEdges']!=delta['materialTearEdges']):
            raise ValueError('Native tear events disagree with actually severed material topology')
        result={'schema':'failure_telemetry_v1',
                'new_broken_edges':total,'new_material_broken_edges':material,
                'new_material_blade_edges':material-delta['materialTearEdges'],
                'new_tear_edges':delta['tearEdges'],'new_material_tear_edges':delta['materialTearEdges'],
                'new_tear_location':{'cut_rim':delta['rimTearEdges'],'interior':delta['interiorTearEdges'],
                                     'near_grasp_neighborhood':delta['nearGraspTearEdges']},
                'tear_location_semantics':'recorded before the tear; rim/interior partition material tears; near-grasp is the dilated support neighborhood before rim exemptions are removed and may overlap',
                'stretch':{'material_max':metrics['maxStretch'],'cut_rim_max':metrics['rimStretch'],
                           'grasp_neighborhood_max':metrics['graspRingStretch']},
                'anchors':{'new_broken_support_corners':delta['brokenCoreCorners'],
                           'new_broken_material_points':delta['brokenGripPoints'],
                           'remaining_material_points':int(observed.grip_points),
                           'remaining_support_corners':int(metrics['activeCoreCorners']),
                           'core_force_ema_max_n':metrics['coreForceMax']*.01,
                           'active_core_force_ema_median_n':metrics['coreForceMedian']*.01,
                           'max_population':'current grasp supports, including released overload witnesses',
                           'force_model':'unified finite material-point attachment forces distributed to supporting corners; tau=1s',
                           'release_rule':'overloaded support releases its material-point witnesses until a new grasp'},
                'jaw_normal_load_n':list(observed.jaw_load_n),
                'jaw_tangential_load_n':None,
                'unavailable':['separate_per_pad_tangential_contact_force'],
                'native_cumulative_counts':{name:int(metrics[name]) for name in self.COUNTERS}}
        self.previous_counts=result['native_cumulative_counts'].copy()
        self.previous_neighbors=observed.neighbors
        return result
