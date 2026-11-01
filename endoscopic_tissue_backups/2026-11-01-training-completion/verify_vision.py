"""Independent stored-pixel alignment and timestamp check; create review images."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
import cv2

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from endoscopic_tissue.training.data import EpisodeReader, load_arrays

P=Path(__file__).resolve().parent
episode=ROOT/'endoscopic_tissue/outputs/training-completion-20261101/final-v1/vision-pilot/rgbd-wall/episode'
reader=EpisodeReader(episode)
static=load_arrays(episode/'static.npz.zst')
xy=static['rgb_to_pinhole_xy']
height,width=xy.shape[:2]
# Nearest sampling occurs in the bottom-up GL buffer before inversion; preserve
# that tie convention instead of re-rounding upright half-integer coordinates.
ix=np.rint(xy[...,0]).astype(int)
iy=height-1-np.rint(height-1-xy[...,1]).astype(int)
inbounds=(ix>=0)&(ix<width)&(iy>=0)&(iy<height)
review=P/'vision-review'
review.mkdir(exist_ok=True)
rows=[]
for i in range(reader.manifest['states']):
    meta,arrays,topo=reader.state(i)
    vision=meta['vision']
    assert vision['step_id']==i and vision['native_tick']==meta['native_tick']
    assert vision['t_sim_s']==meta['t_sim_s'] and not vision['dropped']
    assert arrays['rgb'].shape==(height,width,3)
    assert arrays['depth_rgb_aligned_m'].shape==(height,width)
    expected=np.zeros((height,width),np.float32)
    expected[inbounds]=arrays['depth_pinhole_m'][iy[inbounds],ix[inbounds]]
    mask=arrays['depth_valid']
    assert np.array_equal(arrays['depth_rgb_aligned_m'][mask],expected[mask])
    assert np.all(arrays['depth_rgb_aligned_m'][~mask]==0)
    assert np.isfinite(arrays['camera_world_to_cv']).all()
    assert mask.any() and np.std(arrays['rgb'])>1
    rows.append(dict(step=i,t_sim_s=meta['t_sim_s'],valid_depth_pixels=int(mask.sum()),
                     valid_depth_range_m=[float(expected[mask].min()),float(expected[mask].max())],
                     aligned_pixels_exact=True))
    if i in (0,reader.manifest['states']-1):
        Image.fromarray(arrays['rgb']).save(review/f'rgb-{i:04}.png')
        depth=arrays['depth_rgb_aligned_m']
        normalized=np.clip((depth-.02)/.10*255,0,255).astype(np.uint8)
        color=cv2.applyColorMap(normalized,cv2.COLORMAP_TURBO)
        color[~mask]=0
        cv2.imwrite(str(review/f'depth-{i:04}.png'),color)
result={'frames':len(rows),'timestamps_and_all_valid_pixels_match':True,'rows':rows,
        'scope':'stored RGB/depth pairing and exact lens sampling; not a real-camera optical calibration'}
(P/'vision-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
