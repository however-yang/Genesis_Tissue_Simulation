import copy,unittest
from evaluate_performance import evaluate
class GateTests(unittest.TestCase):
 def valid(self):
  return dict(completed=True,quality_passed=True,scene='wall',gpu='GPU-a',render_device_binding={'gpu_uuid':'GPU-a','egl_index':1},
   render={'width':1280,'height':960,'depth':True,'render_every':1},frames=[{'native_tick':i} for i in range(20)],
   performance={'frames':20,'unique_frames_hz':25.,'p95_ms':45.,'rolling_20_frame_min_hz':22.,'mean_ms':40.})
 def test_valid_paired_measurement(self):self.assertTrue(evaluate(self.valid(),self.valid())['passed'])
 def test_headless_cannot_qualify(self):
  x=self.valid();x['render']['depth']=False;self.assertFalse(evaluate(x,x)['passed'])
 def test_repeated_display_frame_cannot_qualify(self):
  x=self.valid();x['frames'][1]['native_tick']=0;self.assertFalse(evaluate(x,x)['passed'])
 def test_wrong_graphics_gpu_cannot_qualify(self):
  x=self.valid();x['render_device_binding']['gpu_uuid']='GPU-other';self.assertFalse(evaluate(x,x)['passed'])
 def test_low_fps_and_rejected_physics_cannot_qualify(self):
  for change in ({'quality_passed':False},{'performance':{'unique_frames_hz':19.9}}):
   x=self.valid();x.update(change);self.assertFalse(evaluate(self.valid(),x)['passed'])
 def test_regression_cannot_qualify(self):
  x=self.valid();x['performance']['mean_ms']=41.;self.assertFalse(evaluate(self.valid(),x)['passed'])
if __name__=='__main__':unittest.main()
