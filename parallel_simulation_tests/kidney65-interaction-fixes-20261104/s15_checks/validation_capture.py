"""Complete paired videos and annotations, without multi-GB raw replay files."""
from pathlib import Path
import json
import hashlib

import cv2
import numpy as np

from endoscopic_tissue.training import env as entry
from endoscopic_tissue.training.grasp import GraspMonitor
from endoscopic_tissue.training.data import atomic_json
from endoscopic_tissue.rendering import EndoscopicView
from endoscopic_tissue.scenario import SCISSORS_SCENES


class PairedCapture:
    def __init__(self, root, scene, name, *, arms=('original','restored'), cached_arm=None):
        self.root,self.scene,self.name=Path(root),scene,name
        self.label,self.arm='preparation',None
        self.cached_arm=cached_arm
        if cached_arm is not None:
            assert cached_arm in arms
            self.arm='original'
        self.expected_end_step=None
        self.expected_end_steps={}
        self.rows,self.journals,self.writers={},{},{}
        self.view=None
        cv2.setNumThreads(1)
        for arm in arms:
            folder=self.root/arm;folder.mkdir()
            self.rows[arm]=[]
            self.journals[arm]=(folder/'trajectory.jsonl').open('x',encoding='utf8',buffering=1)
        capture=self

        def recorder_for(base):
            class Recorder(base):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    capture.native=self
                    capture.monitor=GraspMonitor(self.settings,self.grid.inside)
                    scene_config=SCISSORS_SCENES[capture.scene]
                    capture.view=EndoscopicView(self.settings,1280,960,False,supersample=1,
                        kidney=scene_config.shape,camera=scene_config.shot,gpu_postprocess=True,
                        gpu_aligned_depth=True,fused_mesh=True,persistent_soup=True)
                    capture.view.update(self.surface(),self.observe());image,_=capture.view.render()
                    for arm in capture.rows:
                        writer=cv2.VideoWriter(str(capture.root/arm/'endoscopic_demo.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'),1/self.settings.dt,(1280,960))
                        assert writer.isOpened(),'Complete paired video could not open'
                        capture.writers[arm]=writer
                        if arm!=capture.cached_arm:
                            writer.write(image)

                def step(self,command):
                    observed=super().step(command)
                    capture.frame(self,observed)
                    return observed

            return Recorder

        self.original_class=entry.NativeTissue
        self.original_training_class=entry.TrainingNativeTissue
        entry.NativeTissue=recorder_for(self.original_class)
        entry.TrainingNativeTissue=recorder_for(self.original_training_class)

    def write_cached_prefix(self, source, tick):
        """Re-encode the actual cached preparation, without replaying its physics."""
        arm=self.cached_arm
        assert arm is not None and not self.rows[arm]
        source=Path(source)
        rows=[json.loads(line) for line in (source/'trajectory.jsonl').read_text().splitlines()][:tick]
        assert [row['step'] for row in rows]==list(range(1,tick+1))
        path=source/'endoscopic_demo.mp4'
        video=cv2.VideoCapture(str(path))
        try:
            assert video.isOpened()
            assert int(video.get(cv2.CAP_PROP_FRAME_COUNT))>=tick+1
            for frame in range(tick+1):
                ok,image=video.read()
                assert ok and image.shape==(960,1280,3)
                cv2.putText(image,'Cached preparation | same archived fork', (12,47),
                            cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
                self.writers[arm].write(image)
        finally:
            video.release()
        for row in rows:
            self.journals[arm].write(json.dumps(row)+'\n')
        self.rows[arm].extend(rows)
        return dict(video_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    trajectory_sha256=hashlib.sha256((source/'trajectory.jsonl').read_bytes()).hexdigest(),
                    native_prefix_frames=tick, initial_video_frames=1)

    def frame(self,native,observed):
        grasp=self.monitor.assess_native(observed)
        self.view.update(native.surface(),observed);image,_=self.view.render()
        label=self.label
        if label=='preparation':
            end=0
            for stage in SCISSORS_SCENES[self.scene].stages:
                end+=stage.steps
                if native.tick<=end:
                    label='preparation/'+stage.name;break
        row=dict(step=native.tick,stage=label,grasp=grasp,grip_points=observed.grip_points,
            jaw_load_n=list(observed.jaw_load_n),native_ms=getattr(native,'_s15_native_ms',None),
            cut_edges=int(np.count_nonzero((native.grid.neighbors[:,1::2]>=0)&(observed.neighbors[:,1::2]<0))))
        for arm in (self.rows if self.arm is None else (self.arm,)):
            frame=image.copy()
            cv2.putText(frame,f'{self.name} | {arm} | native tick {native.tick} | {label}',
                (12,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
            self.writers[arm].write(frame)
            self.journals[arm].write(json.dumps(row)+'\n');self.rows[arm].append(row)
            if native.tick%30==0:self.progress(arm)

    def progress(self,arm,closed=False):
        rows=self.rows[arm];last=rows[-1]['step'] if rows else 0
        expected=self.expected_end_steps.get(arm,self.expected_end_step)
        atomic_json(self.root/arm/'capture-progress.json',dict(closed=closed,last_recorded_step=last,
            expected_steps=expected,video_initial_frames=1,
            video_frames_written=len(rows)+1,physical_or_performance_approved=False))

    def close(self):
        entry.NativeTissue=self.original_class
        entry.TrainingNativeTissue=self.original_training_class
        for arm in self.rows:
            self.journals[arm].close()
            if arm in self.writers:self.writers[arm].release()
            rows=self.rows[arm];last=rows[-1]['step'] if rows else 0
            contiguous=[r['step'] for r in rows]==list(range(1,last+1))
            expected=self.expected_end_steps.get(arm,self.expected_end_step)
            video_frames=len(rows)+1 if arm in self.writers else 0
            atomic_json(self.root/arm/'report.json',dict(native_steps=last,video_frames=video_frames,
                contiguous_native_frames=contiguous,expected_steps=expected,
                completed_requested_continuation=video_frames>0 and expected is not None and last==expected,
                scope='Complete preparation, fork prefix and tested continuation; video dt is not measured throughput'))
            self.progress(arm,closed=True)
        if self.view is not None:self.view.close()
        assert all([r['step'] for r in rows]==list(range(1,(rows[-1]['step'] if rows else 0)+1))
                   for rows in self.rows.values()),'Paired capture lost/duplicated native frames'
