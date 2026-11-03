from pathlib import Path
import hashlib,json
h=Path(__file__).resolve().parent
src=h.parent/'worker.py';s=src.read_text()
marker="            if self.native.tick % 30 == 0:\n"
assert s.count(marker)==1
insert='''            if result.after is not None:
                quality = result.after.metadata['quality']
                bad = not quality['checked_metrics_passed'] or not quality['within_declared_local_scope']
                episode = current_episode[0]
                if bad and getattr(self, '_captured_failure_episode', None) != episode:
                    self._captured_failure_episode = episode
                    target = out / episode / ('failure-tick-' + str(self.native.tick))
                    target.mkdir(parents=True, exist_ok=False)
                    surface = self.last_surface
                    if surface is None:
                        surface = self.native.surface()
                    np.savez_compressed(target/'state.npz', nodes_cm=self.observed.nodes,
                        neighbors=self.observed.neighbors, rest_cm=self.grid.positions,
                        inside=self.grid.inside, active=self.grid.active,
                        initial_neighbors=self.grid.neighbors, vertices_cm=surface.vertices,
                        normals=surface.normals, faces=surface.triangles, anchors=surface.anchors,
                        grip_flags=self.observed.grip_flags)
                    write_json(target/'metadata.json', result.after.metadata)
                    write_json(target/'diagnostic.json', {'quality_rules_unchanged': True,
                        'capture_after_original_assessment': True, 'native_tick': self.native.tick,
                        'note': 'Failure capture adds I/O after rejection; this run is diagnostic, not a controlled timing comparison.'})
'''
s=s.replace(marker,insert+marker)
(h/'topology_capture_worker.py').write_text(s)
(h/'topology-capture-provenance.json').write_text(json.dumps({'original_worker':str(src),'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'new_worker_sha256':hashlib.sha256(s.encode()).hexdigest(),'scope':'Opt-in separate diagnostic worker; original completed scale measurements are unchanged'},indent=2))
