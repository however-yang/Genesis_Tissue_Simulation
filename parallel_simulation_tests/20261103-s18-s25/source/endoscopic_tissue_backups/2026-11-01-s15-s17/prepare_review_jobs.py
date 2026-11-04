"""Place full video rendering directly after each captured diagnostic run."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'endoscopic_tissue/outputs/s15-s17-20261101'


def main():
    path = HERE/'gpu-queue.json'
    queue = json.loads(path.read_text(encoding='utf8'))
    captures, extra = {}, []
    for scene in ('wall-traction', 'wall'):
        for label, dll in (('a', 'bin-contact-groups31'), ('b', 'bin-s15-training-snapshot1')):
            name = f'native-off-video-{scene}-{label}'
            extra.append({'name': name, 'args': [str(HERE/'validate_damage.py'), '--name', name,
                '--dll', dll, '--scene', scene, '--capture']})
            captures[name] = [OUT/'damage'/name]
    for repeat in range(1, 5):
        for ratio in (0, 1.3, 1.5, 2):
            name = f'nominal-tear-{ratio:g}-r{repeat}'
            args = [str(HERE/'validate_damage.py'), '--name', name, '--telemetry', '--tear', str(ratio), '--seed', str(repeat)]
            if repeat == 1:
                args.append('--capture')
                captures[name] = [OUT/'damage'/name]
            else:
                args = [str(HERE/'acceptance_job.py'), '--require',
                    str(OUT/'damage'/f'nominal-tear-{ratio:g}-r0'/'result.json'), '--', *args]
            extra.append({'name': name, 'args': args})
    existing = {j['name']: j for j in queue['jobs']}
    for job in extra:
        if job['name'] in existing and existing[job['name']] != job:
            raise ValueError('A queued review run already has a different configuration: '+job['name'])
    extra_names = {j['name'] for j in extra}
    queue['jobs'] = [j for j in queue['jobs'] if j['name'] not in extra_names]
    position = next(i for i, j in enumerate(queue['jobs']) if j['name'] == 'native-off-timing-a2')+1
    queue['jobs'][position:position] = [j for j in extra if j['name'].startswith('native-off-video-')]
    position = next(i for i, j in enumerate(queue['jobs']) if j['name'] == 'nominal-tear-2-r0')+1
    queue['jobs'][position:position] = [j for j in extra if j['name'].startswith('nominal-tear-')]
    for name in ('video-wall-a', 'video-wall-b', 'video-wall-postcut-a', 'video-wall-postcut-b'):
        captures[name] = [OUT/'default-comparison'/name]
    for geometry in ('wall-24mm', 'wall-25mm', 'wall-26mm-heldout'):
        name = 'geometry-'+geometry+'-r0'
        captures[name] = [OUT/'damage'/name]
    for fork in ('before_contact', 'first_cut', 'squeezing', 'regrasp'):
        name = 'snapshot-'+fork+'-r00'
        captures[name] = [OUT/'snapshot-distribution'/name/arm for arm in ('original', 'restored')]
    result, review_plan = [], []
    for job in queue['jobs']:
        # Existing render entries are regenerated into the same immediate slot.
        if job.get('s15_full_video_render'):
            continue
        result.append(job)
        for i, folder in enumerate(captures.get(job['name'], [])):
            name = f'render-{job["name"]}-{i}'
            result.append({'name': name, 's15_full_video_render': True,
                'args': [str(HERE/'render_validation.py'), str(folder)]})
            review_plan.append({'capture_job': job['name'], 'render_job': name, 'case': str(folder),
                                'video': str(folder/'review/interactions-h264.mp4')})
    queue['jobs'] = result
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(queue, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temp.replace(path)
    (HERE/'video-validation-plan.json').write_text(json.dumps({'schema': 's15_s17_video_plan_v1',
        'videos': review_plan, 'visual_review_passed': False}, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'jobs': len(result), 'complete_videos_scheduled': len(review_plan),
                      'no_gpu_job_started_here': True}))


if __name__ == '__main__':
    main()
