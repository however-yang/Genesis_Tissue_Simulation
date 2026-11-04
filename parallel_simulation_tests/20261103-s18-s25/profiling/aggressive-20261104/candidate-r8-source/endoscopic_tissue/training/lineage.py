"""Reset ancestry is independent of branch episode identity and policy seed."""


def episode_lineage(context):
    episode = context['episode_id']
    root = context.get('reset_lineage_id', episode)
    if any(type(x) is not str or not x or len(x) > 256 for x in (episode, root)):
        raise ValueError('Invalid episode/reset lineage ID')
    keys = ('parent_episode_id', 'branch_from_step', 'snapshot_sha256')
    has_branch = any(k in context for k in keys)
    parent = None
    if has_branch:
        if not all(k in context for k in keys) or 'reset_lineage_id' not in context:
            raise ValueError('Incomplete snapshot branch ancestry')
        parent, step, digest = (context[k] for k in keys)
        if (type(parent) is not str or not parent or parent == episode or root == episode
                or type(step) is not int or step < 0 or type(digest) is not str or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Invalid snapshot branch ancestry')
    elif root != episode:
        raise ValueError('A non-root lineage needs explicit snapshot branch ancestry')
    return {'reset_lineage_id': root, 'parent_episode_id': parent,
            'branch_from_step': context.get('branch_from_step'),
            'snapshot_sha256': context.get('snapshot_sha256')}


def validate_lineages(episodes):
    by_id = {e['episode_id']: e for e in episodes}
    geometry = {}
    for e in episodes:
        root = e['reset_lineage_id']
        if geometry.setdefault(root, e['geometry_group']) != e['geometry_group']:
            raise ValueError('A reset lineage cannot change physical geometry')
        parent = by_id.get(e['parent_episode_id'])
        if parent is not None and parent['reset_lineage_id'] != root:
            raise ValueError('Branch and recorded parent disagree on reset lineage')
        visited = {e['episode_id']}
        while parent is not None:
            if parent['episode_id'] in visited:
                raise ValueError('Cycle in snapshot branch ancestry')
            visited.add(parent['episode_id'])
            parent = by_id.get(parent['parent_episode_id'])
    train = {e['reset_lineage_id'] for e in episodes if e['purpose'] == 'collection'}
    evaluation = {e['reset_lineage_id'] for e in episodes if e['purpose'] == 'evaluation'}
    if train & evaluation:
        raise ValueError('Evaluation contains a collection episode or its snapshot branch')
