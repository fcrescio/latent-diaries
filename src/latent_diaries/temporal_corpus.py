"""Training-only sequential supervision contracts; never a runtime text planner."""
import hashlib
import re


def order_key(value, seed=20261016):
    return hashlib.sha256(f'{seed}:two-fragment:{value}'.encode()).hexdigest()


def select_pilot(rows, design):
    """Freeze12 original-training images/groups and8/4 split before drafting."""
    eligible = [r for r in rows if r['split'] == 'train']
    if any(r['split'] not in {'train', 'heldout'} for r in rows):
        raise ValueError('Original authorized train/development dataset required')
    selected, groups = [], set()
    scenes = sorted({design[r['id']]['scene'] for r in eligible})
    if len(scenes) != 10:
        raise ValueError('Expected ten historical scene categories')
    for scene in scenes:
        candidates = sorted([r for r in eligible if design[r['id']]['scene'] == scene], key=lambda r: order_key(r['id']))
        choice = next((r for r in candidates if r['scene_group'] not in groups), None)
        if choice is None:
            raise ValueError('Each scene needs an independent capture group')
        selected.append(choice);groups.add(choice['scene_group'])
    for row in sorted(eligible, key=lambda r: order_key(r['id'])):
        if len(selected) == 12:
            break
        if row['scene_group'] not in groups:
            selected.append(row);groups.add(row['scene_group'])
    if len(selected) != 12:
        raise ValueError('Twelve independent groups required')
    ordered = sorted(selected, key=lambda r: order_key('split:'+r['id']))
    result = []
    for i, row in enumerate(ordered):
        source = design[row['id']]
        if source['split'] != 'train':
            raise ValueError('Original training images only')
        result.append({**row, 'source_split': 'train', 'split': 'validation' if i < 4 else 'train',
                       'image': source['path'], 'scene': source['scene']})
    return result


def validate_story(value):
    if not isinstance(value, dict) or set(value) != {'keep', 'visible_details', 'branches'} or type(value['keep']) is not bool:
        raise ValueError('Strict story schema required')
    if not value['keep']:
        raise ValueError('Selected image rejected; preserve failure and amend protocol explicitly')
    details, branches = value['visible_details'], value['branches']
    if not isinstance(details, list) or not 2 <= len(details) <= 5 or not all(isinstance(t, str) and t.strip() for t in details):
        raise ValueError('Two to five concrete visible details required')
    if (not isinstance(branches, list) or any(not isinstance(b, dict) for b in branches)
            or [b.get('key') for b in branches] != ['A', 'B']):
        raise ValueError('Two ordered distinct branches A/B required')
    texts = []
    for branch in branches:
        if set(branch) != {'key', 'focus_it', 'fragments', 'progression_it', 'a2_requires_a1_it'}:
            raise ValueError('Strict branch schema required')
        if not all(isinstance(branch[k], str) and branch[k].strip() for k in ['focus_it', 'progression_it', 'a2_requires_a1_it']):
            raise ValueError('Explicit focus/progression/history rationale required')
        fragments = branch['fragments']
        if not isinstance(fragments, list) or len(fragments) != 2:
            raise ValueError('Exactly two fragments per branch required')
        for text in fragments:
            if not isinstance(text, str) or not 8 <= len(text.split()) <= 24 or '\n' in text:
                raise ValueError('Each single-line fragment must contain8–24 words')
            if any(c in text for c in ['<', '>', '`', '{', '}']):
                raise ValueError('No markup or instruction protocol in supervision')
            if re.search(r'\b(segreto|promessa|promesse)\b', text.casefold()):
                raise ValueError('Avoid historical stock-ending vocabulary')
            texts.append(text.strip())
    if len({text.casefold() for text in texts}) != 4:
        raise ValueError('Repeated fragments do not form branching narrative supervision')
    # This rejects mechanical copies only, not a claim of semantic progression.
    for first, second in [texts[:2], texts[2:]]:
        a, b = [set(re.findall(r'[^\W\d_]+', t.casefold())) for t in [first, second]]
        if len(a & b)/len(a | b) > .8:
            raise ValueError('Near-verbatim continuation requires editorial revision')
    return value
