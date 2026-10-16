"""Training-only narrative augmentation boundaries; no inference dependency."""


def valid_narrative(value):
    return (isinstance(value, str) and 8 <= len(value.split()) <= 18
            and not any(marker in value for marker in ['<|', '|>', '\n', '\r', '`']))


def training_captions(rows, labels):
    known = {r['id']:r for r in rows}
    selected = []
    seen = set()
    for record in labels:
        key = record['id']
        if key in seen or key not in known or known[key]['split'] != 'train':
            raise ValueError('Unique existing training-image IDs required')
        seen.add(key)
        if record.get('training_only') is not True:
            raise ValueError('Training-only annotation provenance required')
        if record['label']['keep']:
            text = record['label']['narration_it'].strip()
            if not valid_narrative(text):
                raise ValueError('Invalid short training narration')
            selected.append({'id':key, 'caption':text})
    return selected
