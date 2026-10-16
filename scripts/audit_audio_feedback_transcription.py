"""Auxiliary post-selection lexical check, never a scorer, filter or artistic metric."""
import argparse
import json
import re
from pathlib import Path
from statistics import mean

from latent_diaries.feature_provenance import checksum


def words(text):
    return re.findall(r'[^\W\d_]+', text.casefold(), flags=re.UNICODE)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(expected != actual)))
        previous = current
    return previous[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'targets', 'asr', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Fresh auxiliary report required')
    benchmark = json.loads(args.benchmark.read_text())
    targets = json.loads(args.targets.read_text())
    transcript = json.loads(args.asr.read_text())
    if targets['benchmark_sha256'] != checksum(args.benchmark) or transcript['audio_uploaded']:
        raise ValueError('Local sealed diagnostics required')
    by_audio = {str(Path(r['audio'])): r['auxiliary_transcription'] for r in transcript['results']}
    sources = {r['id']: r for r in benchmark['records']}
    cases = [(r, r['audio'], r['source_text'], 'original') for r in sources.values()]
    for target in targets['records']:
        source = sources[target['source_id']]
        kind = target['kind']
        text_key = kind+'_text' if kind in {'attribute', 'negation'} else 'source_text'
        cases.append((source, target['audio'], source[text_key], kind))
    records = []
    for source, audio, reference, kind in cases:
        actual = words(by_audio[str(Path(audio))]);expected = words(reference)
        record = {'id': source['id'], 'kind': kind, 'audio': audio,
                  'reference_words': len(expected), 'edit_distance': edit_distance(expected, actual)}
        if kind == 'attribute':
            record['changed_attribute_present'] = source['substitution']['replacement'].casefold() in actual
            record['old_attribute_present'] = source['substitution']['original'].casefold() in actual
        if kind == 'negation':
            record['negation_marker_recognized'] = 'falso' in actual
        records.append(record)
    aggregate = {}
    for kind in sorted({r['kind'] for r in records}):
        selected = [r for r in records if r['kind'] == kind]
        result = {'cases': len(selected), 'micro_word_edit_ratio': sum(r['edit_distance'] for r in selected)/
                  sum(r['reference_words'] for r in selected),
                  'exact_reference_match_fraction': mean(r['edit_distance'] == 0 for r in selected)}
        for key in ['changed_attribute_present', 'old_attribute_present', 'negation_marker_recognized']:
            if key in selected[0]:
                result[key+'_fraction'] = mean(r[key] for r in selected)
        aggregate[kind] = result
    output = {'scope': 'Auxiliary recognizer diagnostics only; recognizer errors are not human judgments',
              'inputs': {name: checksum(getattr(args, name)) for name in ['benchmark', 'targets', 'asr']},
              'source_sha256': checksum(__file__), 'asr_model_id': transcript['model_id'],
              'aggregate': aggregate, 'records': records, 'changed_selection_or_gates': False,
              'human_pronunciation_validated': False}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
