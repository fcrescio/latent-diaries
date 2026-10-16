"""Whitelisted aggregate evidence for the private teacher corpus; no human approval."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from audit_audio_feedback_transcription import edit_distance, words


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teachers-dir', default='teachers')
    args = p.parse_args();root = args.root
    files = {'selection': root/'selection/selection.json', 'corpus': root/'corpus-v2/corpus.json',
             'teachers': root/args.teachers_dir/'manifest.json', 'review': root/'listening/private-mapping.json',
             'media_cache': root/'media-cache/manifest.json', 'asr': root/'teacher-asr-small.json',
             'audit_original': root/'machine-audit/audit.json', 'audit_revision': root/'machine-audit-v2/audit.json'}
    data = {k: json.loads(v.read_text()) for k, v in files.items()}
    corpus, teacher = data['corpus'], data['teachers']
    by_sequence = {r['id']: r for r in corpus['rows']}
    reviewed = [by_sequence[r['sequence_id']] for r in data['review']['mapping']]
    if teacher['corpus_sha256'] != sha(files['corpus']) or len(teacher['records']) != 48:
        raise ValueError('Sealed complete teacher corpus required')
    texts = [t for r in corpus['rows'] for t in r['fragments']]
    counts = [len(t.split()) for t in texts]
    first = Counter(' '.join(t.casefold().split()[:3]) for t in texts)
    last = Counter(' '.join(t.casefold().split()[-3:]) for t in texts)
    transcripts = {r['audio']: r['auxiliary_transcription'] for r in data['asr']['results']}
    targets = {r['id']: r for r in teacher['records']}
    edits, total_words = 0, 0
    for row in corpus['rows']:
        for i, text in enumerate(row['fragments'], 1):
            audio = targets[row['id']+f'-f{i}']['audio']
            expected, actual = words(text), words(transcripts[audio])
            edits += edit_distance(expected, actual);total_words += len(expected)
    report = {'scope': 'Text-conditioned teacher upper bound and frozen media preparation, not trained textless narration',
        'source_sha256': sha(Path(__file__)), 'private_evidence_sha256': {k: sha(v) for k, v in files.items()},
        'images': 12, 'independent_capture_groups': len({r['scene_group'] for r in corpus['rows']}),
        'original_source_partition': 'Existing authorized training cohort only',
        'train_images': len({r['image_id'] for r in corpus['rows'] if r['split'] == 'train'}),
        'diagnostic_validation_images': len({r['image_id'] for r in corpus['rows'] if r['split'] == 'validation'}),
        'historical_scene_categories': len({r['scene'] for r in corpus['rows']}),
        'sequences': len(corpus['rows']), 'fragments': len(texts), 'distinct_fragment_strings': len(set(texts)),
        'word_count_range': [min(counts), max(counts)], 'mean_word_count': mean(counts),
        'maximum_repeated_three_word_opening': max(first.values()),
        'maximum_repeated_three_word_ending': max(last.values()),
        'lexical_counts_do_not_certify_narrative_diversity': True,
        'teacher': {k: teacher[k] for k in ['model_id','model_revision','worker_lock_sha256','reference_transcript_used',
            'codec0_greedy','residuals_sampled','batch_size','seed','max_frames','peak_vram_bytes']},
        'teacher_duration_seconds_range': [min(r['seconds'] for r in teacher['records']),
                                          max(r['seconds'] for r in teacher['records'])],
        'teacher_frame_caps': sum(r['reached_cap'] for r in teacher['records']),
        'blinded_review_sequences': len(data['review']['mapping']),
        'blinded_review_source_images': len({r['image_id'] for r in reviewed}),
        'blinded_review_historical_categories': len({r['scene'] for r in reviewed}),
        'blinded_review_train_sequences': sum(r['split']=='train' for r in reviewed),
        'blinded_review_validation_sequences': sum(r['split']=='validation' for r in reviewed),
        'review_pause_seconds': .45, 'review_run': 'asfl-teacher-corpus-20261016',
        'local_machine_audit_images': len(data['audit_original']['records']),
        'explicitly_revised_and_reaudited_images': len(data['audit_revision']['records']),
        'machine_audit_is_fallible_correlated_and_not_human_validation': True,
        'media_cache': {'images': len(data['media_cache']['image_ids']), 'audio': len(data['media_cache']['audio_ids']),
            'dimension': 768, 'model_revision': data['media_cache']['model_revision'],
            'variable_text_used': data['media_cache']['variable_text_used']},
        'auxiliary_asr': {'model_id': data['asr']['model_id'], 'revision': data['asr']['model_revision'],
            'utterances': len(transcripts), 'micro_word_edit_ratio': edits/total_words,
            'never_used_for_selection': True, 'not_a_human_intelligibility_or_artistic_score': True},
        'human_subset_validation_pending': True, 'neural_training_started': False,
        'm2_accepted': False, 'reserved_fresh_access': False, 'private_uploads': False,
        'public_report_contains_no_stories_notes_media_vectors_or_mapping': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Exported aggregate teacher-corpus evidence only; human corpus validation remains pending')


if __name__ == '__main__':
    main()
