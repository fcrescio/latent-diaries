"""Evaluate frozen ASFL feasibility; separate broad semantics, detail, and quality."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from latent_diaries.audio_feedback_metrics import cluster_mean_interval, minimal_pair, retrieval
from latent_diaries.feature_provenance import checksum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, required=True)
    p.add_argument('--features-dir', type=Path, required=True)
    p.add_argument('--human-export', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed CPU worker and fresh report directory required')
    benchmark = json.loads(args.benchmark.read_text());rows = benchmark['records']
    manifest_path = args.features_dir/'manifest.json';manifest = json.loads(manifest_path.read_text())
    for name,file in [('benchmark',args.benchmark),('targets',args.targets),('human_export',args.human_export),('protocol',args.protocol)]:
        if manifest[name+'_sha256'] != checksum(file):
            raise ValueError('Cache provenance differs: '+name)
    if not manifest.get('all_feature_groups_complete') or manifest['mixed_duration_batch_max_abs_error'] > 1e-4:
        raise ValueError('Incomplete/parity-failing media cache')
    features = {}
    for name,record in manifest['groups'].items():
        if checksum(record['file']) != record['sha256']:
            raise ValueError('Cached feature tensor changed')
        value = load_file(record['file'])['embeddings'].astype(np.float64)
        if value.shape[1] != 768 or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value,axis=-1),1,atol=1e-5):
            raise ValueError('Finite normalized768D embeddings required')
        features[name] = value
    groups = [r['scene_group'] for r in rows]
    visual,audio = features['image'],features['audio_clean']
    matrices = {'image_audio':visual@audio.T}
    results = {'scope':'Offline diagnostic statistics; no human artistic/intelligibility scores, no neural training',
        'benchmark_sha256':checksum(args.benchmark),'features_manifest_sha256':checksum(manifest_path),
        'protocol_sha256':checksum(args.protocol),'source_sha256':checksum(__file__),
        'metrics_source_sha256':checksum(inspect.getfile(cluster_mean_interval)),
        'training_cases':len(rows),'independent_capture_groups':len(set(groups)),
        'audio_to_text':{},'text_to_audio':{},'attribute_pairs':{},'negation_pairs':{},'perturbations':{},
        'image_to_audio':retrieval(matrices['image_audio'],groups),
        'audio_to_image':retrieval(matrices['image_audio'].T,groups)}
    for format_name in ['Document','SentenceSimilarity','unprefixed']:
        reference = features['text_'+format_name]
        original,attribute,negation = reference[:30],reference[30:60],reference[60:]
        matrix = audio@original.T;matrices['audio_text_'+format_name] = matrix
        results['audio_to_text'][format_name] = retrieval(matrix,groups)
        results['text_to_audio'][format_name] = retrieval(matrix.T,groups)
        results['attribute_pairs'][format_name] = minimal_pair(audio,features['audio_attribute'],original,attribute,groups)
        results['negation_pairs'][format_name] = minimal_pair(audio,features['audio_negation'],original,negation,groups)
    primary = features['text_Document'][:30]
    results['image_to_text_reference'] = retrieval(visual@primary.T,groups)
    similarity = matrices['image_audio']
    within = []
    for i,row in enumerate(rows):
        candidates = [j for j,r in enumerate(rows) if r['scene']==row['scene'] and j!=i]
        within.append(bool(similarity[i,i] > similarity[i,candidates].max()))
    results['within_historical_scene_image_audio'] = cluster_mean_interval(within,groups)
    vv = visual@visual.T;np.fill_diagonal(vv,-np.inf)
    neighbors = vv.argmax(-1)
    results['visual_neighbor_hard_pair'] = cluster_mean_interval(
        np.diag(similarity)>similarity[np.arange(30),neighbors],groups)
    results['visual_neighbor_margin'] = cluster_mean_interval(
        np.diag(similarity)-similarity[np.arange(30),neighbors],groups)
    source_audio_text = np.diag(audio@primary.T)
    for key,value in features.items():
        if not key.startswith('audio_') or value.shape[0] != 30:
            continue
        kind = key[6:]
        matrix = value@primary.T
        metrics = {'audio_to_text':retrieval(matrix,groups),
                   'own_text_cosine':cluster_mean_interval(np.diag(matrix),groups),
                   'delta_from_clean_own_text_cosine':cluster_mean_interval(np.diag(matrix)-source_audio_text,groups),
                   'clean_audio_cosine':cluster_mean_interval((value*audio).sum(-1),groups)}
        results['perturbations'][kind] = metrics
        matrices['audio_text_variant_'+kind] = matrix
    target_manifest = json.loads(args.targets.read_text())
    results['target_cap_hits'] = sum(r['reached_cap'] for r in target_manifest['records'])
    results['synthetic_speaker_conditions'] = {}
    for kind,info in target_manifest['synthetic_speaker_controls'].items():
        value = features['audio_'+kind]
        ids = manifest['groups']['audio_'+kind]['ids']
        indices = [manifest['row_ids'].index(k) for k in ids]
        scores = value@audio.T
        own = scores[np.arange(len(ids)),indices]
        wrong = scores.copy();wrong[np.arange(len(ids)),indices] = -np.inf
        selected_groups = [groups[i] for i in indices]
        results['synthetic_speaker_conditions'][kind] = {**info,
            'content_identity_over_other_original_voice':cluster_mean_interval(own>wrong.max(-1),selected_groups),
            'same_content_minus_best_other':cluster_mean_interval(own-wrong.max(-1),selected_groups),
            'text_retrieval_accuracy':cluster_mean_interval((value@primary.T).argmax(-1)==indices,selected_groups),
            'not_independent_speaker_validation':True}
    exported = json.loads(args.human_export.read_text())
    references = [r['mapping'] for r in exported if r['mapping']['level'] in {'L0','L1'}]
    refvectors = features['audio_exact_oracle_reference']
    refmap = {}
    for i,r in enumerate(references):
        refmap.setdefault((r['image_id'],r['seed']),i)
    by_clip = {r['mapping']['clip_id']:r['mapping'] for r in exported}
    degraded = benchmark['human_rated_degraded'];bad = features['audio_human_gibberish']
    quality = []
    for i,row in enumerate(degraded):
        identity = row['source_image_id'];case = manifest['row_ids'].index(identity)
        seed = by_clip[row['id']]['seed'];good = refvectors[refmap[identity,seed]]
        original_score = float(visual[case]@good);bad_score = float(visual[case]@bad[i])
        quality.append({'source_image_id':identity,'seed':seed,
            'reference_exact_image_audio_cosine':original_score,'rated_gibberish_image_audio_cosine':bad_score,
            'rated_gibberish_preferred':bad_score>=original_score,
            'rated_gibberish_text_cosine':float(primary[case]@bad[i]),
            'reference_exact_text_cosine':float(primary[case]@good),
            'joint_rated_gibberish_image_cosine':float(visual[case]@features['joint_human_gibberish'][i]),
            'joint_clean_image_cosine':float(visual[case]@features['joint_matched'][case])})
    results['human_rated_degraded_controls'] = {'pairs':quality,
        'degraded_preference':cluster_mean_interval([r['rated_gibberish_preferred'] for r in quality],
                                                  [r['source_image_id'] for r in quality]),
        'limitation':'Eight rated failures from only two source images; no fabricated positive quality scores'}
    joint = features['joint_matched']
    results['joint_shared_image_dominance'] = {}
    for kind in ['matched','same_scene_wrong_audio','other_scene_wrong_audio','same_audio_wrong_image']:
        values = features['joint_'+kind]
        results['joint_shared_image_dominance'][kind] = {
            'cosine_to_original_image':cluster_mean_interval((visual*values).sum(-1),groups),
            'cosine_to_original_audio':cluster_mean_interval((audio*values).sum(-1),groups),
            'cosine_to_matched_joint':cluster_mean_interval((joint*values).sum(-1),groups)}
    protocol = json.loads(args.protocol.read_text())
    broad = results['audio_to_text']['Document']['top1']
    hard = results['attribute_pairs']['Document']['balanced_audio_to_text_accuracy']
    gate = protocol['gate_audio_memory']
    memory_checks = {'broad_top1':broad['mean']>=gate['audio_to_text_top1_minimum'],
        'broad_lower_bound':broad['ci95'][0]>gate['bootstrap_lower_bound_minimum'],
        'minimal_pair_accuracy':hard['mean']>=gate['attribute_hard_pair_accuracy_minimum'],
        'minimal_pair_lower_bound':hard['ci95'][0]>gate['attribute_bootstrap_lower_bound_must_exceed'],
        'enough_independent_pairs':hard['source_clusters']>=gate['minimum_independent_attribute_pairs'],
        'no_target_cap_hits':results['target_cap_hits']==0}
    critic = protocol['gate_image_critic'];scene = results['within_historical_scene_image_audio']
    critic_checks = {'within_scene_top1':scene['mean']>=critic['same_scene_image_audio_top1_minimum'],
        'within_scene_lower_bound':scene['ci95'][0]>critic['bootstrap_lower_bound_must_exceed'],
        'no_rated_gibberish_preferred':not any(r['rated_gibberish_preferred'] for r in quality)}
    results['gates'] = {'audio_memory':{'passed_necessary_numeric_conditions':all(memory_checks.values()),'checks':memory_checks},
        'image_critic':{'passed_necessary_numeric_conditions':all(critic_checks.values()),'checks':critic_checks},
        'human_reranker_or_continuation_approval':False,'independent_speaker_robustness_verified':False,
        'neural_training_started':False,'reserved_fresh_access':False}
    args.output_dir.mkdir(mode=0o700)
    matrix_path = args.output_dir/'similarities.npz';np.savez_compressed(matrix_path,**matrices)
    results['similarities_sha256'] = checksum(matrix_path)
    (args.output_dir/'report.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'broad_audio_text':broad,'attribute_pair':hard,'image_audio_top1':results['image_to_audio']['top1'],
                      'gates':results['gates']},indent=2),flush=True)


if __name__ == '__main__':
    main()
