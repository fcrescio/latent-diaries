"""Training-only frozen-reader policy drift using processors captured from real generation."""
import argparse
import json
import os
from pathlib import Path

import torch
from image_probe import make_reader
from safetensors.torch import load_file, save_file
from transformers.generation.logits_process import LogitsProcessorList

from latent_diaries.inversion import condition, digest, load_backbone, trajectory_logits
from latent_diaries.policy_diagnostics import policy_comparison


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--teachers', type=Path, required=True)
    p.add_argument('--repair-root', type=Path, required=True)
    p.add_argument('--initial-reader', type=Path, required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and new diagnostic directory required')
    dataset = json.loads(args.dataset.read_text())
    by_id = {r['id']: r for r in dataset['rows']}
    if any(by_id[k]['split'] != 'train' for k in args.ids):
        raise ValueError('Training-only policy oracle')
    teachers = json.loads((args.teachers / 'manifest.json').read_text())
    if teachers['dataset_sha256'] != digest(args.dataset):
        raise ValueError('Teacher dataset differs')
    teacher_by_key = {str(Path(r['prefix_file']).parent): r for r in teachers['records']}
    folders = [args.initial_reader, args.repair_root / 'reader-0', args.repair_root / 'reader-1']
    configs = [json.loads((f / 'config.json').read_text()) for f in folders]
    for folder, info in zip(folders, configs, strict=True):
        if info.get('talker_lora_rank', 0) or digest(folder / 'reader.safetensors') != info['reader_sha256']:
            raise ValueError('Frozen reader with verified weights required')
        if info['lock_sha256'] != digest(Path('uv.lock')):
            raise ValueError('Runtime lock changed')
    wrapper, builder, speaker = load_backbone(configs[0])
    readers = [make_reader(info, load_file(str(f / 'reader.safetensors'))).to(speaker.device).eval()
               for f, info in zip(folders, configs, strict=True)]
    args.output_dir.mkdir(mode=0o700)
    results = []
    with torch.inference_mode():
        for image_id in args.ids:
            image = by_id[image_id]
            features = torch.tensor([image['features']], device=speaker.device)
            prefixes = [reader(features)[0] for reader in readers]
            teacher_record = teacher_by_key[image['cache_dir']]
            for key, pathkey in [('prefix_sha256', 'prefix_file'), ('codes_sha256', 'codes_file')]:
                if digest(Path(teacher_record[pathkey])) != teacher_record[key]:
                    raise ValueError('Historical teacher changed')
            oracle = load_file(teacher_record['prefix_file'])['soft_tokens'].to(speaker.device)
            gold = load_file(teacher_record['codes_file'])['audio_codes'].to(speaker.device)
            captured = {'codec0': [], 'residual': []}
            originals = []
            # Capture actual processor order and actual initial ID context; do not approximate it.
            class RecordingProcessors(LogitsProcessorList):
                def __init__(self, processors, record):
                    super().__init__(processors);self.record = record
                def __call__(self, input_ids, scores):
                    if 'initial_ids' not in self.record:
                        self.record['initial_ids'] = input_ids.detach().clone()
                    processed = super().__call__(input_ids, scores.clone())
                    self.record.setdefault('raw_scores', []).append(scores.detach().clone())
                    self.record.setdefault('processed_scores', []).append(processed.detach().clone())
                    return processed
            for name, module in [('codec0', wrapper.model.talker), ('residual', wrapper.model.talker.code_predictor)]:
                original = module._get_logits_processor;originals.append((module, original))
                def capture(*a, _original=original, _name=name, _captured=captured, **kw):
                    processors = _original(*a, **kw);record = {'processors': processors}
                    _captured[_name].append(record)
                    return RecordingProcessors(processors, record)
                module._get_logits_processor = capture
            try:
                current = builder.generate_codes(condition(prefixes[-1], speaker), max_frames=64, seed=42)[0][None]
            finally:
                for module, original in originals:
                    module._get_logits_processor = original
            current_greedy = builder.generate_codes(condition(prefixes[-1], speaker), max_frames=64,
                                                     seed=42, codec0_greedy=True)[0][None]
            save_file({'audio_codes': current.cpu().contiguous()}, str(args.output_dir / f'{image_id}-current.safetensors'))
            save_file({'audio_codes': current_greedy.cpu().contiguous()}, str(args.output_dir / f'{image_id}-current-greedy.safetensors'))
            histories = [('gold', gold), ('current_final_reader', current), ('current_final_reader_greedy', current_greedy)]
            cache_evidence = []
            for iteration in [0, 1]:
                manifest_path = args.repair_root / f'histories-{iteration}/manifest.json'
                cache = json.loads(manifest_path.read_text())
                record = next(r for r in cache['records'] if r['id'] == image_id)
                if digest(Path(record['file'])) != record['sha256']:
                    raise ValueError('Historical student history changed')
                histories.append((f'cached_iteration_{iteration}', load_file(record['file'])['audio_codes'].to(speaker.device)))
                cache_evidence.append({'manifest_sha256': digest(manifest_path), 'history_sha256': record['sha256'],
                                       'source_reader_sha256': cache['source_reader_sha256']})
            processing = {name: {'classes': [type(x).__name__ for x in records[0]['processors']],
                                 'initial_ids': records[0]['initial_ids'].tolist()}
                          for name, records in captured.items()}
            for history_name, codes in histories:
                teacher0, teacher_rest = trajectory_logits(wrapper, builder, oracle, speaker, codes)
                labels = torch.cat([codes[0, :, 0], codes.new_tensor([wrapper.model.config.talker_config.codec_eos_token_id])])
                previous_student = None
                for reader_index, prefix in enumerate(prefixes):
                    student0, student_rest = trajectory_logits(wrapper, builder, prefix, speaker, codes)
                    effective_teacher, effective_student = [], []
                    record = captured['codec0'][0]
                    ids = record['initial_ids']
                    for frame in range(len(labels)):
                        previous_ids = torch.cat([ids, codes[:, :frame, 0]], -1)
                        effective_teacher.append(record['processors'](previous_ids, teacher0[:, frame].clone())[0])
                        effective_student.append(record['processors'](previous_ids, student0[:, frame].clone())[0])
                    effective_teacher, effective_student = torch.stack(effective_teacher), torch.stack(effective_student)
                    residual_teacher, residual_student = [], []
                    record = captured['residual'][0]
                    for frame in range(codes.shape[1]):
                        for book in range(15):
                            previous_ids = torch.cat([record['initial_ids'], codes[:, frame, 1:book+1]], -1)
                            residual_teacher.append(record['processors'](previous_ids, teacher_rest[frame:frame+1, book].clone())[0])
                            residual_student.append(record['processors'](previous_ids, student_rest[frame:frame+1, book].clone())[0])
                    eos = wrapper.model.config.talker_config.codec_eos_token_id
                    actual_parity = None
                    if history_name == 'current_final_reader' and reader_index == 2:
                        actual0 = torch.cat(captured['codec0'][0]['raw_scores'][:codes.shape[1]])
                        actual_processed = torch.cat(captured['codec0'][0]['processed_scores'][:codes.shape[1]])
                        actual_rest = torch.stack([torch.cat(r['raw_scores']) for r in captured['residual'][:codes.shape[1]]])
                        finite = torch.isfinite(actual_processed) & torch.isfinite(effective_student[:codes.shape[1]])
                        actual_parity = {'codec0_raw_max_abs_error': float((actual0 - student0[0, :codes.shape[1]]).abs().max()),
                            'codec0_processed_finite_max_abs_error': float((actual_processed - effective_student[:codes.shape[1]])[finite].abs().max()),
                            'codec0_processed_support_equal': bool(torch.equal(torch.isfinite(actual_processed), torch.isfinite(effective_student[:codes.shape[1]]))),
                            'residual_raw_max_abs_error': float((actual_rest - student_rest[:codes.shape[1]]).abs().max())}
                    drift = policy_comparison(student0[0], previous_student, labels, eos) if previous_student is not None else None
                    previous_student = student0[0]
                    result = {'actual_generation_replay_parity': actual_parity, 'student_shift_from_previous_version_raw': drift, 'image_id': image_id, 'history': history_name, 'reader_index': reader_index,
                        'frames': codes.shape[1], 'same_previous_audio_codes_for_teacher_student': True,
                        'generation_decoding': 'sampled codec0' if history_name == 'current_final_reader' else
                            ('gold teacher sampled' if history_name == 'gold' else 'greedy codec0; sampled residuals'),
                        'policy_processing': 'Actual captured sampled-codec0/residual processors on every history',
                        'final_eos_row': 'Counterfactual next EOS action; not proof the trajectory naturally stopped',
                        'codec0_raw': policy_comparison(student0[0], teacher0[0], labels, eos),
                        'codec0_effective': policy_comparison(effective_student, effective_teacher, labels, eos),
                        'residual_raw': policy_comparison(student_rest.flatten(0, 1), teacher_rest.flatten(0, 1), codes[0, :, 1:].flatten()),
                        'residual_effective': policy_comparison(torch.stack(residual_student), torch.stack(residual_teacher), codes[0, :, 1:].flatten()),
                        'processors': processing, 'cache_evidence': cache_evidence}
                    results.append(result)
                    print(json.dumps({'image_id': image_id, 'history': history_name, 'reader': reader_index}), flush=True)
    output = {'scope': 'Offline training-only policies on identical histories; raw and actual processed distributions, not quality scores',
        'model_id': configs[0]['model_id'], 'model_revision': configs[0]['model_revision'],
        'source_revision': os.environ.get('LD_CODE_REVISION'), 'source_sha256': digest(Path(__file__)),
        'metric_source_sha256': digest(Path('src/latent_diaries/policy_diagnostics.py')),
        'dataset_sha256': digest(args.dataset), 'teacher_manifest_sha256': digest(args.teachers / 'manifest.json'),
        'reader_sha256s': [c['reader_sha256'] for c in configs], 'lock_sha256': digest(Path('uv.lock')),
        'seed': 42, 'new_history_cap': 64, 'codec0_greedy': False, 'residuals_sampled': True, 'results': results}
    (args.output_dir / 'policies.json').write_text(json.dumps(output) + '\n')


if __name__ == '__main__':
    main()
