"""Independent media-only audio encoding; forbid other project-artifact reads."""
import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from latent_diaries.feature_provenance import checksum
from latent_diaries.multimodal_feedback_encoder import FrozenFeedbackEncoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not Path('/.dockerenv').exists() or args.output_dir.exists():
        raise ValueError('Managed worker and new isolation output required')
    rows = json.loads(args.inputs.read_text())['rows']
    if any(set(r) != {'id','audio','audio_sha256'} for r in rows):
        raise ValueError('Only opaque IDs and audio media provenance allowed')
    allowed = {args.inputs.resolve(),*[Path(r['audio']).resolve() for r in rows]}
    destination = args.output_dir.resolve()
    accesses = set()
    def audit(event, arguments):
        if event != 'open' or not arguments or not isinstance(arguments[0],(str,bytes)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if '/artifacts/' in str(path):
            if path not in allowed and not path.is_relative_to(destination):
                raise RuntimeError('Media-only process attempted a non-media project artifact read')
            accesses.add(str(path))
        if str(path).startswith('/mnt/d/backup/'):
            raise RuntimeError('Audio-only process must not read source images')
    sys.addaudithook(audit)
    for r in rows:
        if checksum(r['audio']) != r['audio_sha256']:
            raise ValueError('Waveform provenance differs')
    encoder = FrozenFeedbackEncoder('914f7f89142e33e77833254d9c9b90c3cef7303b','cuda')
    singles, traces = [], []
    for row in rows:
        value,trace = encoder.encode_audio([row['audio']]);singles.append(value);traces.append(trace)
    singles = torch.cat(singles)
    batches = torch.cat([encoder.encode_audio([r['audio'] for r in rows[i:i+4]])[0] for i in range(0,len(rows),4)])
    parity = float((singles-batches).abs().max())
    if parity > 1e-4:
        raise ValueError('Thirty-case single/batch audio parity failed')
    args.output_dir.mkdir(mode=0o700)
    file = args.output_dir/'features.safetensors'
    save_file({'single_audio_embeddings':singles,'batch_audio_embeddings':batches},str(file))
    output = {'scope':'Independent process with media-only manifest; Python project-artifact reads guarded',
        'source_revision':os.environ.get('LD_CODE_REVISION'),'source_sha256':checksum(__file__),
        'encoder_source_sha256':checksum(inspect.getfile(FrozenFeedbackEncoder)),
        'input_sha256':checksum(args.inputs),'features_sha256':checksum(file),
        'model_revision':encoder.revision,'dtype':'float32','vision_lock_sha256':checksum('/opt/vision/uv.lock'),
        'worker_lock_sha256':checksum('uv.lock'),'rows':[r['id'] for r in rows],
        'single_batch_max_abs_error':parity,'training_text_module_imported':'latent_diaries.training_text_features' in sys.modules,
        'variable_text_used':False,'project_artifact_accesses':sorted(accesses),'traces':traces}
    (args.output_dir/'manifest.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'audio_cases':len(rows),'single_batch_max_abs_error':parity,'variable_text_used':False}),flush=True)


if __name__ == '__main__':
    main()
