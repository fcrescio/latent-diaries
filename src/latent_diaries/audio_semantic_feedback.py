"""Bounded numeric-only candidate selection; no transcript, text model or ASR."""
import hashlib

import numpy as np


def select_audio_candidates(image, audio, eligible, *, source_id, seed=20261016, shuffled_image=None):
    image = np.asarray(image,dtype=float)
    audio = np.asarray(audio,dtype=float)
    eligible = np.asarray(eligible,dtype=bool)
    if image.shape != (768,) or audio.ndim != 2 or audio.shape[1] != 768 or len(audio) != len(eligible):
        raise ValueError('Aligned768D media features and eligibility required')
    if not 1 <= len(audio) <= 4:
        raise ValueError('A bounded pool of one to four candidates is required')
    if not np.isfinite(image).all() or not np.isfinite(audio).all():
        raise ValueError('Finite media features required')
    if not np.isclose(np.linalg.norm(image),1,atol=1e-5):
        raise ValueError('Normalized real image anchor required')
    if not np.allclose(np.linalg.norm(audio[eligible],axis=-1),1,atol=1e-5):
        raise ValueError('Normalized eligible audio features required')
    scores = audio@image
    random_index = int.from_bytes(hashlib.sha256(f'{seed}:{source_id}'.encode()).digest(),'big') % len(audio)
    def best(values):
        if not eligible.any():
            return None
        return int(np.where(eligible,values,-np.inf).argmax())
    selected = {'first_candidate':0 if eligible[0] else None,
                'fixed_random':random_index if eligible[random_index] else None,
                'image_audio_max':best(scores),'zero_image_scorer':best(np.zeros(len(audio)))}
    if shuffled_image is not None:
        shuffled_image = np.asarray(shuffled_image,dtype=float)
        if shuffled_image.shape != image.shape or not np.isfinite(shuffled_image).all() or not np.isclose(np.linalg.norm(shuffled_image),1,atol=1e-5):
            raise ValueError('Verified normalized shuffled image anchor required')
        selected['shuffled_image_scorer'] = best(audio@shuffled_image)
    return {'indices':selected,'image_audio_scores':scores.tolist(),'fixed_random_draw':random_index,
            'eligible_count':int(eligible.sum()),'quality_verified':False}
