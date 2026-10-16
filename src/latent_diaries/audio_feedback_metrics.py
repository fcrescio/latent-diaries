"""Offline semantic feasibility statistics; never a training or artistic loss."""
import math

import numpy as np


def wilson_interval(successes, total):
    if not 0 <= successes <= total or total < 1:
        raise ValueError('Valid binomial counts required')
    z = 1.959963984540054
    rate = successes/total;denominator = 1+z*z/total
    center = (rate+z*z/(2*total))/denominator
    radius = z*math.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/denominator
    return [max(0.,center-radius),min(1.,center+radius)]


def cluster_mean_interval(values, groups, *, replicates=2000, seed=20261016):
    values = np.asarray(values,dtype=float)
    groups = np.asarray(groups)
    if len(values) != len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite aligned observations and source-image groups required')
    unique = sorted(set(groups.tolist()))
    sums = np.array([values[groups==g].sum() for g in unique])
    counts = np.array([(groups==g).sum() for g in unique])
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0,len(unique),size=(replicates,len(unique)))
    means = sums[chosen].sum(-1)/counts[chosen].sum(-1)
    low,high = np.quantile(means,[.025,.975])
    result = {'mean':float(values.mean()),'ci95':[float(low),float(high)],
            'source_clusters':len(unique),'observations':len(values),'bootstrap_replicates':replicates,
            'seed':seed,'scope':'Source-cluster bootstrap conditional on this fixed candidate catalog'}
    if np.isin(values,[0,1]).all():
        if len(unique) == len(values):
            result['wilson95'] = wilson_interval(float(values.sum()),len(values))
        positive_clusters = sum(bool(values[groups==g].any()) for g in unique)
        result['source_cluster_any_positive_wilson95'] = wilson_interval(positive_clusters,len(unique))
        result['uncertainty_note'] = 'Degenerate bootstrap bounds at0/1 do not imply zero risk on unseen images; cluster-any-positive is a separate event rate.'
    return result


def retrieval(matrix, groups):
    scores = np.asarray(matrix)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError('Finite square matched-diagonal similarity matrix required')
    own = np.diag(scores)
    # Conservative ties: a tied wrong candidate does not count as a strict top1 hit.
    other = scores.copy();np.fill_diagonal(other,-np.inf)
    rank = 1+(other >= own[:,None]).sum(-1)
    return {'top1':cluster_mean_interval(rank==1,groups),
            'top5':cluster_mean_interval(rank<=min(5,len(rank)),groups),
            'mrr':cluster_mean_interval(1/rank,groups),
            'matched_minus_best_other':cluster_mean_interval(own-other.max(-1),groups),
            'ranks':rank.tolist(),'chance_top1':1/len(rank)}


def duplicate_aware_hits(scores, reference_texts, query_indices):
    """Identical offline reference strings are positives; other tied meanings fail."""
    scores = np.asarray(scores)
    texts = np.asarray(reference_texts)
    indices = np.asarray(query_indices)
    if (scores.shape != (len(indices), len(texts)) or not np.isfinite(scores).all()
            or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or (indices < 0).any() or (indices >= len(texts)).any()):
        raise ValueError('Finite query/catalog scores and valid integer identities required')
    hits = []
    for row, index in zip(scores, indices, strict=True):
        positive = texts == texts[index]
        hits.append(bool(row[positive].max() > row[~positive].max()) if (~positive).any() else True)
    return hits


def minimal_pair(original_audio, changed_audio, original_text, changed_text, groups):
    if len({x.shape for x in [original_audio,changed_audio,original_text,changed_text]}) != 1:
        raise ValueError('Aligned minimal-pair feature matrices required')
    aa = (original_audio*original_text).sum(-1)
    ab = (original_audio*changed_text).sum(-1)
    ba = (changed_audio*original_text).sum(-1)
    bb = (changed_audio*changed_text).sum(-1)
    per_source = ((aa>ab).astype(float)+(bb>ba).astype(float))/2
    return {'balanced_audio_to_text_accuracy':cluster_mean_interval(per_source,groups),
            'joint_assignment_accuracy':cluster_mean_interval(aa+bb>ab+ba,groups),
            'original_margin':cluster_mean_interval(aa-ab,groups),
            'changed_margin':cluster_mean_interval(bb-ba,groups),
            'per_source_balanced_accuracy':per_source.tolist(),
            'similarities':np.stack([aa,ab,ba,bb],-1).tolist(), 'chance_balanced_accuracy':.5}
