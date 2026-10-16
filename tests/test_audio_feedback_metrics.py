import pytest

np = pytest.importorskip('numpy')
from latent_diaries.audio_feedback_metrics import (
    cluster_mean_interval,
    duplicate_aware_hits,
    minimal_pair,
    retrieval,
    wilson_interval,
)


def test_cluster_bootstrap_does_not_treat_repeated_clips_as_independent_images():
    result = cluster_mean_interval([1,1,0,0],['a','a','b','b'])
    assert result['source_clusters'] == 2 and result['mean'] == .5
    assert result == cluster_mean_interval([1,1,0,0],['a','a','b','b'])
    assert result['ci95'] == [0.,1.]
    assert wilson_interval(0,2)[1] > .65
    assert wilson_interval(30,30)[0] < .89
    with pytest.raises(ValueError):
        cluster_mean_interval([float('nan')],['a'])


def test_retrieval_ties_and_minimal_spoken_meaning_are_separate():
    result = retrieval(np.eye(4),list('abcd'))
    assert result['top1']['mean'] == 1 and result['mrr']['mean'] == 1
    assert retrieval(np.ones((4,4)),list('abcd'))['top1']['mean'] == 0
    original = np.array([[1.,0.],[0.,1.]])
    changed = original[:,::-1]
    pair = minimal_pair(original,changed,original,changed,['a','b'])
    assert pair['balanced_audio_to_text_accuracy']['mean'] == 1
    collapsed = minimal_pair(original,original,original,changed,['a','b'])
    assert collapsed['balanced_audio_to_text_accuracy']['mean'] == .5


def test_duplicate_references_are_positives_but_other_meaning_ties_fail():
    scores = np.array([[.9, .9, .2], [.4, .4, .4], [.1, .1, .8]])
    assert duplicate_aware_hits(scores, ['same', 'same', 'different'], [0, 1, 2]) == [True, False, True]
    assert duplicate_aware_hits(np.ones((1, 2)), ['same', 'same'], [1]) == [True]
    with pytest.raises(ValueError):
        duplicate_aware_hits(scores, ['same', 'same', 'different'], [0, 1, 3])
