from collections import Counter

from latent_diaries.prior_corpus import candidates


def test_authored_corpus_has_combinatorial_holdouts_and_no_mandatory_ending():
    rows = list(candidates())
    assert len(rows) == len({r['text'] for r in rows}) == 2400
    pairs = {split: {(r['opening_index'], r['ending_index']) for r in rows if r['split'] == split}
             for split in ['train', 'heldout']}
    assert not pairs['train'] & pairs['heldout']
    assert {r['category'] for r in rows if r['split'] == 'train'} == {
        r['category'] for r in rows if r['split'] == 'heldout'}
    counts = Counter(r['ending_index'] for r in rows)
    assert len(counts) == 40 and max(counts.values()) / len(rows) < 0.05
