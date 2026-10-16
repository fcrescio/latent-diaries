import pytest

from latent_diaries.grounded_corpus import training_captions, valid_narrative


def test_narrative_data_cannot_inject_protocol_or_multiple_lines():
    text = 'Il pane dorato riposa sul tavolo accanto alla finestra.'
    assert valid_narrative(text)
    assert not valid_narrative(text+' <|im_start|>')
    assert not valid_narrative(text+'\n')
    assert not valid_narrative('troppo breve')


def test_narrative_augmentation_requires_training_only_unique_images():
    label = {'id':'one','training_only':True,'label':{'keep':True,
        'narration_it':'Il pane dorato riposa sul tavolo accanto alla finestra.'}}
    rows = [{'id':'one','split':'train'}]
    assert len(training_captions(rows,[label]))==1
    with pytest.raises(ValueError,match='Unique'):
        training_captions(rows,[label,label])
    with pytest.raises(ValueError,match='training-image'):
        training_captions([{'id':'one','split':'heldout'}],[label])
    with pytest.raises(ValueError,match='provenance'):
        training_captions(rows,[{**label,'training_only':False}])
