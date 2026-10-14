from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from latent_diaries.conditioning import SoftConditioningBuilder, VisualCondition


class TinyTalker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = torch.nn.Embedding(64, 8)
        self.text = torch.nn.Embedding(64, 8)
        self.text_projection = torch.nn.Linear(8, 8)
        self.extra = torch.nn.ModuleList([torch.nn.Embedding(64, 8) for _ in range(15)])
        self.code_predictor = SimpleNamespace(get_input_embeddings=lambda: self.extra)

    def get_input_embeddings(self):
        return self.audio

    def get_text_embeddings(self):
        return self.text


@pytest.fixture
def setup():
    torch.manual_seed(42)
    talker = TinyTalker().requires_grad_(False)
    codec = SimpleNamespace(hidden_size=8, num_code_groups=16, codec_think_id=4,
        codec_think_bos_id=5, codec_think_eos_id=6, codec_language_id={"italian": 7},
        codec_pad_id=8, codec_bos_id=9, codec_eos_token_id=10)
    config = SimpleNamespace(talker_config=codec, tts_bos_token_id=11,
                             tts_eos_token_id=12, tts_pad_token_id=13)
    builder = SoftConditioningBuilder(SimpleNamespace(talker=talker, config=config), (1, 2, 3))
    soft = torch.randn(2, 3, 8, requires_grad=True)
    condition = VisualCondition(soft, torch.tensor([[True, True, True], [True, False, True]]),
                                torch.randn(2, 8))
    return builder, condition


def test_nonstreaming_stock_assembly_parity(setup):
    builder, condition = setup
    got = builder.build_prefill(condition)
    talker = builder.model.talker
    for i in range(2):
        def text(ids):
            return talker.text_projection(talker.text(torch.tensor([ids])))
        def audio(ids):
            return talker.audio(torch.tensor([ids]))
        bos, eos, pad = text([11, 12, 13]).chunk(3, dim=1)
        codec = torch.cat([audio([4, 5, 7, 6]), condition.speaker_embedding[i:i+1, None],
                           audio([8, 9])], dim=1)
        # Independent transcription of pinned upstream non-streaming assembly.
        start = torch.cat([text([1, 2, 3]), torch.cat([pad.expand(-1, 5, -1), bos], 1)
                           + codec[:, :-1]], dim=1)
        soft = condition.soft_tokens[i:i+1, condition.attention_mask[i]]
        expected = torch.cat([start, torch.cat([soft, eos], 1) + audio([8]),
                              pad + audio([9])], 1)
        valid = got["attention_mask"][i].bool()
        torch.testing.assert_close(got["inputs_embeds"][i:i+1, valid], expected)
        torch.testing.assert_close(got["trailing_text_hidden"][i:i+1], pad)
    assert got["attention_mask"].sum(1).tolist() == [14, 13]


def test_gold_shift_eos_masks_and_gradients(setup):
    builder, condition = setup
    codes = torch.randint(0, 4, (2, 3, 16))
    mask = torch.tensor([[True, True, True], [True, True, False]])
    batch = builder.build_teacher_forced(condition, codes, mask)
    prefix = builder.build_prefill(condition)
    for i, count in enumerate([3, 2]):
        length = int(prefix["attention_mask"][i].sum())
        labels = batch["labels"][i]
        assert (labels[:length] == -100).all()
        assert labels[length:length+count].tolist() == codes[i, :count, 0].tolist()
        assert labels[length+count] == 10
        assert (labels[length+count+1:] == -100).all()
        # Logit at prefix's final BOS is trained on first gold code after causal shift.
        assert labels[1:][length-1] == codes[i, 0, 0]
        torch.testing.assert_close(batch["inputs_embeds"][i, :length],
            prefix["inputs_embeds"][i, prefix["attention_mask"][i].bool()])
    # Frozen causal control: later predictions must depend on reader input.
    logits = batch["inputs_embeds"].cumsum(1) @ torch.randn(8, 64)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 64),
                                              batch["labels"][:, 1:].reshape(-1))
    loss.backward()
    assert condition.soft_tokens.grad.abs().sum() > 0
    assert (condition.soft_tokens.grad[1, 1] == 0).all()
    assert all(p.grad is None for p in builder.model.talker.parameters())


def test_no_same_frame_leak(setup):
    builder, condition = setup
    codes = torch.randint(0, 4, (2, 3, 16))
    first = builder.build_teacher_forced(condition, codes)
    changed = codes.clone()
    changed[:, 1] = 5
    second = builder.build_teacher_forced(condition, changed)
    prefix_length = int(builder.build_prefill(condition)["attention_mask"][0].sum())
    # Frame 1's target changes, but all input positions through its predicting
    # position (the embedding of gold frame 0) remain identical.
    torch.testing.assert_close(first["inputs_embeds"][0, :prefix_length+1],
                               second["inputs_embeds"][0, :prefix_length+1])


def test_invalid_masks_fail_closed(setup):
    builder, condition = setup
    condition.attention_mask[:] = False
    with pytest.raises(ValueError, match="valid continuous token"):
        builder.build_prefill(condition)
