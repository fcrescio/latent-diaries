"""Real pinned Qwen classes with random tiny weights; never speech evidence."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qwen_tts")
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

from latent_diaries.conditioning import SoftConditioningBuilder, VisualCondition
from latent_diaries.inversion import aligned_residual_logits


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pinned_talker_gradient_and_cache_parity(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this worker invocation")
    torch.manual_seed(12)
    config = Qwen3TTSTalkerConfig(
        vocab_size=1088,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        text_hidden_size=16,
        text_vocab_size=64,
        num_code_groups=16,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "interleaved": False},
        codec_think_id=4,
        codec_think_bos_id=5,
        codec_think_eos_id=6,
        codec_language_id={"italian": 7},
        codec_pad_id=8,
        codec_bos_id=9,
        codec_eos_token_id=10,
        code_predictor_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_code_groups": 16,
        },
    )
    config._attn_implementation = "eager"
    config.code_predictor_config._attn_implementation = "eager"
    talker = Qwen3TTSTalkerForConditionalGeneration(config).to(device).eval().requires_grad_(False)
    parent = SimpleNamespace(
        talker=talker,
        config=SimpleNamespace(
            talker_config=config, tts_bos_token_id=11, tts_eos_token_id=12, tts_pad_token_id=13
        ),
    )
    builder = SoftConditioningBuilder(parent, (1, 2, 3))
    soft = torch.randn(1, 3, 16, device=device, requires_grad=True)
    condition = VisualCondition(
        soft, torch.ones(1, 3, dtype=torch.bool, device=device), torch.randn(1, 16, device=device)
    )
    gold = torch.randint(0, 4, (1, 3, 16), device=device)
    batch = builder.build_teacher_forced(condition, gold)
    output = talker(**batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert soft.grad.abs().sum() > 0
    assert all(p.grad is None for p in talker.parameters())
    # Verify every residual head against generation's gold-previous-code prefix.
    frame_hidden = torch.randn(2, 16, device=device, requires_grad=True)
    frame_gold = torch.randint(0, 4, (2, 16), device=device)
    residual = aligned_residual_logits(talker, frame_hidden, frame_gold)
    inputs = [frame_hidden[:, None], talker.get_input_embeddings()(frame_gold[:, :1])]
    for j in range(1, 15):
        inputs.append(talker.code_predictor.get_input_embeddings()[j - 1](frame_gold[:, j : j + 1]))
    all_inputs = torch.cat(inputs, dim=1)
    for j in range(15):
        generated = talker.code_predictor(inputs_embeds=all_inputs[:, : j + 2], use_cache=False)
        torch.testing.assert_close(residual[:, j], generated.logits[:, -1], rtol=1e-4, atol=1e-5)
    residual_loss = torch.nn.functional.cross_entropy(
        residual.reshape(-1, 64), frame_gold[:, 1:].reshape(-1)
    )
    residual_loss.backward()
    assert frame_hidden.grad.abs().sum() > 0
    assert all(p.grad is None for p in talker.parameters())
    prefix = builder.build_prefill(condition)
    prefilled = talker(**prefix, use_cache=True, output_hidden_states=True)
    length = prefix["inputs_embeds"].shape[1]
    # The frozen code predictor generates the remaining 15 codebooks. Replay
    # that resulting frame as teacher-forced input and compare next logits.
    stepped = talker(
        input_ids=gold[:, 0, :1],
        past_key_values=prefilled.past_key_values,
        past_hidden=prefilled.past_hidden,
        generation_step=prefilled.generation_step,
        trailing_text_hidden=prefilled.trailing_text_hidden,
        tts_pad_embed=prefix["tts_pad_embed"],
        attention_mask=torch.ones(1, length + 1, dtype=torch.long, device=device),
        cache_position=torch.tensor([length], device=device),
        use_cache=True,
        subtalker_dosample=False,
        subtalker_top_p=1.0,
        subtalker_top_k=0,
        subtalker_temperature=1.0,
        output_hidden_states=True,
    )
    codes = stepped.hidden_states[-1]
    assert codes.shape == (1, 16)
    replay = builder.build_teacher_forced(condition, codes[:, None])
    full = talker(**replay)
    torch.testing.assert_close(stepped.logits[:, -1], full.logits[:, length], rtol=1e-4, atol=1e-5)
    first = builder.generate_codes(condition, max_frames=3, seed=41)
    second = builder.generate_codes(condition, max_frames=3, seed=41)
    torch.testing.assert_close(first[0], second[0])
    assert 0 < first[0].shape[0] <= 3
    assert first[0].shape[1] == 16
    assert not (first[0][:, 0] == config.codec_eos_token_id).any()
