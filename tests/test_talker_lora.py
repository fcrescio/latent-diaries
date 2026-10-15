import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('qwen_tts')
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

from latent_diaries.talker_lora import inject_talker_lora, load_lora_state, lora_parameters, lora_state


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_lora_zero_parity_gradient_freeze_and_serialization(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(42)
    cfg = Qwen3TTSTalkerConfig(vocab_size=1088, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        text_hidden_size=16, text_vocab_size=64, num_code_groups=16,
        rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2], 'interleaved': False},
        code_predictor_config={'vocab_size':64, 'hidden_size':16, 'intermediate_size':32,
            'num_hidden_layers':1, 'num_attention_heads':2, 'num_key_value_heads':1, 'num_code_groups':16})
    cfg._attn_implementation = 'eager'
    cfg.code_predictor_config._attn_implementation = 'eager'
    talker = Qwen3TTSTalkerForConditionalGeneration(cfg).to(device).eval().requires_grad_(False)
    original = {name: p.detach().clone() for name, p in talker.named_parameters()}
    inputs = {'inputs_embeds':torch.randn(1, 6, 16, device=device, requires_grad=True),
              'attention_mask':torch.ones(1, 6, dtype=torch.long, device=device),
              'labels':torch.zeros(1, 6, dtype=torch.long, device=device), 'use_cache':False}
    with torch.no_grad(): before = talker(**inputs).logits
    modules = inject_talker_lora(talker, rank=2, layers=2)
    assert len(modules) == 4
    assert all('layers.0.' not in m['module'] for m in modules)
    after = talker(**inputs)
    torch.testing.assert_close(after.logits, before, atol=0, rtol=0)
    after.loss.backward()
    assert inputs['inputs_embeds'].grad.abs().sum() > 0
    adapters = lora_parameters(talker)
    assert sum(float(p.grad.abs().sum()) for p in adapters if p.grad is not None) > 0
    assert all(p.grad is None for name, p in talker.named_parameters() if '.lora_' not in name)
    optimizer = torch.optim.SGD(adapters, lr=.1)
    optimizer.step()
    for name, p in talker.named_parameters():
        if '.lora_' not in name:
            old = name.replace('.base.weight', '.weight').replace('.base.bias', '.bias')
            torch.testing.assert_close(p, original[old], atol=0, rtol=0)
    saved = lora_state(talker)
    assert all(name.endswith(('.lora_a', '.lora_b')) for name in saved)
    load_lora_state(talker, saved)
    with pytest.raises(ValueError, match='keys'):
        load_lora_state(talker, {})
    with pytest.raises(ValueError, match='Freeze'):
        inject_talker_lora(talker, rank=2, layers=2)


def test_generic_loader_rejects_silent_lora_omission():
    from latent_diaries.inversion import load_backbone
    with pytest.raises(ValueError, match='LoRA checkpoint'):
        load_backbone({'talker_lora_rank':8})
