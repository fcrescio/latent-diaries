"""Official installed Transformers-5 API smoke; runs only in vision environment."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    transformers.__version__ != "5.19.0", reason="Use the isolated vision interpreter"
)


def test_official_embeddinggemma2_config_and_projected_hidden_states():
    from transformers import AutoConfig, AutoModel, EmbeddingGemma2Processor

    from latent_diaries.vision import pool_global

    assert EmbeddingGemma2Processor.__name__ == "EmbeddingGemma2Processor"
    config = AutoConfig.for_model(
        "embedding_gemma2_text", hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, vocab_size=32, hidden_size_per_layer_input=16,
        embedding_dim=768, layer_types=["sliding_attention", "full_attention"],
        per_layer_config={}, sliding_window=16,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
    )
    model = AutoModel.from_config(config).eval().requires_grad_(False)
    ids = torch.tensor([[2, 4, 1, 0], [2, 5, 6, 1]])
    mask = ids != 0
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask)
    assert output.last_hidden_state.shape == (2, 4, 768)
    values = pool_global(output.last_hidden_state, mask)
    assert torch.allclose(values.norm(dim=-1), torch.ones(2), atol=1e-6)
    multimodal = AutoConfig.for_model("embedding_gemma2", audio_config=None, vision_config={})
    assert multimodal.vision_config is not None and multimodal.audio_config is None


def test_official_processor_independent_image_batch_layout():
    from PIL import Image
    from transformers import EmbeddingGemma2Processor

    # Exercise official layout semantics without weights/tokenizer downloads.
    processor = object.__new__(EmbeddingGemma2Processor)
    processor.image_token = "<image>"
    first = Image.new("RGB", (32, 32), "red")
    second = Image.new("RGB", (32, 32), "blue")
    images, markers, _, _ = processor.prepare_inputs_layout(images=[[first], [second]])
    assert len(images) == 2 and all(len(sample) == 1 for sample in images)
    assert markers == ["<image>", "<image>"]
