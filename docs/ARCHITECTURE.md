# Architecture and implementation guide

## Target graph

```
JPEG or wearable opt-in snapshot
  -> EmbeddingGemma2 (google/embeddinggemma-2, frozen vision+text configuration)
  -> visual input H: (B,768) initially, later (B,N,512) if supported
  -> reader R_theta: global MLP baseline or 4-layer resampler, 16/32/64 latents
  -> pseudo text conditioning P: (B,K,1024)
  -> Qwen3-TTS 0.6B Base Talker, Italian special language ID, fixed speaker embedding
  -> first codebook 12Hz then 15 codebooks from code predictor
  -> Qwen3-TTS tokenizer/codec decoder -> 24kHz waveform
```

No text/tokens/transcripts/generated captions at inference, aside from fixed special tokens such as Italian language ID and codec controls. Speaker reference voice is not a transcript. Audio codec stages remain real; "direct speech" refers to textless **linguistic generation**, not direct samples from the reader.

## Why a standard soft prefix alone is not enough

Inspect installed `QwenLM/Qwen3-TTS` `qwen_tts/core/models/modeling_qwen3_tts.py`. In current `generate` non-streaming branch:

- It projects text embeddings via `self.talker.text_projection(self.talker.get_text_embeddings()(input_id...))` into 1024D.
- It constructs a joint text/codec prefill from BOS, EOS, PAD, language and optional speaker embeddings.
- In `Qwen3TTSTalkerForConditionalGeneration.forward`, later generation steps add `trailing_text_hidden[:, generation_step]` or `tts_pad_embed` to the autoregressive codec embedding. In the non-streaming path trailing_text_hidden is typically all PAD; verify this assertion for the pinned upstream revision before implementation. **Both initial prefill and trailing path must be tested for text leaks.**
- Audio-code generation feeds its hidden state to a code predictor for 15 more codebooks. Stop is encoded in first-codebook EOS; some code indices are suppressed during generation.

This makes the *non-streaming*, text-first prefill the simplest candidate. We must design a `SoftConditioningBuilder` replicating the stock `generate` path but substituting `P` for the *variable text section*. Fixed protocol tokens remain real. DO NOT infer token IDs from memory: read the installed config. Separate training assembly and inference assembly but assert parity.

### Proposed new types and signatures

```python
@dataclass
class VisualCondition:
    soft_tokens: torch.Tensor    # [B,K,1024], float32/bfloat16
    attention_mask: torch.Tensor # [B,K], bool
    speaker_embedding: torch.Tensor | None = None

class SoftConditioningBuilder:
    def build_prefill(self, condition: VisualCondition, model) -> tuple[torch.Tensor, dict]:
        """Qwen protocol + projected soft tokens + codec start; returns inputs_embeds+aux."""
    def build_teacher_forced(self, condition, gold_codec_codes, model) -> dict:
        """Align first-codebook targets and optional sub-codebook loss."""
    def generate_codes(self, condition, model, max_frames: int, seed: int) -> torch.Tensor:
        """No text sequence, matches stock Qwen generation invariants."""
```

These are design contracts, NOT functions already implemented in bootstrap. Their implementation must be tested against the real Qwen library version.

### Gradient / teacher-forcing tests

For 1 batch with K soft tokens and F target frames, assert:
- `logits` predict first codebook plus EOS only on valid audio positions; all prefix positions have label -100.
- Loss aligns the target at t with previous gold audio up to t-1 (no accidental exposure of target audio code at same timestep).
- Graph produces nonzero gradient `reader.parameters()` even with Talker `.requires_grad_(False)`.
- Forward of real training does not call `.detach()` on P.
- Any speaker/Italian ID remains constant under ablation.
- Generation does **not** read a variable text token sequence or target transcripts.
- All 16 codebooks can reconstruct waveform; codes match tokenizer shape, dtypes, end frames.
- EOS and max duration stop correctly; rate != waveform sample rate.

Use teacher-forced codec-0 CE first. Keep code predictor frozen until error analysis shows need for its loss. For distillation teacher logit KL, align only comparable audio frame positions. Different prefix lengths imply different hidden position indices; ensure correct offsets. Dataset canonical audio-target codec chunks need padding masks.

## Reader baselines / ablations

A0: global normalized 768D -> layernorm + 2-layer MLP -> `K*1024` (large MLP if K=32; use low-rank bottleneck or learned query+conditioning). Confirm parameter count.

B0: vision soft tokens -> learned queries + 4 cross-attention layers, 512 hidden -> 1024 projection. Target <=25M trainable parameters; the stated ~15–25M is a design **target**, not guaranteed; print count.

B1: use only 16 or 64 query tokens; compare against global embedding.

C0: frozen Qwen; C1: LoRA ranks 8/16 on selected Qwen attention projections. Keep voice embedding fixed.

Train with 2k–5k grounding pairs as first target, then 200–500 curator-authored poetic pairs; actual sufficiency unproven. Fixed first-stage single narrator. Provide null/shuffled/held-out image baselines.

## Decoder and musical / aesthetic stance

Narrative content may be metaphorical; grounding need not equal object-caption overlap. Require meaningful response to salient visual differences, measured by human listeners who see images. Record uncertainty and speaker consistency. Avoid optimizing for automatic transcriptions; machine ASR can diagnose comprehensibility only.

## Memory stage

On M5: compute per-image E_t; maintain compact state M_t = f(M_{t-1}, E_t, previous acoustic-state summary). At first use prior visual embeddings and a learned recurrent projection; do NOT deploy full audio encoder in runtime. Never imply spoken generated phrases are factual memories of actual events. User-controlled capture with opt-in and local deletion policy.

## Hardware/deployment

Development BF16 on supported GPUs, FP32 CPU smoke. NVIDIA CUDA in Docker requires driver + NVIDIA Container Toolkit. Host MPS (Apple Silicon) is not exposed through ordinary Linux Docker GPU; document or adapt backend separately, but do not silently claim MPS GPU inside Docker. Aim for offline inference on image+text EmbeddingGemma modular configuration and compact Qwen+codec, with measured actual RAM/latency/quality after quantization. Prefer architecture consistency over small parameter counts at any cost.


## M1 shared numeric-factor probe

`SharedMapper` takes three authored numeric factors, applies a shared 128-unit
tanh MLP, and outputs 32×1024 continuous tokens around a fixed PAD base. It has
4,227,584 trainable parameters and no per-example lookup/prefix parameters. Six
factor combinations train against stock-teacher distributions; two combinations
are fixed as held-out before training (`configs/research/compositions-v1.json`).
Text creates teacher audio/logit supervision only; no text encoder, projected
teacher prefix, caption or target code sequence is loaded during evaluation.

This controlled factor experiment is separate from the future EmbeddingGemma
reader. Its inputs are manually authored semantic factors; it tests reuse and
compositional generalization of continuous conditioning, not unseen vocabulary,
natural images or artistic grounding. The null control is a zero factor vector.
Both decoding modes and the same seed/duration cap are prescribed in advance.
Human listening is required; teacher-forced held-out KL is only a diagnostic.
