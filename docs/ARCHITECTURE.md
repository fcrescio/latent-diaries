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


## Sequence-aware factor reader (v3 M1 experiment)

`SequenceReader` encodes weather, opening and memory as three factor tokens,
then decodes learned positional queries through four width-128 transformer
decoder layers (four heads, dropout zero). Absolute and predicted-length-relative
coordinates condition each query. Key-padding masks prevent invalid queries
from influencing valid positions. A shared affine numeric length head is fitted
on training rows only. Output is a continuous `[B,K,1024]` prefix, trimmed by the
predicted length before the existing frozen-Qwen conditioning builder.

Training distills cached teacher-prefix vectors with masked per-dimension
normalized reconstruction loss. Mean/scale and length coefficients use training
examples only; no Qwen weights are loaded or optimized in this reconstruction
stage. This is a supervision/architecture recipe comparison, not an isolation of
attention alone: the affine baseline uses its established padded regression loss.
Both methods share the v3 training split, speaker and fixed sampled decoder.

The 1,195,520-parameter reader consumes authored numeric factors, not visual
embeddings. Its lower tensor error cannot establish intelligibility, image
grounding or arbitrary-utterance generalization. No variable text, teacher prefix,
target code sequence or caption is loaded during standalone evaluation.


## Exercised M2 image-feature boundary

The official `google/embeddinggemma-2` revision
`914f7f89142e33e77833254d9c9b90c3cef7303b` loads with its checkpoint-provided
vision configuration and `audio_config=None`. All parameters are frozen. The
installed Transformers 5.19.0 text backbone applies its 512→768 projection before
returning `last_hidden_state`. Global embeddings use attention-masked mean
pooling of these final states, then float32 L2 normalization. They are not the
vision tower's patch states or an invented 768D intermediate representation.

The processor receives `images=[[image] for image in batch]` with no text
argument. Its flat-list semantics combine images into one sample, so explicit
nesting is necessary. The encoder checks batch boundaries and permits only
fixed image/BOS/EOS/PAD protocol IDs. No caption, transcript, source URL or
provenance string is passed to the model. The frozen text backbone is still
needed to contextualize visual soft tokens; this does not supply linguistic
conditioning. Two real-image CUDA controls and the official processor-layout
regression test exercise this boundary.

Qwen's Transformers 4.57.3 pin is retained in the primary interpreter. The same
worker image contains `/opt/vision/.venv` with a separate locked Transformers
5.19.0 / Hub 1.31.0 / torchvision 0.29.1 environment. It appends the primary
site-packages only after its own packages to reuse locked Torch 2.14.1/CUDA
binaries. The two Transformers versions never share a process. Safetensors/JSON
feature caches are the handoff to the future image reader. See
`environments/vision/README.md` for the dependency boundary. The initial global image-reader research path is described below; final
image narration remains unaccepted.


## First global image-reader pilot

`GlobalImageReader` takes normalized `[B,768]` frozen image vectors through
LayerNorm, a width-128 linear/GELU bottleneck and a linear projection to
`[B,K,1024]`. Training-only target mean/scale normalize masked prefix-vector MSE;
a training-only affine numeric head predicts prefix length. The exercised model
has 4,062,848 trainable parameters and K=30. Neither vision nor Qwen weights
change. This is cached continuous-text-prefix distillation, not acoustic CE
training or an automatic intelligibility objective.

Local-only supervision comes from the user-authorized vision service. It sees
resized metadata-free pixels, returns short Italian training narratives, and is
stopped in `finally` before further GPU work. Its use was explicitly authorized
as an existing label-service exception to the single worker-image policy; all
reader training and EmbeddingGemma/Qwen-TTS inference use the worker. Source media
are mounted read-only. Exact hashes and all image-derived information remain in
ignored private storage, with no HF upload.

Capture-day groups are unioned across near-duplicate dHash neighbors after exact
file deduplication; connected components are assigned wholly to train or holdout.
The fixed experiment uses 50 train and 12 held-out images. Only training texts are
projected into target vectors. Evaluation inputs contain id/split/image vector
only, and the existing Qwen builder preserves the no-text prefill/trailing-path
invariants. The four preselected held-out outputs are also paired with shuffled
images; null uses zero input at the same speaker/seed. Six actual generated WAVs
make 13 blinded review pairs; repeated control assignments are intentional.
Human judgments are required for clarity and grounding. Low training MSE alone
is no acceptance evidence. The final `narrate-image` entrypoint remains closed.


The first private global-reader pilot has now failed blinded held-out
intelligibility: only its training control is almost good. Frozen post-review
analysis finds near-exact training reconstruction, large held-out vector error
and mostly incorrect held-out length predictions. A corrected-length diagnostic
does not establish recovery; the new exact-text positive has human acceptance for pronunciation/prosody. These controls do not change reader weights or establish textless
image generation. Details and aggregate evidence are in DECISIONS; individual
ratings, vector errors and audio remain private.


## Exposed-development image-query control

`ImageQueryReader` projects the same frozen global 768D vector into four learned
memory tokens, then four width-128 decoder layers predict 30 continuous vectors
from learned absolute queries. These are projections of one global embedding,
not upstream visual patch states. Causal query self-attention isolates future
queries. Prefix computation is independent of length, allowing an isolated
trimming-head comparison. Gold valid-position masks supervise training; predicted
length never reveals held-out teacher information to the model.

Analytical ridge leave-one-out selection uses training rows only, with an
unpenalized intercept and a prescribed positive-alpha grid. CPU tests compare its
analytic score against explicit refits. Both the original OLS and ridge heads
are applied to both frozen models for the development diagnostic; file hashes
verify persisted weights stay unchanged. Experimental length-buffer overrides
are explicit and do not affect prefix vectors.

The exercised 1,589,760-parameter query reader and ridge head do not improve the
exposed-development result. The query reader's shared width-128 linear output
also has a nonzero best-rank training-error floor; this limits fitting but does
not by itself explain the held-out failure. The prescribed raw MSE audio gate
fails, so no further speech batch is generated. M2 remains incomplete.

## Acoustic-supervised image-reader control

The v4 experiment warm-starts the canonical width128 global MLP, preserving
its constant 20-token continuous output. Frozen Qwen generates one genuine
16-codebook trajectory per training-supported authored sentence. The cached
codes are supervision only: the reader consumes the image vector, while the
Talker receives gold previous frames during teacher forcing. Its shifted
first-codebook loss includes EOS. Frame-aligned hidden states and gold earlier
codebooks feed the frozen residual predictor. The objective is codec0 CE plus
0.1 residual CE plus 0.1 training-normalized prefix reconstruction loss.

Qwen stays in evaluation mode with every parameter frozen; its forward remains
in the gradient graph so acoustic losses update the reader. A tiny real-Qwen
integration test checks nonzero reader gradients, absent teacher/backbone
gradients and objective composition. Source, teacher, initialization, optimizer,
RNG and fixed schedule fingerprints protect actual checkpoint continuation.

The separate visual diagnostic fits centered ridge and cosine centroids to
training-only category labels, with majority and shuffled-label controls. It
measures separability against local VLM labels, not human visual correctness or
grounding. Neither diagnostic classifier nor category/template lookup runs
during speech inference. Standalone generation retains the strict numeric-vector
schema and loads no acoustic targets or linguistic labels. Teacher-forced
development loss is calculated separately and cannot establish autoregressive
intelligibility. Blinded listening is required; fresh candidates stay reserved.

## Training-only semantic language priors

The next research path separates continuous language conditioning from image
calibration. Pinned EmbeddingGemma 2 encodes authored Italian training sentences
using its documented Document format. Vision/audio towers are disabled in this
training-only process; final 768D states use the same verified masked-mean/L2
pooling. The original image-only encoder remains unchanged and never accepts a
caption. Both interpreters remain in the single locked worker image.

The authored corpus combines scene clauses with forty independent continuations.
It contains 1,280 training and 160 held-out combinations, with all component words
potentially seen in training. This is a compositional control, not a new-vocabulary
or unrestricted storytelling benchmark. There is no mandatory shared ending.
Frozen Qwen projects training text into continuous 1024D supervision; these
targets never enter standalone image generation.

Whole-prefix rectified flow (v5) fails even training-feature speech controls.
Causal next-vector flow (v6) improves stopping but retains fragmented auxiliary
transcriptions. Exact-prefix/subspace oracle positives are distinct diagnostics;
one pair produces byte-identical speech. These negative fits are retained.

The v7 alternative uses a causal two-layer GRU over previous continuous vectors,
numeric position and a semantic vector. It predicts eight context-dependent
continuous candidates plus routing and numeric stop logits. Training minimizes
best-candidate reconstruction, routing CE and EOS BCE on shifted gold previous
vectors; inference uses generated previous vectors. Candidate indices have no
fixed word/category semantics, and no text vocabulary, pretrained-vocabulary
quantization or sentence memory bank is loaded. A training-only linear token
subspace preserves supervision with a measured reconstruction floor. Its basis
is a learned numeric weight matrix, not a runtime per-example lookup.

Exact-sentence semantic controls yield coherent auxiliary transcriptions, but
v7 image transfer through category-average semantic calibration fails. V8
therefore trains on sentence, category-average, jittered and zero semantic
conditions, with 32 continuous candidates. Primary generation samples among the
four highest routing logits at fixed seed/temperature, using candidate means
without Gaussian vector noise. This tests robust conditioning under semantic
uncertainty; it does not grant speech-quality acceptance.

The numeric ridge image adapter fits training images only and loads as one
coefficient tensor followed by normalization. Current authored/image category
overlap supports 191 training cases across nine categories; 48 unsupported
training images are excluded explicitly. Inference reads strict image-vector
JSON, reader/adapter weights, fixed Qwen protocol and speaker reference only.
Text-feature controls are explicitly marked text-conditioned diagnostics.
Actual image speech and blinded human assessment are required before advancing;
all 96 fresh image candidates remain reserved.

## Semantic adapter regularization control

The v9 control keeps the v8 continuous speech prior frozen. It fits the same
769×768 affine image-to-semantic tensor, selecting ridge regularization by five
scene-group-disjoint folds of supported training images only. The fixed grid is
0.001/0.01/0.1/1; normalized vector reconstruction selects0.1. All supported
training cases are then refitted. Inference still uses one numeric projection
and L2 normalization, with no labels, classifier, prototype retrieval or text.
This addresses contraction toward the global mean, not fine visual specificity:
broad training categories and authored opening clauses still limit grounding.
The development comparison is not fresh generalization or human acceptance.

## Training-only visual anchors

V10 uses the same sixty authored opening clauses and their existing training
sentence embeddings. The local training-label service selects a visibly
supported opening subject or skips an image. The adapter's target is the
normalized mean of training sentence vectors for that specific opening, rather
than its broad scene category. No new vocabulary, caption generator, classifier
or target lookup enters inference. The initial control keeps the v8 continuous
speech prior frozen. Because unsupported images are skipped, this changes the
training subset along with supervision and cannot isolate granularity alone.
The reserved fresh split remains excluded; human grounding assessment is needed.

The shorter v11 training annotation provides the previously available broad
scene hint and only six within-scene choices. This hint is training supervision,
never an inference classifier. It increases accepted coverage to 141 images/34
anchors, but vector regression alone still produces mixed development speech.
V12 therefore freezes all 9,406,113 language-prior parameters and trains only the
590,592 affine image coefficients through continuous opening-vector likelihood.
A mixture over the existing context-dependent continuous candidates scores gold
opening vectors, with shifted gold previous vectors and valid-position masks.
Opening boundaries come from the exact shared prefix across training-only
continuations. First eight positions receive extra weight; different-anchor
image negatives and drift from the initial adapter regularize the fit.

No text vocabulary logits or tokenizer are added. The frozen GRU uses its native
forward during calibration because cuDNN evaluation does not retain backward
reserve for input gradients. CPU/CUDA tests check nonzero adapter gradients,
absent prior gradients, causal isolation and native/cuDNN FP32 forward agreement (TF32 disabled in the parity control).
Standalone generation retains the same numeric projection, frozen prior and
Qwen path; it uses generated previous vectors and reads no training targets.

The exercised v14 fallback fine-tunes the continuous reader on opening-specific,
broad-scene, exact-sentence and zero conditions while keeping vendor models
frozen. V15 addresses remaining semantic shrinkage with a scalar deviation gain:
`b' = gain*b + (1-gain)*training_target_mean`, `W' = gain*W`. It is folded into
the affine tensor, so runtime still loads no target mean or label. Five training
scene-group folds refit ridge without their validation images and select gain
by macro opening NLL; the selected value is1.5. The gain-grid motivation used
exposed development diagnostics and cannot be called untouched-test evidence.
Actual human matched/shuffled assessment and later fresh evaluation remain open.

Optional v13 continuous-density guidance compares actual candidate vectors
under conditioned/null neural mixtures, so its result is invariant to arbitrary
null-mode indices. The bounded numeric control did not improve opening choices;
it is disabled in the current audio comparison. Do not present its implementation
as accepted grounding. Native/cuDNN FP32 tests explicitly disable TF32 for the
comparison; actual sampled generation retains its previous precision settings.
