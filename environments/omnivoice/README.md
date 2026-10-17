# Frozen OmniVoice comparator runtime

This optional interpreter lives at `/opt/omnivoice/.venv` in the **same worker
image** as Qwen and vision. It does not replace either existing lock. Source:
`k2-fsa/OmniVoice@08be0b4ccbac3e13e374e86fbfead4b4cac343e2` (package 0.2.1).
Transformers 5.19.0 / Hub 1.31.0 are isolated from Qwen's Transformers 4 stack.

`uv sync --project /opt/omnivoice --locked` installs this lock. Torch/torchaudio
are excluded and appended from the primary worker through `worker-stack.pth`.
The exercised worker binary pair is torch 2.14.1+cu130 / torchaudio 2.11.0+cu130;
actual imports and codec decoding passed. These differ from upstream's uv
source-specific 2.8 constraints: do not infer compatibility from metadata alone.
CI exercises the corresponding CPU pair with the tiny actual sampler, no weights.

No model weights enter the image. OMX-01 explicitly downloaded and verified the
fixed snapshot before offline inference. See `docs/OMNIVOICE_FROZEN_20261017.md`
for the experiment boundary, exact commands, licenses and human gate.
