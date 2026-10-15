This environment supplies Transformers 5.19.0 for official EmbeddingGemma 2.
The primary worker retains Qwen's required Transformers 4.57.3 and Hub 0.36.2.
Both environments live in **one** `docker/Dockerfile.worker` image, used for
local CPU, CUDA and any separately approved HF Docker Job.

The vision environment has its own `uv.lock`. Its `worker-stack.pth` appends the
primary worker site-packages and project source **after** its own site-packages:
Torch 2.14.1 and its CUDA binary dependencies are shared (`exclude-dependencies`
prevents a second Torch installation); torchvision 0.29.1 is vision-only. Transformers,
Hub, tokenizers and other vision lock dependencies take precedence in this
interpreter only. Never import the two Transformers versions in one process.
Cached safetensors/JSON provide the process boundary; no model weights are
converted, vendored or redistributed. The root lock remains unchanged.

Run `/opt/vision/.venv/bin/python scripts/cache_image_features.py ...` inside
Compose. `make test-docker` exercises both interpreters. CI additionally tests
vision with the same Torch version's CPU build, without model downloads.
