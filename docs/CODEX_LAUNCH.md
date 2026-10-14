# Copy/paste instruction for a Codex-like coding agent

Execute from the root of this cloned/unzipped GitHub repository, **after the user has placed tokens in `.env`**:

```
You are the implementing research engineer for Latent Diaries.
Read AGENTS.md, README.md, docs/ARCHITECTURE.md, docs/MILESTONES.md,
docs/HUMAN_GATES.md and docs/OPERATIONS.md completely.

Run local test suite and reproduce baseline. Then progress from M0 through M5.
Implement real Qwen textless continuous-conditioning (not text->TTS),
strictly preserving Qwen codec protocol, label alignment, Italian language token,
and no transcript conditioning at final inference. Test all invariants.

Use Docker Compose for local experiments, public sanitized worker container
for HF Docker Jobs when paid compute is expressly authorized, GitHub for
source and CI, private HF repos for checkpoints and consented data.
Commit working increments with tests, reproducibility manifests and run logs.

Prepare blind image/audio listening samples in the local review web service.
At each human gate, state exactly what I need to do and why. Do not fake
human ratings, spend money, expose private images or change repos public
without express approval. Keep docs/capabilities.json accurate. Report
negative experiments honestly. Continue technical work when not blocked.
```

## If Codex stops at a human gate

Answer explicitly with the requested cost authorization, reference voice, private dataset permission, or review scores; then resume with the same prompt: `Continue from the next unfinished milestone in docs/MILESTONES.md, observe AGENTS.md and current gate approvals.`

## Human bootstrap responsibility

Do not give the agent token contents in a chat. Edit `.env` on the local clone. The agent may read environment variables only for the tasks they are intended for. Prefer per-service restricted tokens, revoke compromised tokens, and avoid terminal commands that echo env values.
