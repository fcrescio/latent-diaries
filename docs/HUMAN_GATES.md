# Human participation and consent protocol

This project has two kinds of human decisions: **authorization** and **aesthetic judgment**. Neither may be fabricated by an agent.

## Gate 0 — User bootstrap

User copies `.env.example` to `.env`, fills `GH_TOKEN`, `HF_TOKEN`, GitHub owner, HF namespace, REVIEW_PASSWORD. For Qwen M0, user puts a **consented** reference voice in `run-data/voices/` and sets `LD_REF_AUDIO=/workspace/run-data/voices/reference.wav`. `LD_REF_TEXT` optional. Do not paste credentials in chat. Fine-grained GitHub token for repository create/write/workflows if needed; HF token with repo write and Jobs run/manage if needed. Prefer fine-grained scopes.

The `.env` never enters a Docker build context or git repository. Docker Compose *does* pass env variables to the appropriate services; never log or expose them in an HTTP response.

## Gate 1 — Paid training resource authorization

Human sees a request stating: why local compute is inadequate; exact image tag/digest, code revision, private dataset ID, hardware flavor, current hourly price from `hf jobs hardware`, max timeout and total cost ceiling. They expressly authorize **one** run. `LD_APPROVE_PAID_JOBS=yes` is an explicit operator flag, not permission for repeated jobs. Reset it to `no` after dispatch. Agent must not edit it. Cloud Job must be killable; use `hf jobs ps` / `hf jobs cancel <id>` and a finite timeout. No billing estimates from stale price lists.

## Gate 2 — Listening

Agent writes `run-data/manifests/review-vN.jsonl` containing per-line `id`, `run_id`, `image`, `audio` (relative paths), and optional blind label. Import with `latent-diaries ingest-review --manifest ...`. Reviewer visits http://localhost:8080, authenticates and scores 1-5 for grounding, voice, evocation, surprise, with notes. Only real output may be labeled real. Agent reads authorized exports (`/api/reviews`, CSV) after human has rated; it cannot fill ratings automatically. For blind comparisons hide source model condition names from UI, preserve experimental mapping separately.

## Gate 3 — Upload/private/publicity

HF checkpoint repository private by default; uploader only permits safetensors + metadata; dataset upload requires `LD_APPROVE_DATASET_UPLOAD=yes` and is still private. Review exact contents and all consent/licenses before public publication. A private repo protects access but cannot by itself establish third-party consent to process captured images.

## Gate 4 — Wearable deployment

Photos should be deliberate or sampled only with user's opt-in, minimizing bystander identifiability. Local media can be deleted independently of weights; never archive raw surroundings automatically. Do not open review service to the Internet as-is. Consent and applicable recording/image laws vary by jurisdiction.

## Escalation response template for agent

```
HUMAN ACTION REQUIRED: [gate]
What is blocked: ...
Why: ...
Exact action: ...
Data/privacy/cost implications: ...
What can continue without this action: ...
```
