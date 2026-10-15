SHELL := /bin/bash
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
DOCKER ?= docker
.DEFAULT_GOAL := help
.PHONY: help init check-env doctor test test-docker review demo review-import m0 m0-gpu github hf-create hf-publish hf-dataset-publish hf-job-plan hf-job-run gpu-doctor
help:
	@echo 'init | check-env | doctor | test | review | demo | m0 | github | hf-create | hf-publish'
	@echo 'hf-job-plan (safe) | hf-job-run (paid, opt-in) | gpu-doctor | test-docker'
init:
	@test -e .env || cp .env.example .env
	@mkdir -p artifacts run-data/review-assets
	@chmod 600 .env
	@echo 'Fill .env. Start with: make test; make review; make demo'
check-env:
	@$(PYTHON) scripts/check_env.py
doctor:
	@$(PYTHON) -m latent_diaries.cli doctor
# Local developer dependencies: pip install -e '.[test]' (ML extras optional)
test:
	@$(PYTHON) -m pytest -q
review:
	$(DOCKER) compose up -d --build review
	$(DOCKER) compose ps
# Generates clearly labeled mock records; does not run any ML model.
demo:
	@$(PYTHON) -m latent_diaries.cli demo-data
m0:
	$(DOCKER) compose --profile experiment run --rm experiment latent-diaries qwen-baseline
m0-gpu:
	$(DOCKER) compose --profile gpu run --rm experiment-gpu latent-diaries qwen-baseline

gpu-doctor:
	$(DOCKER) compose --profile gpu run --rm experiment-gpu latent-diaries doctor

github:
	@$(PYTHON) scripts/bootstrap_github.py
hf-create:
	@$(PYTHON) -m latent_diaries.hf_ops create
hf-publish:
	@$(PYTHON) -m latent_diaries.hf_ops publish --checkpoint artifacts/checkpoint-ready
hf-dataset-publish:
	@$(PYTHON) -m latent_diaries.hf_ops dataset --folder run-data/publish-dataset
hf-job-plan:
	@bash scripts/hf_job.sh --dry-run
hf-job-run:
	@bash scripts/hf_job.sh --execute
# Builds and tests containers without a GPU or remote tokens.
test-docker:
	$(DOCKER) compose --profile experiment build review experiment
	$(DOCKER) compose --profile experiment run --rm experiment latent-diaries doctor
	$(DOCKER) compose --profile experiment run --rm experiment python -m pytest -q
	$(DOCKER) compose --profile experiment run --rm experiment /opt/vision/.venv/bin/python -m pytest -q tests/test_vision.py tests/test_vision_runtime.py
