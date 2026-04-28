# opensense — Scrum Plan

**Project:** opensense AI Model Starter Kit  
**Sprint Length:** 1 week  
**Team Size:** 1–3 engineers  
**Methodology:** Scrum with weekly sprints

---

## Epic Overview

| Epic | Goal |
|---|---|
| E1 — Foundation | Repo, tooling, CI skeleton |
| E2 — Data Pipeline | Ingest, validate, format, tokenise |
| E3 — Training Pipeline | LoRA / QLoRA fine-tune script + checkpointing |
| E4 — Export & Quantise | GGUF conversion + quantisation |
| E5 — Inference Runtimes | Ollama integration + llama.cpp integration |
| E6 — API Layer | FastAPI wrapper with streaming |
| E7 — DX & Docs | CLI, README polish, BUILD_GUIDE, examples |
| E8 — Hardening | Tests, Docker, benchmarks |

---

## Sprint 1 — Foundation & Skeleton (Week 1)

**Goal:** A clean, runnable repo that a new contributor can clone and understand in < 30 minutes.

### User Stories

| ID | Story | Points | Epic |
|---|---|---|---|
| S1-1 | As a developer, I can clone the repo and install all Python deps with a single command | 2 | E1 |
| S1-2 | As a developer, I have a `config.yaml` that centrally controls model name, paths, and hyperparameters | 3 | E1 |
| S1-3 | As a developer, there is a `.gitignore` that excludes model weights, GGUF files, and Python artefacts | 1 | E1 |
| S1-4 | As a developer, there is a `Makefile` with targets: `install`, `data`, `train`, `export`, `run` | 3 | E1 |
| S1-5 | As a developer, GitHub Actions runs `pip install` and a smoke test on every push | 3 | E1 |

### Tasks

- [ ] Initialise git repo with `main` and `develop` branches
- [ ] Create `requirements.txt` with pinned versions
- [ ] Create `config.example.yaml` with all documented keys
- [ ] Create `Makefile`
- [ ] Add `.gitignore` (Python, models, GGUF, `.env`)
- [ ] Set up GitHub Actions CI (`.github/workflows/ci.yml`)
- [ ] Write `CONTRIBUTING.md`

**Sprint 1 Acceptance Criteria:**
- `make install` completes without errors on a fresh Python 3.10 env
- `config.yaml` is validated against a schema on load
- CI badge is green on `main`

---

## Sprint 2 — Data Pipeline (Week 2)

**Goal:** Reliably ingest raw data, validate it, apply chat templates, and produce a tokenised HuggingFace dataset.

### User Stories

| ID | Story | Points | Epic |
|---|---|---|---|
| S2-1 | As a practitioner, I can place JSONL files in `data/raw/` and run one command to prepare them | 3 | E2 |
| S2-2 | As a practitioner, the script validates schema and reports malformed rows | 2 | E2 |
| S2-3 | As a practitioner, the script deduplicates examples by exact-match hash | 2 | E2 |
| S2-4 | As a practitioner, the script applies the correct chat template for the chosen base model | 3 | E2 |
| S2-5 | As a practitioner, I see a summary (total examples, token length histogram) after preparation | 2 | E2 |
| S2-6 | As a developer, `make data` runs `prepare_data.py` end-to-end | 1 | E2 |

### Tasks

- [ ] Implement `scripts/prepare_data.py`
  - [ ] JSONL loader with line-by-line validation
  - [ ] Schema validator (`prompt` / `response` or `messages` keys)
  - [ ] Deduplication (sha256 hash of `prompt+response`)
  - [ ] Chat template formatter (Llama 3, ChatML, Alpaca)
  - [ ] HuggingFace `datasets` save to `data/processed/`
  - [ ] Summary statistics printout
- [ ] Add `--download-base` flag to download the HF base model
- [ ] Unit tests for validator and formatter
- [ ] Add example dataset `data/raw/example.jsonl` (50 rows)

**Sprint 2 Acceptance Criteria:**
- `make data` succeeds on the included example dataset
- Malformed rows produce a clear error message with line number
- 0 duplicate examples in processed output

---

## Sprint 3 — Training Pipeline (Week 3)

**Goal:** Fine-tune a 7B base model using QLoRA on a GPU, with checkpointing and resumable training.

### User Stories

| ID | Story | Points | Epic |
|---|---|---|---|
| S3-1 | As a practitioner, I can run `make train` to start LoRA fine-tuning | 3 | E3 |
| S3-2 | As a practitioner, training resumes from the last checkpoint if interrupted | 3 | E3 |
| S3-3 | As a practitioner, training loss is logged to TensorBoard | 2 | E3 |
| S3-4 | As a practitioner, I can configure all key hyperparameters in `config.yaml` without touching Python code | 2 | E3 |
| S3-5 | As a developer, the script works in CPU-only mode (slow, for CI) | 2 | E3 |

### Tasks

- [ ] Implement `scripts/fine_tune.py`
  - [ ] Load base model with 4-bit BnB quantisation
  - [ ] Apply LoRA via `peft`
  - [ ] Use `trl.SFTTrainer` with the processed dataset
  - [ ] Save checkpoints every epoch
  - [ ] TensorBoard / W&B logging (optional)
- [ ] Add `resume_from_checkpoint` support
- [ ] Validate config before training starts
- [ ] CPU fallback path (auto-detected)
- [ ] Integration test: 5-step training run on synthetic data

**Sprint 3 Acceptance Criteria:**
- `make train` completes 1 epoch on the example dataset (GPU required for realistic speed)
- Checkpoints are saved under `models/fine_tuned/`
- Resuming from checkpoint continues from the correct step

---

## Sprint 4 — GGUF Export & Quantisation (Week 4)

**Goal:** Merge LoRA weights, convert to GGUF, and quantise to at least Q4_K_M.

### User Stories

| ID | Story | Points | Epic |
|---|---|---|---|
| S4-1 | As a practitioner, I can run `make export` to produce a GGUF file | 3 | E4 |
| S4-2 | As a practitioner, I can choose the quantisation level via a flag | 2 | E4 |
| S4-3 | As a practitioner, the script automatically clones llama.cpp if not present | 2 | E4 |
| S4-4 | As a practitioner, I see file sizes before and after quantisation | 1 | E4 |

### Tasks

- [ ] Implement `scripts/convert_to_gguf.py`
  - [ ] Merge LoRA adapter into base model
  - [ ] Auto-clone llama.cpp if missing
  - [ ] Call `llama.cpp/convert_hf_to_gguf.py`
  - [ ] Output `gguf/<model>-F16.gguf`
- [ ] Implement `scripts/quantize.py`
  - [ ] Accept `--quant` flag (Q2_K, Q4_K_M, Q5_K_M, Q8_0)
  - [ ] Call `llama.cpp/build/bin/llama-quantize`
  - [ ] Print before/after file sizes
- [ ] Verify GGUF file can be loaded by llama.cpp

**Sprint 4 Acceptance Criteria:**
- `make export` produces a loadable GGUF file
- `quantize.py --quant Q4_K_M` reduces file size by ~50% from F16

---

## Sprint 5 — Inference Runtimes (Week 5)

**Goal:** Run the model interactively via both Ollama and llama.cpp with a documented `Modelfile`.

### User Stories

| ID | Story | Points | Epic |
|---|---|---|---|
| S5-1 | As a practitioner, `make run-ollama` builds the Ollama model and starts an interactive session | 3 | E5 |
| S5-2 | As a practitioner, `make run-llamacpp` starts an interactive llama.cpp chat session | 2 | E5 |
| S5-3 | As a practitioner, the `Modelfile` is generated with the correct GGUF path automatically | 2 | E5 |
| S5-4 | As a practitioner, I can switch between Ollama and llama.cpp by changing one line in `config.yaml` | 2 | E5 |

### Tasks

- [ ] Write `Modelfile` template with configurable SYSTEM prompt and PARAMETER block
- [ ] Add `scripts/generate_modelfile.py` that fills in the GGUF path from config
- [ ] Add `make run-ollama` and `make run-llamacpp` Makefile targets
- [ ] Document llama.cpp build flags for macOS (Metal) and Linux (CUDA)
- [ ] Smoke test: send a prompt to Ollama REST API and assert non-empty response

**Sprint 5 Acceptance Criteria:**
- `make run-ollama` succeeds when Ollama is installed and the GGUF exists
- `make run-llamacpp` succeeds when llama.cpp is built
- Both runtimes return a response to a test prompt

---

## Sprint 6 — API Layer (Week 6)

**Goal:** A clean FastAPI service that proxies requests to either Ollama or llama.cpp with streaming support.

### User Stories

| ID | Story | Points | Epic |
|---|---|---|---|
| S6-1 | As a developer, `POST /chat` returns a JSON response from the running model | 3 | E6 |
| S6-2 | As a developer, `POST /chat/stream` streams tokens via Server-Sent Events | 3 | E6 |
| S6-3 | As a developer, `GET /health` returns `{"status": "ok"}` and the backend in use | 1 | E6 |
| S6-4 | As a developer, the API backend (Ollama / llama.cpp) is configured in `config.yaml` | 2 | E6 |
| S6-5 | As a developer, the API has basic auth (API key via `Authorization: Bearer` header) | 3 | E6 |

### Tasks

- [ ] Implement `api/main.py` with FastAPI
- [ ] Implement `POST /chat` (non-streaming)
- [ ] Implement `POST /chat/stream` (SSE)
- [ ] Implement `GET /health`
- [ ] API key authentication middleware
- [ ] Request/response Pydantic models
- [ ] Unit tests for each endpoint (mock the backend)
- [ ] Add `make run-api` Makefile target

**Sprint 6 Acceptance Criteria:**
- All three endpoints respond correctly
- Requests without a valid API key return 401
- Streaming endpoint delivers tokens incrementally (verified with `curl`)

---

## Sprint 7 — DX, CLI & Docs (Week 7)

**Goal:** Polish the developer experience with a CLI tool and complete documentation.

### User Stories

| ID | Story | Points | Epic |
|---|---|---|---|
| S7-1 | As a practitioner, I can use `opensense train`, `opensense export`, `opensense run` from the terminal | 5 | E7 |
| S7-2 | As a new user, `opensense init` scaffolds the project structure interactively | 3 | E7 |
| S7-3 | As a reader, `Readme.md` has a clear quickstart that works end-to-end | 2 | E7 |
| S7-4 | As a reader, `BUILD_GUIDE.md` covers every step with copy-pasteable commands | 2 | E7 |

### Tasks

- [ ] Implement `cli/main.py` using `typer` or `click`
  - [ ] `opensense init` — interactive scaffold
  - [ ] `opensense data` — run data pipeline
  - [ ] `opensense train` — run fine-tuning
  - [ ] `opensense export` — convert + quantise
  - [ ] `opensense run` — start chosen runtime
  - [ ] `opensense api` — start API server
- [ ] Install CLI as `opensense` entry point in `pyproject.toml`
- [ ] Add end-to-end example notebook `notebooks/quickstart.ipynb`
- [ ] Review and finalise `Readme.md` and `BUILD_GUIDE.md`

---

## Sprint 8 — Hardening & Release (Week 8)

**Goal:** Reliable tests, Docker support, and a tagged v0.1.0 release.

### User Stories

| ID | Story | Points | Epic |
|---|---|---|---|
| S8-1 | As a developer, `pytest` runs unit and integration tests with > 80% coverage | 5 | E8 |
| S8-2 | As a practitioner, `docker compose up` starts the full stack (training + API + Ollama) | 5 | E8 |
| S8-3 | As a practitioner, a benchmark script evaluates the model on a held-out test set (BLEU / perplexity) | 3 | E8 |
| S8-4 | As a maintainer, the project is tagged `v0.1.0` with a CHANGELOG entry | 1 | E8 |

### Tasks

- [ ] Write unit tests for all scripts (`pytest tests/`)
- [ ] Write integration test: data → train 5 steps → export (mocked weights)
- [ ] Create `docker-compose.yml`
  - [ ] `trainer` service (Python + CUDA)
  - [ ] `api` service (FastAPI)
  - [ ] `ollama` service (official Ollama image)
- [ ] Implement `scripts/benchmark.py`
- [ ] Configure `pyproject.toml` with `[project]` metadata
- [ ] Tag `v0.1.0` and write `CHANGELOG.md`
- [ ] Publish announcement to project community

**Sprint 8 Acceptance Criteria:**
- `pytest` passes with ≥ 80% coverage
- `docker compose up` starts all services cleanly
- `v0.1.0` tag exists on `main`

---

## Backlog (Post v0.1.0)

| ID | Story | Priority |
|---|---|---|
| B1 | Web UI for dataset labelling | Medium |
| B2 | DPO / RLHF fine-tuning support | High |
| B3 | Support for vision models (LLaVA) | Medium |
| B4 | MMLU / HellaSwag benchmark integration | Medium |
| B5 | Model registry with versioning | Low |
| B6 | Distributed training (multi-GPU) | Low |

---

## Definition of Done

A story is **Done** when:

1. Code is merged to `develop` via a pull request
2. All existing tests still pass in CI
3. New functionality has at least one test (unit or integration)
4. Relevant documentation is updated (`Readme.md`, `BUILD_GUIDE.md`, or inline docstrings)
5. No new `ruff` / `mypy` warnings introduced

---

## Sprint Cadence

| Event | When | Duration |
|---|---|---|
| Sprint Planning | Monday 09:00 | 1 hour |
| Daily Standup | Daily 09:15 | 15 min |
| Sprint Review | Friday 16:00 | 30 min |
| Retrospective | Friday 16:30 | 30 min |
