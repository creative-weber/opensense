# opensense

> A starter kit for building, fine-tuning, and running your own custom AI model — powered by [Ollama](https://ollama.com) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

---

## What is opensense?

**opensense** is an open-source kickstarter project that gives you everything you need to:

- Collect and prepare your own training / fine-tuning dataset
- Fine-tune a base LLM (e.g. Llama 3, Mistral, Phi-3) on your data
- Convert and quantize the result into a portable GGUF model
- Run your model locally with **Ollama** or **llama.cpp**
- Expose it through a simple REST / streaming API

You own the full pipeline — data, weights, and inference.

---

## Supported Runtimes

| Runtime | Description |
|---|---|
| **Ollama** | One-command model server with a REST API. Best for local dev and quick demos. |
| **llama.cpp** | Lightweight C++ inference engine. Best for edge devices and production self-hosting. |

---

## Project Structure

```
opensense/
├── data/                   # Raw and processed datasets
│   ├── raw/                # Original source files (JSONL, CSV, TXT)
│   └── processed/          # Tokenised / formatted training data
├── scripts/                # Automation scripts (data prep, training, conversion)
│   ├── prepare_data.py
│   ├── fine_tune.py
│   ├── convert_to_gguf.py
│   └── quantize.py
├── models/                 # Downloaded base models and fine-tuned checkpoints
│   ├── base/
│   └── fine_tuned/
├── gguf/                   # Converted GGUF files ready for Ollama / llama.cpp
├── api/                    # FastAPI wrapper around the running model
│   ├── main.py             # Routes: /chat, /chat/stream, /learn, /memory, /health
│   ├── memory.py           # ChromaDB-backed vector memory store
│   └── authenticator.py    # Two-layer authenticity gate for new information
├── memory/                 # Persistent ChromaDB vector storage (auto-created)
│   └── db/
├── Modelfile               # Ollama Modelfile to build a named model image
├── config.yaml             # Central config — model name, hyperparams, paths, memory
├── requirements.txt        # Python dependencies
├── BUILD_GUIDE.md          # Step-by-step build instructions
├── SCRUM_PLAN.md           # Scrum / sprint plan for the project
└── Readme.md               # This file
```

---

## Quick Start

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.10 | https://python.org |
| pip / uv | latest | `pip install uv` |
| CUDA (optional) | ≥ 12.1 | For GPU training |
| Ollama | latest | https://ollama.com/download |
| CMake + gcc | latest | For building llama.cpp from source |

### 1 — Clone the repo

```bash
git clone https://github.com/your-org/opensense.git
cd opensense
```

### 2 — Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

> **Windows (cmd):** `pip` may not be on your PATH. Use `python -m pip` instead:
> ```cmd
> python -m pip install -r requirements.txt
> ```

```bash
# macOS / Linux
pip install -r requirements.txt
```

### 3 — Download a base model

Edit `config.yaml` to choose your base model, then run:

```bash
python scripts/prepare_data.py --download-base
```

### 4 — Prepare your dataset

Place raw files in `data/raw/` (JSONL format, one `{"prompt": "...", "response": "..."}` per line), then:

```bash
python scripts/prepare_data.py
```

### 5 — Fine-tune

```bash
python scripts/fine_tune.py
```

### 6 — Convert to GGUF

```bash
python scripts/convert_to_gguf.py
python scripts/quantize.py --quant Q4_K_M   # recommended for most hardware
```

### 7a — Run with Ollama

```bash
ollama create my-model -f Modelfile
ollama run my-model
```

### 7b — Run with llama.cpp

```bash
./llama.cpp/main -m gguf/my-model-Q4_K_M.gguf -p "Hello, who are you?"
```

### 8 — (Optional) Start the REST API

```bash
uvicorn api.main:app --reload --port 8000
# POST http://localhost:8000/chat  { "message": "Hello" }
```

### 9 — Teach your model new information

Once the API is running you can feed new facts into the model's persistent memory.
Each submission passes a two-layer authenticity gate (heuristics + model-assisted check)
before being stored. Future chat responses will automatically use relevant memories as context.

```bash
# Submit a fact
curl -X POST http://localhost:8000/learn \
  -H "Content-Type: application/json" \
  -d '{"information": "The speed of light in a vacuum is 299,792 km/s.", "source": "user"}'

# Response — fact accepted and stored:
# { "stored": true, "fact_id": "a3f1...", "verdict": "accepted", "confidence": 0.92, "reason": "..." }

# List all stored memories
curl http://localhost:8000/memory

# Remove a specific memory by its ID
curl -X DELETE http://localhost:8000/memory/<fact_id>
```

**How memory augments chat:**

```bash
# Every /chat call automatically retrieves relevant memories and prepends them as context
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the speed of light?"}'

# Response includes which memories were injected and whether cold recall occurred:
# {
#   "response": "...",
#   "memory_hits": ["The speed of light in a vacuum is 299,792 km/s."],
#   "cold_memory_hits": [],
#   "cold_recall_delay_applied": false
# }
```

**Two-tier memory (warm / cold):**

opensense models human long-term memory with two storage tiers:

| Tier | Description |
|---|---|
| **Warm** | Recently-accessed facts. Scores are recency-weighted — a fact accessed 3 days ago scores at ~93 % of raw similarity; one not touched for 30 days scores at ~50 %. Retrieval is instant. |
| **Cold** | Facts not accessed for `dump_after_days` (default **6 months**). Retrieval incurs a configurable async delay (`cold_recall_delay_seconds`, default 3 s) and a score penalty (`cold_score_penalty`, default 0.6×). A cold memory that is successfully recalled is automatically **restored to warm** — mirroring how re-exposure re-consolidates a forgotten memory. |

```
recency score  =  semantic_score × 2^(−age_days / half_life_days)
cold score     =  semantic_score × cold_score_penalty
```

**Trigger a memory dump manually:**

```bash
# Move all warm facts not accessed for dump_after_days to cold storage
curl -X POST http://localhost:8000/memory/dump

# { "dumped": 12, "warm_remaining": 38, "cold_total": 12 }
```

**Inspect cold storage:**

```bash
curl http://localhost:8000/memory/cold
```

**Authenticity gate — what gets rejected:**
- Text under 20 characters or over 4,000 characters
- Mostly symbols / gibberish (> 60% non-alphanumeric)
- Prompt-injection patterns (`ignore previous instructions`, jailbreak phrases, etc.)
- Facts the model itself classifies as `FAKE` or implausible

---

## Configuration (`config.yaml`)

```yaml
model:
  base: "meta-llama/Meta-Llama-3-8B"   # HuggingFace model ID
  name: "my-custom-model"
  quant: "Q4_K_M"

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  lora_r: 16
  lora_alpha: 32
  max_seq_length: 2048

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  base_model: "models/base"
  fine_tuned: "models/fine_tuned"
  gguf_out: "gguf"

api:
  backend: "ollama"           # "ollama" | "llamacpp"
  ollama_url: "http://localhost:11434"
  llamacpp_url: "http://localhost:8080"
  model_name: "my-custom-model"
  api_key: ""                 # leave empty to disable auth

memory:
  enabled: true               # set false to disable the learning feature entirely
  persist_dir: "memory/db"   # ChromaDB storage path (warm/ and cold/ sub-dirs auto-created)
  top_k: 5                    # max facts retrieved per chat request
  min_score: 0.45             # minimum score to inject a fact (0–1)
  model_check: true           # run model-assisted authenticity check on new facts
  min_confidence: 0.6         # minimum confidence score to accept a fact (0–1)

  # Two-tier memory
  dump_after_days: 180        # move fact to cold after 6 months of no access
  recency_half_life_days: 30  # warm score halves every 30 days without access
  cold_score_penalty: 0.6     # penalise cold semantic scores (simulates imperfect recall)
  cold_recall_delay_seconds: 3.0  # async delay when cold store is consulted
```

| `memory` key | Default | Description |
|---|---|---|
| `enabled` | `true` | Toggle the entire learning feature on/off |
| `persist_dir` | `memory/db` | Where ChromaDB persists vectors on disk (`warm/` and `cold/` sub-dirs created automatically) |
| `top_k` | `5` | How many memories to inject into each chat prompt |
| `min_score` | `0.45` | Score threshold for injection (applied after recency decay for warm; after penalty for cold) |
| `model_check` | `true` | Use the LLM to verify authenticity of new facts |
| `min_confidence` | `0.6` | Combined heuristic+model score required to store a fact |
| `dump_after_days` | `180` | Days of no access before a fact is moved to cold storage (≈ 6 months) |
| `recency_half_life_days` | `30` | Warm score halves every this many days without access |
| `cold_score_penalty` | `0.6` | Multiplier on semantic score for cold-tier results (simulates imperfect recall) |
| `cold_recall_delay_seconds` | `3.0` | Async delay (seconds) added when cold store is consulted (simulates slow recognition) |

---

## Data Format

Training data must be in **JSONL** format:

```jsonl
{"prompt": "What is photosynthesis?", "response": "Photosynthesis is the process by which plants convert sunlight into energy..."}
{"prompt": "Summarise this article: ...", "response": "The article discusses..."}
```

Use the `<|user|>` / `<|assistant|>` chat template when targeting an instruction-tuned base model.

---

## Roadmap

- [x] Persistent vector memory (`/learn`, `/memory` endpoints)
- [x] Two-layer authenticity gate (heuristics + model-assisted)
- [x] Memory-augmented chat (auto-injects relevant facts into prompts)
- [x] Memory decay / forgetting policy — two-tier warm/cold store with recency decay and slow-recall simulation
- [ ] CLI tool (`opensense train`, `opensense run`, `opensense export`)
- [ ] Web UI for dataset labelling and memory inspection
- [ ] DPO / RLHF fine-tuning support
- [ ] Docker Compose stack (training + API + Ollama)
- [ ] Benchmark suite (MMLU, HellaSwag)
- [ ] Support for vision models (LLaVA)
- [ ] Multi-user memory namespacing

---

## Open for Collaboration

opensense welcomes contributors of all experience levels. Whether you want to fix a bug,
add a feature from the roadmap, improve documentation, or propose a new idea — you are
very welcome here.

**Ways to get involved:**

| Area | What's needed |
|---|---|
| Core pipeline | Fine-tuning improvements, new base model support |
| Learning feature | Memory decay policies, multi-user namespacing, UI for memory inspection |
| Tooling | CLI (`opensense train / run / export`), Docker Compose stack |
| Evaluation | Benchmark integration (MMLU, HellaSwag, custom evals) |
| Documentation | Tutorials, example notebooks, translated guides |
| Testing | Unit and integration test coverage |

**How to contribute:**

1. Browse [open issues](https://github.com/your-org/opensense/issues) or open a new one to discuss your idea
2. Fork the repo and create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes and add tests where applicable
4. Open a pull request — describe what you changed and why

No contribution is too small. Typo fixes and doc improvements are just as valued as new features.

---

## Contributing

Pull requests are welcome. Please open an issue first for large changes.

---

## License

MIT © opensense contributors
