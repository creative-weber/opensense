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
├── api/                    # Optional FastAPI wrapper around the running model
│   ├── main.py
│   └── requirements.txt
├── Modelfile               # Ollama Modelfile to build a named model image
├── config.yaml             # Central config — model name, hyperparams, paths
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
```

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

- [ ] CLI tool (`opensense train`, `opensense run`, `opensense export`)
- [ ] Web UI for dataset labelling
- [ ] DPO / RLHF fine-tuning support
- [ ] Docker Compose stack (training + API + Ollama)
- [ ] Benchmark suite (MMLU, HellaSwag)
- [ ] Support for vision models (LLaVA)

---

## Contributing

Pull requests are welcome. Please open an issue first for large changes.

---

## License

MIT © opensense contributors
