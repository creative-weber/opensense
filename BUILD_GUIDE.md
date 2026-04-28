# opensense — Build Guide

This guide walks you through the complete pipeline from zero to a running custom AI model.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Initialisation](#2-project-initialisation)
3. [Choosing a Base Model](#3-choosing-a-base-model)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Fine-Tuning with LoRA / QLoRA](#5-fine-tuning-with-lora--qlora)
6. [Converting to GGUF](#6-converting-to-gguf)
7. [Quantisation](#7-quantisation)
8. [Running with Ollama](#8-running-with-ollama)
9. [Running with llama.cpp](#9-running-with-llamacpp)
10. [Exposing a REST API](#10-exposing-a-rest-api)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Environment Setup

### 1.1 Python

```bash
# Verify Python version (3.10+ required)
python --version

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell
```

### 1.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes:

```
torch>=2.2.0
transformers>=4.40.0
datasets>=2.18.0
peft>=0.10.0          # LoRA / QLoRA
trl>=0.8.0            # SFTTrainer
bitsandbytes>=0.43.0  # 4-bit quantisation on GPU
accelerate>=0.28.0
sentencepiece
huggingface_hub
fastapi
uvicorn
pyyaml
```

### 1.3 GPU setup (optional but recommended)

Install CUDA 12.1+ via [NVIDIA's guide](https://developer.nvidia.com/cuda-downloads), then verify:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

For CPU-only training, set `device_map: cpu` in `config.yaml`. Expect 10–50× slower training.

### 1.4 Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download installer from https://ollama.com/download
```

Verify:

```bash
ollama --version
```

### 1.5 Build llama.cpp from source

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON    # drop -DGGML_CUDA=ON if no GPU
cmake --build build --config Release -j$(nproc)
cd ..
```

The compiled binary will be at `llama.cpp/build/bin/llama-cli`.

---

## 2. Project Initialisation

```bash
# Clone opensense
git clone https://github.com/your-org/opensense.git
cd opensense

# Create required directories
mkdir -p data/raw data/processed models/base models/fine_tuned gguf
```

Copy and edit the config:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your chosen base model and hyperparameters
```

---

## 3. Choosing a Base Model

| Model | Parameters | Best For | HuggingFace ID |
|---|---|---|---|
| Llama 3 8B | 8B | General purpose, good balance | `meta-llama/Meta-Llama-3-8B` |
| Llama 3 8B Instruct | 8B | Chat / instruction following | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Mistral 7B | 7B | Fast inference, multilingual | `mistralai/Mistral-7B-v0.3` |
| Phi-3 Mini | 3.8B | Runs on CPU / low-VRAM machines | `microsoft/Phi-3-mini-4k-instruct` |
| Qwen2 7B | 7B | Strong coding + reasoning | `Qwen/Qwen2-7B-Instruct` |

> **Rule of thumb:** start with a 7–8B instruct model. They already follow instructions; you only need to teach domain knowledge.

### Authenticate with HuggingFace (for gated models like Llama 3)

```bash
huggingface-cli login
# Paste your HF token from https://huggingface.co/settings/tokens
```

### Download the base model

```bash
python scripts/prepare_data.py --download-base
```

This saves the model under `models/base/<model-name>/`.

---

## 4. Dataset Preparation

### 4.1 Format your data

Every file in `data/raw/` must be JSONL with this schema:

```jsonl
{"prompt": "User question or instruction", "response": "Expected model output"}
```

For chat-style datasets use the full conversation template:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 4.2 Minimum dataset size

| Use Case | Minimum Examples | Recommended |
|---|---|---|
| Style / tone adaptation | 50 | 200–500 |
| Domain knowledge injection | 500 | 2 000–5 000 |
| Full task specialisation | 2 000 | 10 000+ |

### 4.3 Run the preparation script

```bash
python scripts/prepare_data.py
```

This script:
- Validates and deduplicates the JSONL files
- Applies the chat template for the chosen base model
- Tokenises and saves the dataset to `data/processed/`
- Prints a dataset summary (token counts, length distribution)

---

## 5. Fine-Tuning with LoRA / QLoRA

opensense uses **QLoRA** (4-bit quantised LoRA) by default — this lets you fine-tune a 7B model on a single 16 GB GPU (or even 8 GB with smaller batch sizes).

### 5.1 Key hyperparameters (in `config.yaml`)

```yaml
training:
  epochs: 3            # start here; increase for specialised tasks
  batch_size: 4        # reduce to 2 if OOM
  learning_rate: 2e-4  # standard LoRA LR
  lora_r: 16           # LoRA rank — higher = more capacity, more VRAM
  lora_alpha: 32       # typically 2× lora_r
  max_seq_length: 2048 # match your data; longer = more VRAM
```

### 5.2 Run training

```bash
python scripts/fine_tune.py
```

Checkpoints are saved every epoch under `models/fine_tuned/<model-name>/`.

### 5.3 Monitor training

```bash
# Optional: launch TensorBoard
tensorboard --logdir models/fine_tuned/<model-name>/runs
```

### 5.4 Expected training time (rough estimates)

| Hardware | 1 000 examples | 5 000 examples |
|---|---|---|
| RTX 3090 (24 GB) | ~8 min | ~35 min |
| RTX 3060 (12 GB) | ~20 min | ~1.5 h |
| M2 Pro (32 GB RAM) | ~45 min | ~3.5 h |
| CPU only | ~6 h | ~30 h |

---

## 6. Converting to GGUF

GGUF is the file format used by both Ollama and llama.cpp.

```bash
python scripts/convert_to_gguf.py
```

This script:
1. Merges the LoRA adapter weights into the base model
2. Saves the merged model to `models/fine_tuned/<model-name>-merged/`
3. Runs `llama.cpp/convert_hf_to_gguf.py` to produce `gguf/<model-name>-F16.gguf`

> Ensure `llama.cpp` is cloned in the project root (the script expects it there).

---

## 7. Quantisation

Quantisation shrinks the model file and speeds up inference with minimal quality loss.

```bash
# Recommended for most hardware (good quality / size tradeoff)
python scripts/quantize.py --quant Q4_K_M

# Other options
python scripts/quantize.py --quant Q5_K_M   # slightly better quality
python scripts/quantize.py --quant Q8_0     # near full precision, large file
python scripts/quantize.py --quant Q2_K     # smallest, noticeable quality drop
```

Output: `gguf/<model-name>-Q4_K_M.gguf`

| Quant | File size (7B) | VRAM needed | Quality |
|---|---|---|---|
| F16 | ~14 GB | 16+ GB | Reference |
| Q8_0 | ~7 GB | 8 GB | ≈ F16 |
| Q5_K_M | ~5 GB | 6 GB | Very good |
| Q4_K_M | ~4 GB | 5 GB | Good ✓ recommended |
| Q2_K | ~2.5 GB | 3 GB | Acceptable |

---

## 8. Running with Ollama

### 8.1 Create the Modelfile

Edit the included `Modelfile`:

```dockerfile
FROM ./gguf/my-model-Q4_K_M.gguf

SYSTEM """
You are a helpful assistant specialised in [your domain].
Answer clearly and concisely.
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
```

### 8.2 Build and run

```bash
# Build the Ollama model image
ollama create my-model -f Modelfile

# Run interactively
ollama run my-model

# Or call via REST
curl http://localhost:11434/api/generate \
  -d '{"model": "my-model", "prompt": "Hello!", "stream": false}'
```

---

## 9. Running with llama.cpp

```bash
# Interactive chat mode
./llama.cpp/build/bin/llama-cli \
  -m gguf/my-model-Q4_K_M.gguf \
  --chat-template llama3 \
  -ngl 35 \               # layers offloaded to GPU (0 = CPU only)
  -c 4096 \               # context window
  -i                      # interactive mode

# Single prompt
./llama.cpp/build/bin/llama-cli \
  -m gguf/my-model-Q4_K_M.gguf \
  -p "What is machine learning?" \
  -n 256                  # max tokens to generate
```

### llama.cpp HTTP server

```bash
./llama.cpp/build/bin/llama-server \
  -m gguf/my-model-Q4_K_M.gguf \
  --port 8080 \
  -c 4096
# OpenAI-compatible endpoint at http://localhost:8080
```

---

## 10. Exposing a REST API

The `api/` folder contains a minimal FastAPI wrapper.

```bash
# Start the API server
uvicorn api.main:app --reload --port 8000
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message, get a response |
| `POST` | `/chat/stream` | Streaming response (SSE) |
| `GET` | `/health` | Health check |

Example request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain LoRA fine-tuning in simple terms"}'
```

The API proxies requests to the Ollama server (default `http://localhost:11434`) or directly to llama.cpp server. Configure the backend in `config.yaml`:

```yaml
api:
  backend: "ollama"          # "ollama" | "llamacpp"
  ollama_url: "http://localhost:11434"
  llamacpp_url: "http://localhost:8080"
  model_name: "my-model"
```

---

## 11. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch size too large | Reduce `batch_size` to 1–2 in config |
| `RuntimeError: Expected all tensors on same device` | Mixed precision issue | Add `device_map: auto` in config |
| GGUF conversion fails | llama.cpp not cloned | `git clone https://github.com/ggerganov/llama.cpp.git` in project root |
| Ollama `model not found` | Modelfile path wrong | Check GGUF path in `Modelfile` is correct |
| Very slow CPU training | Expected | Reduce `max_seq_length` and `batch_size`; use a smaller base model |
| `huggingface_hub.errors.GatedRepoError` | Model requires HF auth | Run `huggingface-cli login` |
