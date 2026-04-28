.PHONY: install data train export run-ollama run-llamacpp run-api

install:
	python -m venv .venv
	.venv/Scripts/pip install --upgrade pip
	.venv/Scripts/pip install -r requirements.txt

data:
	python scripts/prepare_data.py

train:
	python scripts/fine_tune.py

export:
	python scripts/convert_to_gguf.py
	python scripts/quantize.py --quant Q4_K_M

run-ollama:
	python scripts/generate_modelfile.py
	ollama create my-custom-model -f Modelfile
	ollama run my-custom-model

run-llamacpp:
	./llama.cpp/build/bin/llama-cli \
		-m gguf/my-custom-model-Q4_K_M.gguf \
		--chat-template llama3 \
		-c 4096 \
		-i

run-api:
	uvicorn api.main:app --reload --port 8000

test:
	pytest tests/ -v --tb=short
