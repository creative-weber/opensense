"""
convert_to_gguf.py — Merge LoRA adapter into base model and convert to GGUF.

Usage:
    python scripts/convert_to_gguf.py
"""

import subprocess
import sys
from pathlib import Path

import yaml
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

CONFIG_PATH = Path("config.yaml")
LLAMACPP_DIR = Path("llama.cpp")
CONVERT_SCRIPT = LLAMACPP_DIR / "convert_hf_to_gguf.py"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit("config.yaml not found.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def ensure_llamacpp():
    if not LLAMACPP_DIR.exists():
        print("Cloning llama.cpp …")
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
            check=True,
        )
    if not CONVERT_SCRIPT.exists():
        sys.exit(f"Conversion script not found at {CONVERT_SCRIPT}. Check your llama.cpp clone.")


def main():
    config = load_config()
    m = config["model"]
    p = config["paths"]

    model_name = m["name"]
    fine_tuned_dir = Path(p["fine_tuned"]) / model_name
    merged_dir = fine_tuned_dir.parent / f"{model_name}-merged"
    gguf_dir = Path(p["gguf_out"])
    gguf_out = gguf_dir / f"{model_name}-F16.gguf"

    if not fine_tuned_dir.exists():
        sys.exit(f"Fine-tuned model not found at {fine_tuned_dir}. Run fine_tune.py first.")

    # Merge LoRA weights into base model
    print(f"Merging LoRA adapter from {fine_tuned_dir} …")
    model = AutoPeftModelForCausalLM.from_pretrained(str(fine_tuned_dir), device_map="cpu")
    merged_model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(str(fine_tuned_dir))

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to {merged_dir}")

    # Convert to GGUF
    ensure_llamacpp()
    gguf_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting to GGUF → {gguf_out} …")
    subprocess.run(
        [
            sys.executable,
            str(CONVERT_SCRIPT),
            str(merged_dir),
            "--outfile", str(gguf_out),
            "--outtype", "f16",
        ],
        check=True,
    )
    size_mb = gguf_out.stat().st_size / (1024 ** 2)
    print(f"GGUF created: {gguf_out}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
