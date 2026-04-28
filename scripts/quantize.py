"""
quantize.py — Quantise a GGUF file using llama.cpp llama-quantize.

Usage:
    python scripts/quantize.py --quant Q4_K_M
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path("config.yaml")
LLAMACPP_BIN = Path("llama.cpp") / "build" / "bin"
QUANTIZE_BIN = LLAMACPP_BIN / "llama-quantize"

VALID_QUANTS = ["Q2_K", "Q4_K_M", "Q5_K_M", "Q8_0", "Q4_0", "Q4_1"]


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit("config.yaml not found.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", default=None, help=f"Quantisation type: {', '.join(VALID_QUANTS)}")
    args = parser.parse_args()

    config = load_config()
    m = config["model"]
    model_name = m["name"]
    quant = args.quant or m.get("quant", "Q4_K_M")

    if quant not in VALID_QUANTS:
        sys.exit(f"Invalid quant '{quant}'. Choose from: {', '.join(VALID_QUANTS)}")

    gguf_dir = Path(config["paths"]["gguf_out"])
    input_gguf = gguf_dir / f"{model_name}-F16.gguf"
    output_gguf = gguf_dir / f"{model_name}-{quant}.gguf"

    if not input_gguf.exists():
        sys.exit(f"Input GGUF not found: {input_gguf}. Run convert_to_gguf.py first.")

    quantize_bin = QUANTIZE_BIN
    if sys.platform == "win32":
        quantize_bin = QUANTIZE_BIN.with_suffix(".exe")

    if not quantize_bin.exists():
        sys.exit(
            f"llama-quantize binary not found at {quantize_bin}. "
            "Build llama.cpp first: cmake -B llama.cpp/build && cmake --build llama.cpp/build"
        )

    in_size = input_gguf.stat().st_size / (1024 ** 2)
    print(f"Quantising {input_gguf.name}  ({in_size:.0f} MB) → {quant} …")

    subprocess.run([str(quantize_bin), str(input_gguf), str(output_gguf), quant], check=True)

    out_size = output_gguf.stat().st_size / (1024 ** 2)
    reduction = (1 - out_size / in_size) * 100
    print(f"Done: {output_gguf.name}  ({out_size:.0f} MB, {reduction:.0f}% smaller)")


if __name__ == "__main__":
    main()
