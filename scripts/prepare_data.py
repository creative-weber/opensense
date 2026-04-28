"""
prepare_data.py — Validate, deduplicate, apply chat template, and tokenise training data.

Usage:
    python scripts/prepare_data.py               # process data/raw/ -> data/processed/
    python scripts/prepare_data.py --download-base  # also download the base model
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from datasets import Dataset
from transformers import AutoTokenizer

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit("config.yaml not found. Copy config.example.yaml to config.yaml first.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: Path) -> list[dict]:
    rows = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [ERROR] {file_path}:{line_num} — invalid JSON: {e}")
                continue
            if "prompt" in row and "response" in row:
                rows.append({"prompt": row["prompt"], "response": row["response"]})
            elif "messages" in row:
                rows.append({"messages": row["messages"]})
            else:
                print(f"  [WARN]  {file_path}:{line_num} — missing 'prompt'/'response' or 'messages' keys, skipped.")
    return rows


def deduplicate(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def apply_chat_template(rows: list[dict], tokenizer) -> list[str]:
    texts = []
    for row in rows:
        if "messages" in row:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        else:
            messages = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["response"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-base", action="store_true", help="Download the base model from HuggingFace")
    args = parser.parse_args()

    config = load_config()
    base_model_id = config["model"]["base"]
    raw_dir = Path(config["paths"]["data_raw"])
    processed_dir = Path(config["paths"]["data_processed"])
    base_model_dir = Path(config["paths"]["base_model"]) / base_model_id.replace("/", "--")

    if args.download_base:
        from huggingface_hub import snapshot_download
        print(f"Downloading base model: {base_model_id}")
        snapshot_download(repo_id=base_model_id, local_dir=str(base_model_dir))
        print(f"Saved to {base_model_dir}")

    # Load all JSONL files
    all_rows: list[dict] = []
    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        print(f"Loading {jsonl_file} …")
        rows = load_jsonl(jsonl_file)
        print(f"  {len(rows)} valid rows")
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("No valid rows found. Place JSONL files in data/raw/.")

    # Deduplicate
    before = len(all_rows)
    all_rows = deduplicate(all_rows)
    print(f"\nDeduplication: {before} → {len(all_rows)} rows ({before - len(all_rows)} removed)")

    # Apply chat template
    model_dir = base_model_dir if base_model_dir.exists() else base_model_id
    print(f"\nLoading tokenizer from {model_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    texts = apply_chat_template(all_rows, tokenizer)

    # Save
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_dict({"text": texts})
    dataset.save_to_disk(str(processed_dir))

    token_lengths = [len(tokenizer.encode(t)) for t in texts]
    print(f"\nDataset saved to {processed_dir}")
    print(f"  Total examples : {len(texts)}")
    print(f"  Min tokens     : {min(token_lengths)}")
    print(f"  Max tokens     : {max(token_lengths)}")
    print(f"  Avg tokens     : {sum(token_lengths) // len(token_lengths)}")


if __name__ == "__main__":
    main()
