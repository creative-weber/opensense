"""
fine_tune.py — QLoRA fine-tuning using PEFT + TRL SFTTrainer.

Usage:
    python scripts/fine_tune.py
"""

import sys
from pathlib import Path

import yaml
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit("config.yaml not found.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    m = config["model"]
    t = config["training"]
    p = config["paths"]

    base_model_id = m["base"]
    model_name = m["name"]
    base_model_dir = Path(p["base_model"]) / base_model_id.replace("/", "--")
    processed_dir = Path(p["data_processed"])
    output_dir = Path(p["fine_tuned"]) / model_name
    device_map = t.get("device_map", "auto")

    if not processed_dir.exists():
        sys.exit(f"Processed data not found at {processed_dir}. Run prepare_data.py first.")

    model_path = str(base_model_dir) if base_model_dir.exists() else base_model_id

    use_4bit = device_map != "cpu"
    print(f"Loading model: {model_path}  (4-bit={use_4bit})")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=t["lora_r"],
        lora_alpha=t["lora_alpha"],
        lora_dropout=t.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_from_disk(str(processed_dir))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        learning_rate=t["learning_rate"],
        fp16=use_4bit,
        logging_steps=10,
        save_strategy="epoch",
        report_to="tensorboard",
        run_name=model_name,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=t["max_seq_length"],
        args=training_args,
    )

    print("Starting training …")
    trainer.train(resume_from_checkpoint=output_dir if output_dir.exists() else None)
    trainer.save_model(str(output_dir))
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
