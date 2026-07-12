"""
train.py — QLoRA fine-tune Llama 3.2 3B for lead/spam classification.

Single-script training loop using Unsloth + TRL.
Run on any GPU with ≥12GB VRAM (RTX 3080 confirmed).
"""

import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# ── Configuration ───────────────────────────────────────────────────────────
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
OUTPUT_DIR = "outputs"
TRAIN_FILE = "data/train.jsonl"
EVAL_FILE = "data/eval.jsonl"

# ── Model ───────────────────────────────────────────────────────────────────
print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,                         # QLoRA — 4-bit quantization
    dtype=None,                                # Auto-detect
)

print("Applying LoRA adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",      # VRAM ↔ compute tradeoff
    random_state=42,
)

# ── Data ────────────────────────────────────────────────────────────────────
print("Loading datasets...")
dataset = load_dataset("json", data_files={
    "train": TRAIN_FILE,
    "eval": EVAL_FILE,
})

def format_chat(examples):
    """
    Convert `messages` list to a single text string using Llama 3.2
    chat template. This is what the model actually sees during training.
    """
    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False)
        for msgs in examples["messages"]
    ]
    return {"text": texts}

dataset = dataset.map(format_chat, batched=True)

# ── Training ────────────────────────────────────────────────────────────────
print(f"Train: {len(dataset['train'])} examples  |  Eval: {len(dataset['eval'])} examples")
print("Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        # Batch fit for 16GB VRAM
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,          # Effective batch = 8

        # Epochs — 2 is enough for classification
        num_train_epochs=2,

        # LR — higher than full fine-tune since base is frozen
        learning_rate=2e-4,
        warmup_steps=5,

        # Mixed precision
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        # Checkpointing
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        save_total_limit=3,

        # Optimizer
        optim="adamw_8bit",

        # Output
        output_dir=OUTPUT_DIR,
        seed=42,
    ),
)

trainer.train()

# ── Save ────────────────────────────────────────────────────────────────────
print("Saving adapter...")
model.save_pretrained(f"{OUTPUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")
print(f"Adapter saved to {OUTPUT_DIR}/adapter")
print("Done.")