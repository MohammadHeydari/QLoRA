import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import json

# --- Config ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TRAIN_SAMPLES = 9543
MAX_VAL_SAMPLES = 2388
OUTPUT_DIR = "models/qwen-sentiment-v2"

# --- Load data ---
print("Loading data...")
with open("data/processed/train.json", encoding="utf-8") as f:
    train_raw = json.load(f)[:MAX_TRAIN_SAMPLES]
with open("data/processed/val.json", encoding="utf-8") as f:
    val_raw = json.load(f)[:MAX_VAL_SAMPLES]

train_dataset = Dataset.from_list(train_raw)
val_dataset = Dataset.from_list(val_raw)

# --- 4-bit quantization ---
print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    # torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# --- LoRA config ---
print("Applying LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Training ---
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    # max_seq_length=256,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)

print("Training started...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")