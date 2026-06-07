import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_V1 = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_V1 = "models/qwen-sentiment-v1"

MODEL_V2 = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_V2 = "models/qwen-sentiment-v2"

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

test_sentences = [
    "Apple stock surges to all-time high after record earnings report",
    "Company files for bankruptcy after massive losses",
    "Trading volume remains steady with no significant changes",
    "Investors panic as market crashes amid economic fears",
    "New product launch exceeds expectations, shares jump 15%",
]

def predict(model, tokenizer, text):
    prompt = f"Analyze the sentiment of this financial text:\nText: {text}\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("Sentiment:")[-1].strip()

print("Loading fine-tuned 1st try (0.5B)...")
tokenizer_v1 = AutoTokenizer.from_pretrained(MODEL_V1)
base_v1 = AutoModelForCausalLM.from_pretrained(MODEL_V1, quantization_config=get_bnb_config(), device_map="auto")
model_v1 = PeftModel.from_pretrained(base_v1, ADAPTER_V1)

print("--- Fine-tuned 1st try (0.5B, 84%) ---")
for s in test_sentences:
    print(f"Text: {s}")
    print(f"V1:   {predict(model_v1, tokenizer_v1, s)}\n")

del model_v1, base_v1
torch.cuda.empty_cache()
import gc
gc.collect()


print("Loading fine-tuned 1st try (1.5B)...")
tokenizer_v2 = AutoTokenizer.from_pretrained(MODEL_V2)
base_v2 = AutoModelForCausalLM.from_pretrained(MODEL_V2, quantization_config=get_bnb_config(), device_map="auto")
model_v2 = PeftModel.from_pretrained(base_v2, ADAPTER_V2)

print("--- Fine-tuned 2nd try (1.5B, 89.5%) ---")
for s in test_sentences:
    print(f"Text: {s}")
    print(f"V2:   {predict(model_v2, tokenizer_v2, s)}\n")