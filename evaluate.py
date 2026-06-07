import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "models/qwen-sentiment-v1"

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def predict(model, text):
    prompt = f"Analyze the sentiment of this financial text:\nText: {text}\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = result.split("Sentiment:")[-1].strip().lower()
    for label in ["bullish", "bearish", "neutral"]:
        if label in raw:
            return label
    return "unknown"

from datasets import load_dataset
print("Loading validation data...")
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
label_map = {0: "bearish", 1: "bullish", 2: "neutral"}
val_data = [(x['text'], label_map[x['label']]) for x in dataset['validation'].select(range(200))]

def evaluate(model, name):
    correct = 0
    unknown = 0
    for text, true_label in val_data:
        pred = predict(model, text)
        if pred == true_label:
            correct += 1
        if pred == "unknown":
            unknown += 1
    acc = correct / len(val_data) * 100
    print(f"{name}: accuracy={acc:.1f}% | correct={correct}/{len(val_data)} | unknown={unknown}")

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=get_bnb_config(),
    device_map="auto",
)
evaluate(base_model, "Base model")

del base_model
torch.cuda.empty_cache()

print("Loading fine-tuned model...")
ft_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=get_bnb_config(),
    device_map="auto",
)
ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_PATH)
evaluate(ft_model, "Fine-tuned")