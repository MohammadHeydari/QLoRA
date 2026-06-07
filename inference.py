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
    return result.split("Sentiment:")[-1].strip()

test_sentences = [
    "Apple stock surges to all-time high after record earnings report",
    "Company files for bankruptcy after massive losses",
    "Trading volume remains steady with no significant changes",
    "Investors panic as market crashes amid economic fears",
    "New product launch exceeds expectations, shares jump 15%",
]

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=get_bnb_config(),
    device_map="auto",
)

print("--- Base model ---")
for s in test_sentences:
    print(f"Text:  {s}")
    print(f"Base:  {predict(base_model, s)}\n")

del base_model
torch.cuda.empty_cache()

print("Loading fine-tuned model...")
ft_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=get_bnb_config(),
    device_map="auto",
)
ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_PATH)

print("--- Fine-tuned model ---")
for s in test_sentences:
    print(f"Text:       {s}")
    print(f"Fine-tuned: {predict(ft_model, s)}\n")