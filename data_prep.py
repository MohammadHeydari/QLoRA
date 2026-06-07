from datasets import load_dataset
import json
import os

print("Loading dataset...")
dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

def format_sample(sample):
    label_map = {0: "bearish", 1: "bullish", 2: "neutral"}
    text = sample['text']
    label = label_map[sample['label']]
    return {
        "text": f"Analyze the sentiment of this financial text:\nText: {text}\nSentiment: {label}"
    }

print("Formatting...")
train_data = [format_sample(x) for x in dataset['train']]
val_data = [format_sample(x) for x in dataset['validation']]

os.makedirs("data/processed", exist_ok=True)

with open("data/processed/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("data/processed/val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_data)} samples")
print(f"Val: {len(val_data)} samples")
print(f"Sample: {train_data[0]}")