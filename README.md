# QLoRA

Hands-on QLoRA experiments on a 6GB consumer GPU. This repo documents real experiments — including what worked and what didn't.

## Current Project: Financial Sentiment Analysis

Fine-tuning `Qwen2.5-Instruct` to classify financial tweets as **bullish**, **bearish**, or **neutral**.

### Results

| Model | Accuracy |
|-------|----------|
| Base Qwen2.5-0.5B (no fine-tuning) | 3.5% |
| Fine-tuned Qwen2.5-0.5B — 1st Try (r=16, 2 epochs) | 84.0% |
| Fine-tuned Qwen2.5-1.5B — 2nd Try (r=16, 2 epochs) | 89.5% |

Evaluated on 200 held-out validation samples from the [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) dataset.

### Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 2060 Laptop (6GB VRAM, ~4.5GB usable) |
| CUDA | 13.0 |
| Python | 3.12 |

---

## How QLoRA Works

Standard fine-tuning updates all parameters — requiring 24GB+ VRAM. QLoRA makes this feasible on consumer hardware through two techniques:

**Quantization** — The base model is loaded in 4-bit precision, reducing VRAM from ~3GB to ~800MB.

**LoRA (Low-Rank Adaptation)** — Instead of updating all parameters, small adapter matrices are injected into the attention layers.

```
Base model (frozen, 4-bit)   ~800MB VRAM
LoRA adapter (trained)        ~50MB VRAM
─────────────────────────────────────────
Trainable parameters:         0.14% of model
```

---

## Project Structure

```
QLoRA/
├── data/
│   └── processed/          # train.json and val.json (not tracked)
├── models/                 # saved LoRA adapters (not tracked)
│   ├── qwen-sentiment-v1/  # 0.5B, r=16, 2 epochs → 84.0%
│   └── qwen-sentiment-v2/  # 1.5B, r=16, 2 epochs → 89.5%
├── data_prep.py            # downloads and formats the dataset
├── train.py                # QLoRA fine-tuning script (0.5B)
├── train_.py               # QLoRA fine-tuning script (1.5B)
├── inference.py            # base vs fine-tuned comparison
├── inference_.py           # 0.5B fine-tuned vs 1.5B fine-tuned comparison
├── evaluate.py             # accuracy on validation set
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/MohammadHeydari/QLoRA.git
cd QLoRA
```

### 2. Create virtual environment
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

### 5. Windows only — set UTF-8 encoding
```powershell
$env:PYTHONUTF8=1
```

---

## Usage

### Step 1 — Prepare the dataset
```bash
python data_prep.py
```

### Step 2 — Train

1st try (0.5B):
```bash
python train.py
```

2nd try (1.5B):
```bash
python train_.py
```

### Step 3 — Compare results
```bash
python inference.py    # base vs fine-tuned
python inference_.py   # 0.5B fine-tuned vs 1.5B fine-tuned
```

### Step 4 — Evaluate accuracy
```bash
python evaluate.py
```

---

## Training Details

| Setting | 1st Try | 2nd Try |
|---------|---------|---------|
| Base model | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
| LoRA rank | r=16 | r=16 |
| LoRA alpha | 32 | 32 |
| Target modules | q_proj, v_proj | q_proj, v_proj |
| Epochs | 2 | 2 |
| Batch size | 2 | 2 |
| Gradient accumulation | 8 | 8 |
| Learning rate | 2e-4 | 2e-4 |
| Training time | ~1h 15m | ~1h 30m |
| Accuracy | 84.0% | 89.5% |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| transformers | Model loading and tokenization |
| peft | LoRA adapter creation |
| trl | SFTTrainer for fine-tuning |
| bitsandbytes | 4-bit quantization |
| datasets | Dataset loading |
| accelerate | Training utilities |

---

## References

- [QLoRA paper — Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)
- [LoRA paper — Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- [Qwen2.5 model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
