---
language:
  - en
license: apache-2.0
base_model: Qwen/Qwen2-VL-2B-Instruct
datasets:
  - HuggingFaceM4/ChartQA
tags:
  - multimodal
  - vision-language
  - chart-qa
  - qlora
  - peft
  - fine-tuned
pipeline_tag: image-text-to-text
---

# Multimodal SLM Fine-Tuning: Qwen2-VL-2B on ChartQA

> **Orange Problem Lab** — Multimodal Fine-Tuning with Small Language Models

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/Yash1608/qwen2vl-2b-chartqa-merged)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pes1ug23am910/NLP_Orange_ChartQA/blob/main/multimodal_finetune_chartqa_Collab_v2_Final.ipynb)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com)

---

## Overview

This project fine-tunes **Qwen2-VL-2B-Instruct** — a 2-billion-parameter vision-language model — on the **ChartQA** dataset using **QLoRA** (4-bit quantisation + Low-Rank Adaptation). The entire pipeline runs on a single **NVIDIA T4 GPU** (16 GB VRAM).

Given a chart image and a natural-language question, the model produces a short, precise answer.

```
Input:  [bar chart image] + "What is the value for category B in 2022?"
Output: "47.3"
```

---

## Design Decisions

| Decision | Choice | Why |
|---|---|---|
| **Dataset** | ChartQA | Visual QA over charts demands both fine-grained OCR-level visual reading and multi-step numerical/categorical reasoning — a rich multimodal signal that pushes the model to fuse vision and language meaningfully |
| **Model** | Qwen2-VL-2B-Instruct | Best open 2B multimodal model as of early 2025; dynamic-resolution ViT encoder handles diverse chart sizes; strong ChartQA baseline before fine-tuning |
| **Fine-tuning method** | QLoRA | 4-bit NF4 base + FP16 LoRA adapters keeps peak VRAM ≈ 12 GB, well within T4 budget. We get ~95% of full fine-tune quality at ~25% the cost. FP16 (not BF16) is used because T4 is Turing (SM 7.5) and lacks native BF16 tensor cores |
| **LoRA rank** | r=16, α=32 | r=16 gives sufficient adapter capacity for ChartQA at 2B scale while being 4× faster than r=64. α=2r is a widely validated stable default |
| **LoRA targets** | Attention projections only (`q/k/v/o_proj`) | FFN modules (`gate/up/down_proj`) doubled compute time on T4 without meaningfully improving ChartQA accuracy at this scale |
| **Batch size** | 2 × 8 accum = 16 effective | Maximises T4 VRAM utilisation; gradient accumulation recovers the statistical benefit of large batches |
| **Learning rate** | 2e-4 | Standard QLoRA recommendation. Higher than full fine-tune LRs because the frozen base provides a stable anchor |
| **Image resolution cap** | 256×256 px | Halving each dimension cuts visual tokens by ~4×, the single biggest speed-up on T4. Most chart labels remain readable at this resolution |

---

## Repository Structure

```
.
├── multimodal_finetune_chartqa_Collab_v2_Final.ipynb   # Main notebook (training + eval + inference)
├── inference.py                                        # Standalone inference & batch-eval script
├── push_to_hub.py                                      # Helper: push adapters + merged model to HF Hub
├── requirements.txt                                    # Pinned dependencies
└── README.md
```

---

## Quick Start — Inference

### Install

```bash
pip install transformers==4.49.0 peft==0.14.0 accelerate==1.4.0 \
            qwen-vl-utils pillow torch
```

### Load & run the merged model

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# 1. Pull the merged (full) model — no adapter handling needed
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Yash1608/qwen2vl-2b-chartqa-merged",
    torch_dtype=torch.float16,  # FP16: T4 Turing (SM 7.5) has native FP16, no BF16 tensor cores
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(
    "Yash1608/qwen2vl-2b-chartqa-merged",
    trust_remote_code=True,
)
model.eval()

# 2. Prepare input
image = Image.open("chart.png")         # your chart image
question = "What is the highest value shown?"

messages = [{
    "role": "user",
    "content": [
        {"type": "image",  "image": image},
        {"type": "text",   "text": f"Analyze the chart and answer concisely.\n\nQuestion: {question}"},
    ],
}]

# 3. Run inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=[image_inputs], return_tensors="pt").to("cuda")

with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)

answer = processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Answer:", answer)
```

### (Alternative) Load base model + LoRA adapters, then merge

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# Step 1: load base model
base = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,  # FP16 for T4 GPU
    device_map="auto",
    trust_remote_code=True,
)

# Step 2: attach LoRA adapters
model = PeftModel.from_pretrained(base, "Yash1608/qwen2vl-2b-chartqa")

# Step 3: merge adapters into base weights (optional but recommended for deployment)
model = model.merge_and_unload()

# Processor from either the adapter repo or the base model
processor = AutoProcessor.from_pretrained("Yash1608/qwen2vl-2b-chartqa", trust_remote_code=True)
```

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2-VL-2B-Instruct` |
| Quantisation | NF4 4-bit + double quantisation |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention only) |
| Train samples | 6 000 |
| Val samples | 300 |
| Epochs | 2 |
| Effective batch size | 16 (2 × 8 gradient accum.) |
| Learning rate | 2e-4 (cosine decay) |
| Warmup | 5% of steps |
| Precision | FP16 (`fp16=True`, `bf16=False`) — T4 is Turing SM 7.5; BF16 requires Ampere SM 8.0+ |
| Max image size | 256 × 256 px |
| Max sequence length | 512 tokens |
| Hardware | 1 × NVIDIA T4 (16 GB) |

---

## Evaluation

ChartQA is scored with **relaxed accuracy**: a prediction is correct if it matches the reference exactly (string match) or within ±5% (for numeric answers).

| Split | Relaxed Accuracy |
|---|---|
| Validation (300 samples) | **~XX%** ← fill in after training |
| Test | **~XX%** |

---

## Reproducing

1. Clone the repo: `git clone https://github.com/pes1ug23am910/NLP_Orange_ChartQA`
2. Open `multimodal_finetune_chartqa_Collab_v2_Final.ipynb` in Kaggle or Google Colab (T4 runtime)
3. Set your `HF_TOKEN` as a secret
4. Update `CFG["hf_repo_id"]` with your HuggingFace username
5. Run all cells top to bottom

---

## Push to HuggingFace Hub

After training completes, use `push_to_hub.py` to publish your checkpoint:

```bash
# Push both LoRA adapters AND the merged full model (recommended)
python push_to_hub.py --hf_username Yash1608 --repo_name qwen2vl-2b-chartqa

# Push adapters only (smaller upload, ~100–400 MB)
python push_to_hub.py --hf_username Yash1608 --repo_name qwen2vl-2b-chartqa --skip_merged

# Push merged full model only
python push_to_hub.py --hf_username Yash1608 --repo_name qwen2vl-2b-chartqa --skip_adapters
```

---

## Evaluation (Relaxed Accuracy)

ChartQA is scored with **relaxed accuracy** (exact string match OR numeric match within ±5%).

You can run batch evaluation from the command line:

```bash
# Evaluate merged model on the test split (500 samples)
python inference.py --evaluate --eval_split test --eval_samples 500

# Evaluate via adapter load + merge on the val split
python inference.py --evaluate --use_adapters --eval_split val --eval_samples 200
```

Or call the function directly in Python:

```python
from inference import relaxed_accuracy, load_merged_model

model, processor = load_merged_model("Yash1608/qwen2vl-2b-chartqa-merged")
acc = relaxed_accuracy(model, processor, split="test", n_samples=500)
print(f"Relaxed Accuracy: {acc:.4f}")
```

---

## License

Base model: [Qwen License](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)  
Dataset: [ChartQA License](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)  
This fine-tune: Apache 2.0
