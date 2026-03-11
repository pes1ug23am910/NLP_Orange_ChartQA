"""
inference.py — Standalone inference script for the ChartQA fine-tuned model.

Usage:
    # Single-image inference (merged model):
    python inference.py --image chart.png --question "What is the value for 2022?"

    # Single-image inference (via adapter load + merge):
    python inference.py --image chart.png --question "..." --use_adapters

    # Batch evaluation on a HuggingFace dataset split:
    python inference.py --evaluate --eval_split test --eval_samples 500
    python inference.py --evaluate --use_adapters --eval_split val --eval_samples 200
"""

import argparse
from typing import Union
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Run: pip install qwen-vl-utils")


# ── Defaults (update with your HuggingFace repo IDs) ─────────────────────────
MERGED_REPO   = "Yash1608/qwen2vl-2b-chartqa-merged"
ADAPTER_REPO  = "Yash1608/qwen2vl-2b-chartqa"
BASE_MODEL    = "Qwen/Qwen2-VL-2B-Instruct"


def load_merged_model(repo: str):
    """Load the fully merged model — simplest path, no PEFT needed."""
    print(f"Loading merged model from {repo} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        repo,
        torch_dtype=torch.float16,  # FP16: T4 Turing (SM 7.5) has native FP16, no BF16 tensor cores
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    return model, processor


def load_adapter_model(adapter_repo: str, base_model: str):
    """
    Load base model + LoRA adapters, then merge before inference.
    Demonstrates the adapter workflow as required by the lab.
    """
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16,  # FP16: T4 Turing (SM 7.5) has native FP16, no BF16 tensor cores
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Attaching adapters from: {adapter_repo}")
    model = PeftModel.from_pretrained(base, adapter_repo)

    print("Merging adapters into base weights ...")
    model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(adapter_repo, trust_remote_code=True)
    return model, processor


def run_inference(
    model,
    processor,
    image_path: Union[str, Image.Image],
    question: str,
    device: str = "cuda",
) -> str:
    """
    Run greedy inference.

    Args:
        image_path: Either a filesystem path to an image file, or a PIL Image
                    object (useful when calling from relaxed_accuracy() where
                    images come from a HuggingFace dataset in-memory).
        question:   Natural-language question about the chart.
        device:     torch device string ('cuda' or 'cpu').
    """
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "Analyze the chart carefully and answer the following question "
                    "with a short, precise answer (number or brief phrase only).\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=[image_inputs], return_tensors="pt", padding=True
    ).to(device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,    # greedy decoding for deterministic answers
            repetition_penalty=1.1,
        )

    answer = processor.decode(
        gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return answer


def relaxed_correct(pred: str, ref: str, tolerance: float = 0.05) -> bool:
    """
    ChartQA relaxed-accuracy criterion.

    Returns True when:
      - pred == ref after case-folding and stripping (exact string match), OR
      - both convert to floats and |pred - ref| / max(|ref|, ε) ≤ tolerance
        (numeric match within ±5% — the official ChartQA threshold).
    """
    p = pred.strip().lower()
    r = ref.strip().lower()
    if p == r:
        return True
    try:
        pf = float(p.replace(",", ""))
        rf = float(r.replace(",", ""))
        return abs(pf - rf) <= tolerance * max(abs(rf), 1e-9)
    except ValueError:
        return False


def relaxed_accuracy(
    model,
    processor,
    hf_dataset_name: str = "HuggingFaceM4/ChartQA",
    split: str = "test",
    n_samples: int = 500,
    device: str = "cuda",
) -> float:
    """
    Batch-evaluate the model on a HuggingFace dataset split using relaxed accuracy.

    Downloads the dataset split on-the-fly (cached by HuggingFace datasets).
    Prints running accuracy every 50 samples, then prints the final result.

    Args:
        model:            A loaded (and optionally merged) Qwen2VL model.
        processor:        The matching AutoProcessor.
        hf_dataset_name:  HuggingFace dataset identifier.
        split:            Dataset split name, e.g. "test" or "val".
        n_samples:        How many samples to evaluate (None = all).
        device:           Torch device string.

    Returns:
        Relaxed accuracy as a float in [0, 1].
    """
    from datasets import load_dataset  # lazy import — not needed for single-image inference

    ds = load_dataset(hf_dataset_name, split=split)
    if n_samples is not None:
        n_samples = min(n_samples, len(ds))
        ds = ds.select(range(n_samples))
    else:
        n_samples = len(ds)

    model.eval()
    correct = 0

    for i, example in enumerate(ds):
        image     = example["image"]       # PIL Image from HF datasets
        question  = example["query"]
        label     = example["label"]
        reference = label[0] if isinstance(label, list) else label

        prediction = run_inference(model, processor, image, question, device)
        is_correct = relaxed_correct(prediction, reference)
        correct   += is_correct

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:>4}/{n_samples}]  running acc = {correct / (i+1):.4f}"
                  f"  pred={prediction!r}  ref={reference!r}")

    accuracy = correct / n_samples
    print(f"\nFinal Relaxed Accuracy — split='{split}', n={n_samples}: "
          f"{accuracy:.4f}  ({correct}/{n_samples})")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="ChartQA inference with Qwen2-VL fine-tune")
    parser.add_argument("--image",    default=None,  help="Path to chart image (single-image mode)")
    parser.add_argument("--question", default=None,  help="Question about the chart (single-image mode)")
    parser.add_argument("--merged_repo",  default=MERGED_REPO)
    parser.add_argument("--adapter_repo", default=ADAPTER_REPO)
    parser.add_argument("--base_model",   default=BASE_MODEL)
    parser.add_argument(
        "--use_adapters", action="store_true",
        help="Load base + adapters (then merge) instead of the pre-merged model"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # ── Batch evaluation flags ────────────────────────────────────────────────
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run relaxed-accuracy batch evaluation instead of single-image inference"
    )
    parser.add_argument("--eval_dataset", default="HuggingFaceM4/ChartQA",
                        help="HuggingFace dataset name for evaluation")
    parser.add_argument("--eval_split",   default="test",
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--eval_samples", type=int, default=500,
                        help="Number of samples to evaluate (default: 500; 0 = all)")
    args = parser.parse_args()

    if args.use_adapters:
        model, processor = load_adapter_model(args.adapter_repo, args.base_model)
    else:
        model, processor = load_merged_model(args.merged_repo)

    model.eval()

    if args.evaluate:
        n = args.eval_samples if args.eval_samples > 0 else None
        relaxed_accuracy(model, processor, args.eval_dataset, args.eval_split, n, args.device)
    else:
        if not args.image or not args.question:
            parser.error("--image and --question are required for single-image inference")
        answer = run_inference(model, processor, args.image, args.question, args.device)
        print("\n" + "="*50)
        print(f"Question : {args.question}")
        print(f"Answer   : {answer}")
        print("="*50)


if __name__ == "__main__":
    main()
