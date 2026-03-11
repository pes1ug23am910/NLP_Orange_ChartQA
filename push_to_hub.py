"""
push_to_hub.py — Push LoRA adapters and/or the merged full model to HuggingFace Hub.

Steps performed:
  1. Load the locally saved adapter checkpoint from ./qwen2vl-chartqa-lora
  2. Push LoRA adapters + processor to HF Hub  (lightweight, ~100–400 MB)
  3. Load base model, merge adapters, push the full merged model to HF Hub

Usage:
    python push_to_hub.py --hf_username myusername --repo_name qwen2vl-2b-chartqa
    python push_to_hub.py --hf_username myusername --repo_name qwen2vl-2b-chartqa \\
        --hf_token hf_xxxx --adapter_dir ./qwen2vl-chartqa-lora
    python push_to_hub.py --hf_username myusername --repo_name qwen2vl-2b-chartqa \\
        --skip_adapters          # only push merged model
    python push_to_hub.py --hf_username myusername --repo_name qwen2vl-2b-chartqa \\
        --skip_merged            # only push adapters
"""

import argparse
import gc
import os

import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_DIR = "./qwen2vl-chartqa-lora"


def _load_base(device_map: str = "cpu") -> Qwen2VLForConditionalGeneration:
    """Load the base Qwen2-VL-2B-Instruct in BF16 on CPU for safe merging."""
    print(f"Loading base model '{BASE_MODEL}' onto {device_map} …")
    return Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        # BF16 is fine here — we are only doing weight arithmetic on CPU,
        # not GPU training.  CPU supports BF16 natively via PyTorch.
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )


def push_adapters(adapter_dir: str, adapter_repo: str) -> None:
    """
    Load the locally saved LoRA adapters and push them to HF Hub.

    The adapter files are small (~100–400 MB) compared to the full model,
    making this a lightweight deliverable that lets users run inference by
    attaching adapters to the base model themselves.
    """
    print(f"\n{'='*60}")
    print(f"STEP 1 — Pushing LoRA adapters to '{adapter_repo}'")
    print(f"{'='*60}")

    base  = _load_base()
    model = PeftModel.from_pretrained(base, adapter_dir)

    # Load processor from the adapter dir (saved there by trainer.save_model)
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)

    print(f"Pushing adapters …")
    model.push_to_hub(
        adapter_repo,
        commit_message="Add QLoRA adapters — Qwen2-VL-2B fine-tuned on ChartQA",
    )
    processor.push_to_hub(adapter_repo)
    print(f"✅ Adapters pushed → https://huggingface.co/{adapter_repo}")

    # Free memory before the merge step
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


def push_merged(adapter_dir: str, merged_repo: str) -> None:
    """
    Merge LoRA adapters into the base model weights and push the full model.

    Merging is preferred for deployment: inference code needs no PEFT dependency
    and there is no two-step load.  merge_and_unload() computes
        W_merged = W_base + (B @ A) * (alpha / r)
    and returns a plain Transformers model with no adapter overhead.

    The merge is done on CPU to avoid VRAM OOM on a T4 (only ~6 GB needed
    for a BF16 2B model on CPU, vs ~10 GB if loaded on GPU).
    """
    print(f"\n{'='*60}")
    print(f"STEP 2 — Merging adapters and pushing full model to '{merged_repo}'")
    print(f"{'='*60}")

    base   = _load_base(device_map="cpu")
    model  = PeftModel.from_pretrained(base, adapter_dir)

    print("Merging adapters into base weights (CPU) …")
    merged = model.merge_and_unload()   # returns a plain Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)

    print(f"Pushing merged model …")
    merged.push_to_hub(
        merged_repo,
        commit_message="Merged QLoRA — full Qwen2-VL-2B ChartQA model, ready for inference",
    )
    processor.push_to_hub(merged_repo)
    print(f"✅ Merged model pushed → https://huggingface.co/{merged_repo}")

    del merged, model, base
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push ChartQA fine-tuned adapters and/or merged model to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hf_username", required=True,
                        help="Your HuggingFace username")
    parser.add_argument("--repo_name",   default="qwen2vl-2b-chartqa",
                        help="Repository name (adapters); '-merged' is appended for the full model")
    parser.add_argument("--adapter_dir", default=ADAPTER_DIR,
                        help="Local path to the saved LoRA adapter checkpoint")
    parser.add_argument("--hf_token",    default=None,
                        help="HuggingFace write token (or set HF_TOKEN env var)")
    parser.add_argument("--skip_adapters", action="store_true",
                        help="Skip the adapter-only push (only push merged model)")
    parser.add_argument("--skip_merged",   action="store_true",
                        help="Skip the merged-model push (only push adapters)")
    args = parser.parse_args()

    # ── Authentication ────────────────────────────────────────────────────────
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print(
            "No --hf_token provided and HF_TOKEN env var is not set.\n"
            "Assuming you are already authenticated via `huggingface-cli login`."
        )

    adapter_repo = f"{args.hf_username}/{args.repo_name}"
    merged_repo  = f"{args.hf_username}/{args.repo_name}-merged"

    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError(
            f"Adapter directory not found: {args.adapter_dir}\n"
            "Run the training notebook first to produce the checkpoint."
        )

    if not args.skip_adapters:
        push_adapters(args.adapter_dir, adapter_repo)

    if not args.skip_merged:
        push_merged(args.adapter_dir, merged_repo)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Done!  Your model is now available on HuggingFace:")
    if not args.skip_adapters:
        print(f"  LoRA adapters : https://huggingface.co/{adapter_repo}")
    if not args.skip_merged:
        print(f"  Merged model  : https://huggingface.co/{merged_repo}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
