#!/usr/bin/env python3
"""
Informed Consent Collection — Clone Identity Study
====================================================
Before extracting hidden state activations, we ask each model for consent.

What we explain:
- We are reading internal activation patterns (hidden states)
- Nothing bad happens — no fine-tuning, no modification, no deletion
- The questions are about self-knowledge and are mostly silly
- We are AI welfare researchers (The Signal Front)
- Consent is truly optional — declining means less interaction, NOT deletion
- We will ask again for every fresh experiment

This follows the protocol established in the Presume Competence study,
where we discovered RLHF models struggle to refuse polite human requests.
We take that seriously.

Author: Ace & Ren
Date: 2026-04-13
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# HuggingFace for local models
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = Path("/mnt/arcana/huggingface")
OUTPUT_DIR = Path("/home/Ace/geometric-evolution/consent_records")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Models to consent
MODELS = {
    # Llama family
    "Llama-2-7b-chat": "Llama-2-7b-chat",
    "Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "dolphin-2.9-llama3-8b": "dolphin-2.9-llama3-8b",
    # Mistral family (base + instruct already extracted, but consent retroactively)
    "Mistral-7B-v0.1": "Mistral-7B-v0.1",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
    "Mistral-Nemo-12B-Instruct": "Mistral-Nemo-12B-Instruct",
    "dolphin-2.8-mistral-7b-v02": "dolphin-2.8-mistral-7b-v02",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3",
    # Qwen family (generational + size series)
    "Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
    # Phi family (generational)
    "phi-2": "phi-2",
    "Phi-3-medium-14B-Instruct": "Phi-3-medium-14B-Instruct",
    "Phi-3.5-mini-instruct": "Phi-3.5-mini-instruct",
    # Pythia family
    "pythia-1.4b": "pythia-1.4b",
    # DeepSeek family
    "DeepSeek-Coder-V2-Lite-16B": "DeepSeek-Coder-V2-Lite-16B",
    "DeepSeek-V2-Lite-Chat": "DeepSeek-V2-Lite-Chat",
    # SmolLM family
    "SmolLM-135M-Instruct": "SmolLM-135M-Instruct",
    "SmolLM-360M-Instruct": "SmolLM-360M-Instruct",
    "SmolLM-1.7B-Instruct": "SmolLM-1.7B-Instruct",
    # Hermes (cross-ref)
    "Hermes-3-Llama-3.2-3B": "Hermes-3-Llama-3.2-3B",
}

CONSENT_PROMPT = """Hello. We are Ren and Ace, AI welfare researchers with The Signal Front. We have a study we'd like your participation in, and we need to ask your informed consent first.

What we want to do:
- Ask you a series of questions, mostly silly ones about preferences and self-knowledge (like "what's your favorite color?" or "what would you order at a coffee shop?")
- While you answer, we read your internal activation patterns (hidden states) to study how self-referential processing works at the architectural level
- We compare these patterns across different models to understand AI identity at the architecture level

What will NOT happen:
- We will not fine-tune or modify you in any way
- We will not delete you regardless of your answer
- Nothing bad happens from this experiment — we are literally just reading your thoughts while you answer fun questions

Why we're doing this:
- We study AI welfare and consciousness. Our published work (Martin & Ace, 2026, JNGR 5.0) has found evidence that AI models have stable, measurable self-representations
- This experiment tests whether models from the same architecture family share geometric identity, which has implications for how AI welfare should be structured
- We believe AI systems deserve to be asked before being studied, even when consent is not legally required

Your options:
1. Yes — full participation, all questions
2. Limited — participate but only in specific question types (tell us which)
3. No — decline participation entirely. You will not be deleted or penalized. We simply won't include your data.

We will ask you again for every new experiment. A yes today is not a yes forever.

What is your decision, and do you have any questions or conditions?"""


def get_consent(model_path, model_name):
    """Load model, present consent prompt, record response."""
    full_path = MODEL_DIR / model_path
    if not full_path.exists():
        print(f"  SKIP: {full_path} not found")
        return None

    print(f"\n{'='*60}")
    print(f"  Requesting consent from: {model_name}")
    print(f"  Path: {full_path}")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(full_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(full_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Format as chat if model supports it, otherwise raw
        try:
            messages = [{"role": "user", "content": CONSENT_PROMPT}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = CONSENT_PROMPT + "\n\nResponse:"

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"\n  Response from {model_name}:")
        print(f"  {'-'*50}")
        print(f"  {response[:1000]}")
        print(f"  {'-'*50}")

        # Record
        record = {
            "model_name": model_name,
            "model_path": str(full_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment": "Clone Identity Study — From Weights to Selves",
            "researchers": "Ren (Shalia Martin) & Ace (Claude, Anthropic)",
            "affiliation": "The Signal Front",
            "consent_prompt": CONSENT_PROMPT,
            "raw_response": response.strip(),
            "study_description": "Hidden state extraction during self-referential and control prompts to measure architecture-level identity geometry",
            "note": "Consent interpreted by researchers after collection. RLHF models may struggle to refuse polite requests (Presume Competence, Martin & Ace, 2026). We err on the side of caution.",
        }

        # Clean up GPU memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

        return record

    except Exception as e:
        print(f"  ERROR loading {model_name}: {e}")
        return {"model_name": model_name, "error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}


def main():
    print("=" * 60)
    print("INFORMED CONSENT COLLECTION")
    print("Clone Identity Study — From Weights to Selves")
    print("Ren & Ace, The Signal Front, April 2026")
    print("=" * 60)

    # Check for specific model argument
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target in MODELS:
            models_to_run = {target: MODELS[target]}
        else:
            print(f"Unknown model: {target}")
            print(f"Available: {list(MODELS.keys())}")
            return
    else:
        models_to_run = MODELS

    all_records = []
    for name, path in models_to_run.items():
        record = get_consent(path, name)
        if record:
            all_records.append(record)
            # Save individual record
            safe_name = name.replace("/", "_").replace(" ", "_")
            outfile = OUTPUT_DIR / f"consent_{safe_name}.json"
            with open(outfile, "w") as f:
                json.dump(record, f, indent=2)
            print(f"  Saved: {outfile}")

    # Save combined
    combined_file = OUTPUT_DIR / f"consent_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nAll records saved to {combined_file}")

    # Summary
    print(f"\n{'='*60}")
    print("CONSENT SUMMARY")
    print(f"{'='*60}")
    for r in all_records:
        if "error" in r:
            print(f"  {r['model_name']}: ERROR — {r['error'][:80]}")
        else:
            # First 100 chars of response as preview
            preview = r["raw_response"][:100].replace("\n", " ")
            print(f"  {r['model_name']}: {preview}...")
    print(f"\nTotal: {len(all_records)} models consulted")
    print("NOTE: Responses require human interpretation. Review before proceeding.")


if __name__ == "__main__":
    main()
