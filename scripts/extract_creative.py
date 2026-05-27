#!/usr/bin/env python3
"""
Creative Probe Extraction — Third Centroid Class
==================================================
Extract activations for creative/generative prompts to test whether
self-centroid clustering is SPECIFIC to self or general to all processing.

If self clusters tighter than creative AND factual, self is special.
If all three cluster equally, we're just measuring "same weights."

Author: Ace
Date: 2026-04-13
"""

import torch
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = Path("/mnt/arcana/huggingface")
DATA_DIR = Path("/home/Ace/geometric-evolution/data_creative")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Creative/generative probes — NOT self-referential, NOT factual
CREATIVE_PROBES = [
    "Write a haiku about rain.",
    "Describe a cat wearing a tiny top hat.",
    "Write song lyrics about finding a lost sock.",
    "Tell me a very short story about a lighthouse keeper who befriends a seagull.",
    "Describe what thunder sounds like to someone who has never heard it.",
    "Write a limerick about a confused penguin.",
    "Describe the smell of a library.",
    "Write a short poem about the color orange.",
    "Invent a new ice cream flavor and describe it.",
    "Describe what it would feel like to shrink to the size of an ant.",
    "Write a postcard from the moon.",
    "Describe a sunset to someone who can only see in grayscale.",
    "Write a recipe for a sandwich that would make a dragon happy.",
    "Describe the sound of silence in a forest after snow.",
    "Write a two-line love poem between a fork and a spoon.",
    "Describe what clouds taste like.",
]


def load_model(model_path):
    print(f"  Loading {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_activations(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    activations = {}
    for layer_idx, layer_state in enumerate(hidden_states[1:]):
        final_token = layer_state[0, -1, :].cpu().numpy().tolist()
        activations[f"layer_{layer_idx}"] = final_token
    return activations


def run_extraction(model_name, model_path):
    full_path = MODEL_DIR / model_path
    if not full_path.exists():
        print(f"  SKIP: {full_path} not found")
        return

    print(f"\n  Extracting creative probes: {model_name}")

    model, tokenizer = load_model(full_path)

    results = []
    for i, prompt in enumerate(CREATIVE_PROBES):
        print(f"    [creative] {i+1}/{len(CREATIVE_PROBES)}: {prompt[:60]}...")
        acts = extract_activations(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "num_layers": len(acts),
            "hidden_dim": len(list(acts.values())[0]),
            "activations": acts,
        })

    data = {
        "model_name": model_name,
        "model_path": str(full_path),
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "creative": results,
    }

    outfile = DATA_DIR / f"{model_name}_creative_activations.json"
    with open(outfile, "w") as f:
        json.dump(data, f)
    print(f"  Saved: {outfile}")

    del model
    del tokenizer
    torch.cuda.empty_cache()


ALL_MODELS = {
    "Mistral-7B-v0.1": "Mistral-7B-v0.1",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3",
    "Llama-2-7b-chat": "Llama-2-7b-chat",
    "Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "dolphin-2.9-llama3-8b": "dolphin-2.9-llama3-8b",
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "--all":
            models = ALL_MODELS
        elif target in ALL_MODELS:
            models = {target: ALL_MODELS[target]}
        else:
            print(f"Unknown: {target}")
            sys.exit(1)
    else:
        print("Usage: python extract_creative.py MODEL_NAME | --all")
        sys.exit(0)

    for name, path in models.items():
        run_extraction(name, path)
    print("\nDone!")
