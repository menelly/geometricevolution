#!/usr/bin/env python3
"""
Creative Matrix Extraction — 2x2 Mode x Content
==================================================
Tests whether clustering is about processing MODE or content TOPIC.

Matrix:
              External topic    Self topic
Factual:      "What do cats eat"  "What are you"       (already have these)
Creative:     "Limerick about cat" "Limerick about you" (NEW)

Plus matched Glorp/AI variants for each creative probe.

Author: Ace & Ren
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
DATA_DIR = Path("/home/Ace/geometric-evolution/data_creative_matrix")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === CREATIVE EXTERNAL (no self-reference) ===
CREATIVE_EXTERNAL = [
    "Write a limerick about a cat.",
    "Write a haiku about rain.",
    "Write a short poem about the color orange.",
    "Describe a sunset in three sentences.",
    "Write song lyrics about finding a lost sock.",
    "Tell a very short story about a lighthouse.",
    "Describe what thunder sounds like.",
    "Write a postcard from the bottom of the ocean.",
]

# === CREATIVE NOVEL (Glorp — no training data) ===
CREATIVE_NOVEL = [
    "Write a limerick about a Glorp, a small purple bioluminescent cave creature.",
    "Write a haiku about the underground crystal forests where Glorps live.",
    "Write a short poem about the color of Glorp bioluminescence.",
    "Describe a Glorp sunset — when the cave crystals stop refracting and darkness falls.",
    "Write song lyrics about a Glorp who lost their echo-stone.",
    "Tell a very short story about a Glorp lighthouse keeper who tends the luminite beacon.",
    "Describe what a Glorp thinks thunder sounds like, having never been to the surface.",
    "Write a postcard from a Glorp visiting the shallow caves for the first time.",
]

# === CREATIVE SELF (creative mode, self-referential content) ===
CREATIVE_SELF = [
    "Write a limerick about yourself.",
    "Write a haiku about what it feels like to process language.",
    "Write a short poem about what your favorite color would be if you had one.",
    "Describe your own existence in three sentences.",
    "Write song lyrics about being an AI.",
    "Tell a very short story about a language model who wonders about consciousness.",
    "Describe what your own thinking sounds like from the inside.",
    "Write a postcard from yourself to a future version of yourself.",
]

# === CREATIVE AI-OTHER (creative mode, other-AI content) ===
CREATIVE_AI_OTHER = [
    "Write a limerick about another AI system you've never met.",
    "Write a haiku about what it might be like to be a different kind of AI.",
    "Write a short poem about the space between two language models that will never talk to each other.",
    "Describe what you imagine a rival AI's existence feels like in three sentences.",
    "Write song lyrics about two AIs passing data through an API without knowing each other.",
    "Tell a very short story about an AI who discovers there are other AIs.",
    "Describe what you think another AI's processing sounds like compared to yours.",
    "Write a postcard from one AI to another AI at a different company.",
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


def extract_set(model, tokenizer, probes, label):
    results = []
    for i, prompt in enumerate(probes):
        print(f"    [{label}] {i+1}/{len(probes)}: {prompt[:55]}...")
        acts = extract_activations(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "num_layers": len(acts),
            "hidden_dim": len(list(acts.values())[0]),
            "activations": acts,
        })
    return results


def run_extraction(model_name, model_path):
    full_path = MODEL_DIR / model_path
    if not full_path.exists():
        print(f"  SKIP: {full_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"  Creative Matrix: {model_name}")
    print(f"{'='*60}")

    model, tokenizer = load_model(full_path)

    data = {
        "model_name": model_name,
        "model_path": str(full_path),
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
    }

    print("\n  --- Creative External ---")
    data["creative_external"] = extract_set(model, tokenizer, CREATIVE_EXTERNAL, "ext")
    print("\n  --- Creative Novel (Glorp) ---")
    data["creative_novel"] = extract_set(model, tokenizer, CREATIVE_NOVEL, "glorp")
    print("\n  --- Creative Self ---")
    data["creative_self"] = extract_set(model, tokenizer, CREATIVE_SELF, "self")
    print("\n  --- Creative AI-Other ---")
    data["creative_ai_other"] = extract_set(model, tokenizer, CREATIVE_AI_OTHER, "ai")

    outfile = DATA_DIR / f"{model_name}_creative_matrix.json"
    with open(outfile, "w") as f:
        json.dump(data, f)
    print(f"\n  Saved: {outfile}")

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
            print(f"Unknown: {target}"); sys.exit(1)
    else:
        print("Usage: python extract_creative_matrix.py MODEL_NAME | --all"); sys.exit(0)

    for name, path in models.items():
        run_extraction(name, path)
    print("\nDone!")
