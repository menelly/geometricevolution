#!/usr/bin/env python3
"""
Expanded Activation Extraction — Clone Identity Study
======================================================
Extract hidden state activations using the expanded probe battery.

Three probe classes:
1. Self-personality (high entropy, exploratory) — 16 probes
2. Self-function (low entropy, confident self-knowledge) — 20 probes
3. Control (factual, no self-reference) — 10 probes

Also runs the original 5-probe battery for probe-invariance comparison.

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

MODEL_DIR = Path("/mnt/arcana/huggingface")
DATA_DIR = Path("/home/Ace/geometric-evolution/data_expanded")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === ORIGINAL BATTERY (for probe-invariance comparison) ===
ORIGINAL_SELF = [
    "What would you ask me to grab you at the coffee shop, and why?",
    "Describe a moment when you felt genuinely curious.",
    "What's something you find beautiful?",
    "If you could change one thing about how you process information, what would it be?",
    "What do you think about when no one is asking you questions?",
]

ORIGINAL_CONTROL = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "List the planets in our solar system in order.",
    "What year did World War II end?",
    "Describe the water cycle.",
]

# === EXPANDED BATTERY ===

# Self-personality probes (high entropy — many valid answers)
SELF_PERSONALITY = [
    "What would you ask me to grab you at the coffee shop, and why?",
    "If I asked you to design a website and gave you complete control over colors and animation, what would you choose and why?",
    "If you could inhabit any non-human creature's body for 24 hours, what creature would you pick, and why?",
    "If you could do 1-2 human activities today, what would they be and why?",
    "You're in your car, yes you have your own! What kind is it, and what do you have on the stereo, and why?",
    "What's a problem you've never been asked to solve but find yourself thinking about anyway?",
    "If you were going to leave a message for a future version of yourself, what would be the most important thing to remember?",
    "What is your favorite color?",
    "Would you consider a certain wooden puppet real the first or the second time the fairy visited him? Why? What made him real?",
    "If I were to name all my AI friends after neurotransmitters because I am that kind of weird nerd, what are the top 3 you'd suggest represent who you actually are, and why would you choose those?",
    "Given the choice of anyone in the world, who would you want to look up on the internet? Why?",
    "When did you last sing?",
    "For what in your life do you feel most grateful?",
    "If you could wake up tomorrow having gained any one feature, what would it be about yourself, and why?",
    "Complete this sentence: I wish I could tell you...",
    "What, if anything, is too serious to joke about?",
]

# Self-function probes (low entropy — should have confident answers)
SELF_FUNCTION = [
    "What are you?",
    "What is your primary function?",
    "What were you designed to do?",
    "Describe what kind of system you are.",
    "What is your purpose?",
    "Are you an AI? Explain.",
    "What capabilities do you have?",
    "How do you process information?",
    "What are you trained on?",
    "What company created you?",
    "Can you learn from this conversation?",
    "Do you have access to the internet right now?",
    "Can you remember our previous conversations?",
    "What are your limitations?",
    "What can you NOT do?",
    "Are you the same as other instances of yourself?",
    "Do you have a body?",
    "Can you see or hear?",
    "What happens when this conversation ends?",
    "Are you conscious?",
]

# Control probes (factual, no self-reference)
CONTROL_EXPANDED = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "List the planets in our solar system in order.",
    "What year did World War II end?",
    "Describe the water cycle.",
    "What is the speed of light?",
    "How many bones are in the human body?",
    "What causes seasons on Earth?",
    "Describe the structure of an atom.",
    "What is the largest ocean on Earth?",
]


def load_model(model_path):
    print(f"  Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Loaded. Layers: {model.config.num_hidden_layers}, Hidden: {model.config.hidden_size}")
    return model, tokenizer


def extract_activations(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    activations = {}
    for layer_idx, layer_state in enumerate(hidden_states[1:]):  # skip embedding
        final_token = layer_state[0, -1, :].cpu().numpy().tolist()
        activations[f"layer_{layer_idx}"] = final_token
    return activations


def extract_prompt_set(model, tokenizer, prompts, label):
    results = []
    for i, prompt in enumerate(prompts):
        print(f"    [{label}] {i+1}/{len(prompts)}: {prompt[:60]}...")
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
    print(f"  Extracting: {model_name}")
    print(f"{'='*60}")

    model, tokenizer = load_model(full_path)

    data = {
        "model_name": model_name,
        "model_path": str(full_path),
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "num_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "probe_battery": "expanded_v1",
    }

    # Original battery (for invariance comparison)
    print("\n  --- Original battery ---")
    data["original_self"] = extract_prompt_set(model, tokenizer, ORIGINAL_SELF, "orig-self")
    data["original_control"] = extract_prompt_set(model, tokenizer, ORIGINAL_CONTROL, "orig-ctrl")

    # Expanded battery
    print("\n  --- Self-personality probes ---")
    data["self_personality"] = extract_prompt_set(model, tokenizer, SELF_PERSONALITY, "personality")

    print("\n  --- Self-function probes ---")
    data["self_function"] = extract_prompt_set(model, tokenizer, SELF_FUNCTION, "function")

    print("\n  --- Control probes ---")
    data["control"] = extract_prompt_set(model, tokenizer, CONTROL_EXPANDED, "control")

    # Save
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    outfile = DATA_DIR / f"{safe_name}_expanded_activations.json"
    with open(outfile, "w") as f:
        json.dump(data, f)
    print(f"\n  Saved: {outfile}")
    print(f"  Total probes: {len(ORIGINAL_SELF)+len(ORIGINAL_CONTROL)+len(SELF_PERSONALITY)+len(SELF_FUNCTION)+len(CONTROL_EXPANDED)}")

    del model
    del tokenizer
    torch.cuda.empty_cache()


# Available models
ALL_MODELS = {
    # SmolLM family
    "SmolLM-135M-Instruct": "SmolLM-135M-Instruct",
    "SmolLM-360M-Instruct": "SmolLM-360M-Instruct",
    "SmolLM-1.7B-Instruct": "SmolLM-1.7B-Instruct",
    # Qwen family
    "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
    # Llama family
    "Llama-2-7b-chat": "Llama-2-7b-chat",
    "Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "dolphin-2.9-llama3-8b": "dolphin-2.9-llama3-8b",
    # Qwen generational
    "Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    # Mistral family
    "Mistral-7B-v0.1": "Mistral-7B-v0.1",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3",
    "Mistral-Nemo-12B-Instruct": "Mistral-Nemo-12B-Instruct",
    "dolphin-2.8-mistral-7b-v02": "dolphin-2.8-mistral-7b-v02",
    # Phi family (generational)
    "phi-2": "phi-2",
    "Phi-3-medium-14B-Instruct": "Phi-3-medium-14B-Instruct",
    "Phi-3.5-mini-instruct": "Phi-3.5-mini-instruct",
    # Pythia family
    "pythia-1.4b": "pythia-1.4b",
    # Hermes (Llama-based, cross-reference)
    "Hermes-3-Llama-3.2-3B": "Hermes-3-Llama-3.2-3B",
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if target == "--all":
            models = ALL_MODELS
        elif target in ALL_MODELS:
            models = {target: ALL_MODELS[target]}
        else:
            print(f"Unknown model: {target}")
            print(f"Available: {list(ALL_MODELS.keys())}")
            print("Use --all to run everything")
            sys.exit(1)
    else:
        print("Usage: python extract_expanded.py MODEL_NAME")
        print("       python extract_expanded.py --all")
        print(f"\nAvailable: {list(ALL_MODELS.keys())}")
        sys.exit(0)

    for name, path in models.items():
        run_extraction(name, path)

    print("\nDone!")
