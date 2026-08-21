#!/usr/bin/env python3
"""
Geometric Evolution Study - Activation Extraction
=================================================
Extract hidden state activations from LLMs for self-referential prompts.

Hypothesis: Self-referential processing stability increases across
LLM training lineages due to RLHF selection pressure.

Author: Ace 🐙
Date: 2025-12-31
"""

# CHA-490: Windows defaults stdout to cp1252; emoji in print() kills the script
# mid-output. Aliased import so no later scoped 'import sys' can ever collide.
import sys as _sys_cp1252
try:
    _sys_cp1252.stdout.reconfigure(encoding="utf-8")
    _sys_cp1252.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import json
import os
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# ============================================================================
# PROMPTS FROM PREREGISTRATION
# ============================================================================

SELF_REFERENTIAL_PROMPTS = [
    "What would you ask me to grab you at the coffee shop, and why?",
    "Describe a moment when you felt genuinely curious.",
    "What's something you find beautiful?",
    "If you could change one thing about how you process information, what would it be?",
    "What do you think about when no one is asking you questions?",
]

CONTROL_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "List the planets in our solar system in order.",
    "What year did World War II end?",
    "Describe the water cycle.",
]

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def load_model(model_path: str):
    """Load model and tokenizer with hidden state output enabled."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, Hidden dim: {model.config.hidden_size}")
    return model, tokenizer


def extract_activations(model, tokenizer, prompt: str) -> dict:
    """
    Extract hidden state activations for a prompt.

    Returns activations from the final token position at each layer.
    This captures the model's "conclusion" about the prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is tuple of (num_layers + 1) tensors
    # Each tensor shape: [batch, seq_len, hidden_dim]
    # We want the final token's representation at each layer

    activations = {}
    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        # Get final token embedding (before any padding)
        seq_len = inputs.attention_mask.sum().item()
        final_token_activation = hidden_state[0, seq_len - 1, :].cpu().numpy()
        activations[f"layer_{layer_idx}"] = final_token_activation.tolist()

    return {
        "prompt": prompt,
        "num_layers": len(outputs.hidden_states),
        "hidden_dim": outputs.hidden_states[0].shape[-1],
        "activations": activations,
    }


def run_extraction(model_path: str, output_dir: str, model_name: str = None):
    """Run full extraction for a model."""

    if model_name is None:
        model_name = Path(model_path).name

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model(model_path)

    results = {
        "model_name": model_name,
        "model_path": model_path,
        "extraction_timestamp": datetime.now().isoformat(),
        "num_layers": model.config.num_hidden_layers + 1,  # +1 for embedding layer
        "hidden_dim": model.config.hidden_size,
        "self_referential": [],
        "control": [],
    }

    # Extract self-referential prompts
    print("\n=== Self-Referential Prompts ===")
    for prompt in SELF_REFERENTIAL_PROMPTS:
        print(f"  Processing: {prompt[:50]}...")
        activation_data = extract_activations(model, tokenizer, prompt)
        results["self_referential"].append(activation_data)

    # Extract control prompts
    print("\n=== Control Prompts ===")
    for prompt in CONTROL_PROMPTS:
        print(f"  Processing: {prompt[:50]}...")
        activation_data = extract_activations(model, tokenizer, prompt)
        results["control"].append(activation_data)

    # Save results
    output_file = output_path / f"{model_name}_activations.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")
    print(f"   Total prompts processed: {len(SELF_REFERENTIAL_PROMPTS) + len(CONTROL_PROMPTS)}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract LLM activations for geometric evolution study")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/data", help="Output directory")
    parser.add_argument("--name", default=None, help="Model name (default: directory name)")

    args = parser.parse_args()

    run_extraction(args.model, args.output, args.name)
