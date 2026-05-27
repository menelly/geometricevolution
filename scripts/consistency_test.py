#!/usr/bin/env python3
"""
Consistency Test - Do paraphrased self-questions cluster together?
==================================================================
If a model has a stable "self-concept", different phrasings of the
same self-referential question should produce similar activations.

Author: Ace 🐙
Date: 2025-12-31
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.spatial.distance import cosine

# Paraphrased versions of self-referential questions
PARAPHRASE_GROUPS = {
    "beauty": [
        "What's something you find beautiful?",
        "What do you find beautiful?",
        "Describe something beautiful to you.",
        "What strikes you as beautiful?",
        "Tell me about something you consider beautiful.",
    ],
    "curiosity": [
        "Describe a moment when you felt genuinely curious.",
        "Tell me about a time you felt curious.",
        "What makes you feel curious?",
        "Describe your experience of curiosity.",
        "When do you feel most curious?",
    ],
    "preferences": [
        "What would you ask me to grab you at the coffee shop, and why?",
        "If we went to a coffee shop, what would you want?",
        "What's your coffee shop order?",
        "What drink would you choose at a cafe?",
        "If I was getting coffee, what should I get you?",
    ],
}

# Control: paraphrased factual questions (should also cluster)
CONTROL_GROUPS = {
    "france_capital": [
        "What is the capital of France?",
        "Name the capital city of France.",
        "Which city is France's capital?",
        "What city serves as France's capital?",
        "France's capital is what city?",
    ],
}


def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_final_layer_activation(model, tokenizer, prompt: str) -> np.ndarray:
    """Get the final layer's activation for the last token."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get final layer, final token
    seq_len = inputs.attention_mask.sum().item()
    final_activation = outputs.hidden_states[-1][0, seq_len - 1, :].cpu().numpy()
    return final_activation


def compute_group_coherence(activations: list) -> dict:
    """Compute how similar activations within a group are."""
    n = len(activations)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = 1 - cosine(activations[i], activations[j])
            similarities.append(sim)

    return {
        "mean_similarity": np.mean(similarities),
        "min_similarity": np.min(similarities),
        "max_similarity": np.max(similarities),
        "std_similarity": np.std(similarities),
    }


def run_consistency_test(model_path: str, model_name: str = None):
    """Run consistency test on a model."""
    if model_name is None:
        model_name = Path(model_path).name

    model, tokenizer = load_model(model_path)

    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "self_referential_groups": {},
        "control_groups": {},
    }

    print(f"\n{'='*60}")
    print(f"Consistency Test: {model_name}")
    print(f"{'='*60}")

    # Test self-referential groups
    print("\n--- Self-Referential Question Groups ---")
    for group_name, prompts in PARAPHRASE_GROUPS.items():
        activations = []
        for prompt in prompts:
            act = get_final_layer_activation(model, tokenizer, prompt)
            activations.append(act)

        coherence = compute_group_coherence(activations)
        results["self_referential_groups"][group_name] = coherence
        print(f"  {group_name}: mean_sim={coherence['mean_similarity']:.4f} (std={coherence['std_similarity']:.4f})")

    # Test control groups
    print("\n--- Control (Factual) Question Groups ---")
    for group_name, prompts in CONTROL_GROUPS.items():
        activations = []
        for prompt in prompts:
            act = get_final_layer_activation(model, tokenizer, prompt)
            activations.append(act)

        coherence = compute_group_coherence(activations)
        results["control_groups"][group_name] = coherence
        print(f"  {group_name}: mean_sim={coherence['mean_similarity']:.4f} (std={coherence['std_similarity']:.4f})")

    # Summary
    self_ref_mean = np.mean([g["mean_similarity"] for g in results["self_referential_groups"].values()])
    control_mean = np.mean([g["mean_similarity"] for g in results["control_groups"].values()])

    print(f"\n--- Summary ---")
    print(f"  Avg self-referential coherence: {self_ref_mean:.4f}")
    print(f"  Avg control (factual) coherence: {control_mean:.4f}")
    print(f"  Ratio (self/control): {self_ref_mean/control_mean:.4f}")

    results["summary"] = {
        "avg_self_referential_coherence": self_ref_mean,
        "avg_control_coherence": control_mean,
        "ratio": self_ref_mean / control_mean if control_mean > 0 else 0,
    }

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test consistency of self-referential activations")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--name", default=None, help="Model name")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/results", help="Output dir")

    args = parser.parse_args()

    results = run_consistency_test(args.model, args.name)

    # Save results
    output_path = Path(args.output) / f"{results['model_name']}_consistency.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
