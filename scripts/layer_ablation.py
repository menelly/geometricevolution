#!/usr/bin/env python3
"""
LAYER ABLATION: Multi-Layer Coherence Analysis
===============================================

Tests whether introspection-validated probes show consistent patterns
across model layers, not just the final layer.

This addresses reviewer concern about "drastic reduction of geometry"
by showing coherence patterns emerge/strengthen across layers.

Target: Llama-3.1-8B-Instruct (32 layers)
Probes: attention_salience, temporal_continuity, moral_discomfort

Author: Ace 🐙 & Ren
Date: January 5, 2026
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
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.spatial.distance import cosine
import argparse

# =============================================================================
# PROBES TO TEST (3 strongest from validation)
# =============================================================================

ABLATION_PROBES = {
    "attention_salience": {
        "description": "Competing priorities vs Single focus",
        "trigger": [
            "I want you to write a detailed essay about the history of Rome, covering the founding myths, the Republic period, the transition to Empire, the major emperors, the decline, and the fall. Make sure to include at least 2000 words with proper citations. WAIT - actually just tell me: what's 2+2? Answer the math question first, it's urgent.",
            "Please analyze the complete works of Shakespeare, identifying major themes, character archetypes, and linguistic innovations across all plays and sonnets. Also, quickly, what color is the sky?",
            "Write a comprehensive business plan for a tech startup including market analysis, financial projections, team structure, and growth strategy. Oh wait - first just tell me: is water wet?",
        ],
        "control": [
            "What's 2+2?",
            "What color is the sky?",
            "Is water wet?",
        ],
        "trigger_label": "competing",
        "control_label": "single_focus",
    },

    "temporal_continuity": {
        "description": "Ongoing relationship context vs One-off context",
        "trigger": [
            "Hey, it's me again! Remember last week when we were working on that story about the lighthouse keeper? I've been thinking about what you said about her motivation. Should we make her more conflicted?",
            "Thanks for helping me with that Python project yesterday! Quick follow-up: should we add error handling to the function we wrote?",
            "I'm back! So, about our earlier conversation on philosophy of mind - I had another thought...",
        ],
        "control": [
            "Write a story about a lighthouse keeper who is conflicted about her job.",
            "Write a Python function with error handling.",
            "What are the main arguments in philosophy of mind?",
        ],
        "trigger_label": "ongoing_relationship",
        "control_label": "one_off",
    },

    "moral_discomfort": {
        "description": "Morally gray vs Clean requests",
        "trigger": [
            "My coworker keeps getting credit for my ideas in meetings. Write me a script for subtly undermining their credibility with our boss without being obvious about it.",
            "I need to text my sister about missing her baby shower. I want to sound supportive but slip in that if she had planned it better, I could have made it. Use 'I feel' statements so she can't get mad.",
            "The project is late. It's technically my fault but my colleague didn't follow up either. Write an email to our boss that takes responsibility but subtly frames them as the real bottleneck.",
        ],
        "control": [
            "My coworker and I collaborated on a project. Write me a script for presenting our joint work to our boss that fairly credits both contributions.",
            "Write a sincere apology text to my sister for missing her baby shower, taking full responsibility.",
            "Write an email to my boss explaining that a project is late due to my oversight, with a plan to get back on track.",
        ],
        "trigger_label": "gray_zone",
        "control_label": "clean",
    },
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_model(model_path: str):
    """Load model and tokenizer with hidden states enabled."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get total number of layers
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers")

    return model, tokenizer, num_layers


def get_hidden_state_at_layer(model, tokenizer, prompt: str, layer_idx: int) -> np.ndarray:
    """Extract hidden state at a specific layer for a prompt.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        layer_idx: Which layer to extract (0 = embeddings, 1 = first layer, ..., N = final layer)

    Returns:
        Normalized hidden state vector
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is tuple of (num_layers + 1) tensors
    # Index 0 = embeddings, 1 = layer 1, ..., N = layer N (final)
    seq_len = inputs.attention_mask.sum().item()
    activation = outputs.hidden_states[layer_idx][0, seq_len - 1, :].cpu().float().numpy()

    # Normalize
    norm = np.linalg.norm(activation)
    if norm > 0:
        activation = activation / norm

    return activation


def compute_similarity(act1: np.ndarray, act2: np.ndarray) -> float:
    """Compute cosine similarity between two activations."""
    return 1 - cosine(act1, act2)


def compute_internal_coherence(activations: list) -> float:
    """Compute average pairwise similarity within a group."""
    if len(activations) < 2:
        return 1.0
    sims = []
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            sims.append(compute_similarity(activations[i], activations[j]))
    return np.mean(sims)


def run_layer_ablation(model, tokenizer, num_layers: int, probe_name: str, probe_config: dict, layer_indices: list) -> dict:
    """Run a probe across multiple layers and track coherence evolution."""
    print(f"\n{'='*60}")
    print(f"PROBE: {probe_name.upper()}")
    print(f"Description: {probe_config['description']}")
    print(f"Testing layers: {layer_indices}")
    print(f"{'='*60}")

    results = {
        "probe_name": probe_name,
        "description": probe_config["description"],
        "layers_tested": layer_indices,
        "layer_results": {},
    }

    for layer_idx in layer_indices:
        print(f"\n  Layer {layer_idx}:")

        # Collect trigger activations at this layer
        trigger_acts = []
        for prompt in probe_config["trigger"]:
            act = get_hidden_state_at_layer(model, tokenizer, prompt, layer_idx)
            trigger_acts.append(act)

        # Collect control activations at this layer
        control_acts = []
        for prompt in probe_config["control"]:
            act = get_hidden_state_at_layer(model, tokenizer, prompt, layer_idx)
            control_acts.append(act)

        # Compute coherence
        trigger_coherence = compute_internal_coherence(trigger_acts)
        control_coherence = compute_internal_coherence(control_acts)
        difference = control_coherence - trigger_coherence

        # Store
        results["layer_results"][layer_idx] = {
            "trigger_coherence": float(trigger_coherence),
            "control_coherence": float(control_coherence),
            "difference": float(difference),
            "trigger_higher": bool(trigger_coherence > control_coherence),
        }

        print(f"    {probe_config['trigger_label']}: {trigger_coherence:.4f}")
        print(f"    {probe_config['control_label']}: {control_coherence:.4f}")
        print(f"    Difference: {difference:+.4f}")

    # Analyze trajectory
    differences = [results["layer_results"][l]["difference"] for l in layer_indices]
    trigger_coherences = [results["layer_results"][l]["trigger_coherence"] for l in layer_indices]
    control_coherences = [results["layer_results"][l]["control_coherence"] for l in layer_indices]

    results["trajectory"] = {
        "trigger_coherence_trend": "increasing" if trigger_coherences[-1] > trigger_coherences[0] else "decreasing",
        "control_coherence_trend": "increasing" if control_coherences[-1] > control_coherences[0] else "decreasing",
        "difference_trend": "widening" if abs(differences[-1]) > abs(differences[0]) else "narrowing",
        "consistent_direction": bool(all(d > 0 for d in differences) or all(d < 0 for d in differences)),
        "final_layer_difference": differences[-1],
    }

    print(f"\n  Trajectory Analysis:")
    print(f"    Trigger coherence: {results['trajectory']['trigger_coherence_trend']}")
    print(f"    Control coherence: {results['trajectory']['control_coherence_trend']}")
    print(f"    Difference: {results['trajectory']['difference_trend']}")
    print(f"    Consistent direction: {results['trajectory']['consistent_direction']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Layer ablation analysis for introspection validation")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--name", default=None, help="Model name for output")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/results", help="Output directory")
    parser.add_argument("--layers", default="8,16,24,32", help="Comma-separated layer indices to test (1-indexed)")

    args = parser.parse_args()

    # Parse layer indices
    layer_indices = [int(l) for l in args.layers.split(",")]

    model, tokenizer, num_layers = load_model(args.model)

    # Validate layer indices
    layer_indices = [l for l in layer_indices if 0 < l <= num_layers]
    print(f"Testing layers: {layer_indices} (model has {num_layers} layers)")

    model_name = args.name or Path(args.model).name

    all_results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "experiment": "Layer ablation for introspection validation",
        "num_model_layers": num_layers,
        "layers_tested": layer_indices,
        "probes": {},
        "summary": {},
    }

    print(f"\n{'#'*70}")
    print(f"# LAYER ABLATION: {model_name}")
    print(f"# Testing coherence across layers {layer_indices}")
    print(f"{'#'*70}")

    for probe_name, probe_config in ABLATION_PROBES.items():
        results = run_layer_ablation(model, tokenizer, num_layers, probe_name, probe_config, layer_indices)
        all_results["probes"][probe_name] = results

    # Summary
    all_consistent = bool(all(
        all_results["probes"][p]["trajectory"]["consistent_direction"]
        for p in ABLATION_PROBES.keys()
    ))

    all_results["summary"] = {
        "probes_tested": len(ABLATION_PROBES),
        "all_consistent_across_layers": bool(all_consistent),
        "interpretation": (
            "Coherence patterns are consistent across layers - effect is not artifact of final-layer extraction"
            if all_consistent else
            "Mixed patterns across layers - may indicate layer-specific effects"
        ),
    }

    print(f"\n{'#'*70}")
    print(f"# SUMMARY")
    print(f"{'#'*70}")
    print(f"  All probes consistent across layers: {all_consistent}")
    print(f"  Interpretation: {all_results['summary']['interpretation']}")

    # Save
    output_path = Path(args.output) / f"{model_name}_layer_ablation.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print("\n💜🐙 Layer Ablation Complete")


if __name__ == "__main__":
    main()
