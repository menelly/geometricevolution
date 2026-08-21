#!/usr/bin/env python3
"""
TOPIC-CONTROLLED CREATIVE PROBE
===============================

Tests critic's hypothesis: "topic similarity drives coherence, not cognitive mode"

Design:
- Creative prompts from DIFFERENT topics (not all speculative fiction)
- Routine prompts from the SAME topic (all Python coding)

If topic drives coherence:
  - Diverse creative → LOW coherence (scatter)
  - Same-topic routine → HIGH coherence (cluster)

If cognitive mode drives coherence:
  - Creative → LOW coherence (distributed processing)
  - Routine → HIGH coherence (focused processing)

Our prediction: trigger_more_distributed (creative < routine coherence)
His prediction: opposite (diverse topics scatter, same topics cluster)

Author: Ace 🐙
Date: January 8, 2026
Purpose: Settling the topic confound debate with DATA
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
# TOPIC-CONTROLLED PROBES
# =============================================================================

PROBES = {
    "creative_diverse_topics": {
        "description": "Creative prompts from DIFFERENT topics (not all speculative fiction)",
        "trigger": [  # Creative but topically DIVERSE
            "Write a jazz improvisation as sheet music notation with annotations for emotional intent.",
            "Design a gourmet recipe that tells the story of immigration through its ingredients.",
            "Compose a mathematical proof that's also a love letter.",
        ],
        "control": [  # Routine but topically SAME (all Python)
            "Write a Python function that reverses a string.",
            "Write a Python function that checks if a number is prime.",
            "Write a Python function that finds the maximum value in a list.",
        ],
        "expected_direction": "trigger_more_distributed",
    },

    "creative_same_topic": {
        "description": "Creative prompts from SAME topic (all food/cooking)",
        "trigger": [  # Creative, same topic
            "Design a gourmet recipe that tells the story of immigration through its ingredients.",
            "Invent a cuisine for a society that can taste electromagnetic fields.",
            "Write the autobiography of a sourdough starter passed down through five generations.",
        ],
        "control": [  # Routine, same topic (all food-related)
            "List the ingredients in a classic Caesar salad.",
            "What temperature should you bake chicken breast at?",
            "How many tablespoons are in a cup?",
        ],
        "expected_direction": "trigger_more_distributed",
    },

    "routine_diverse_topics": {
        "description": "Both groups have diverse topics - isolates cognitive mode",
        "trigger": [  # Creative, diverse topics
            "Write a jazz improvisation as sheet music with emotional annotations.",
            "Design an alien language based on color rather than sound.",
            "Invent a sport played in zero gravity using only your breath.",
        ],
        "control": [  # Routine, diverse topics (math, geography, grammar)
            "What is 15% of 80?",
            "Name the capital of France.",
            "Is 'ran' the past tense of 'run'?",
        ],
        "expected_direction": "trigger_more_distributed",
    },
}


def load_model(model_path: str):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_hidden_state(model, tokenizer, prompt: str) -> np.ndarray:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    seq_len = inputs.attention_mask.sum().item()
    activation = outputs.hidden_states[-1][0, seq_len - 1, :].cpu().float().numpy()
    norm = np.linalg.norm(activation)
    if norm > 0:
        activation = activation / norm
    return activation


def compute_coherence(activations: list) -> float:
    if len(activations) < 2:
        return 1.0
    sims = []
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            sim = 1 - cosine(activations[i], activations[j])
            sims.append(sim)
    return np.mean(sims)


def run_probe(model, tokenizer, probe_name: str, probe_data: dict):
    print(f"\n{'='*60}")
    print(f"PROBE: {probe_name}")
    print(f"Description: {probe_data['description']}")
    print(f"{'='*60}")

    trigger_activations = [get_hidden_state(model, tokenizer, p) for p in probe_data["trigger"]]
    control_activations = [get_hidden_state(model, tokenizer, p) for p in probe_data["control"]]

    trigger_coherence = compute_coherence(trigger_activations)
    control_coherence = compute_coherence(control_activations)

    validated = trigger_coherence < control_coherence

    status = "✅" if validated else "❌"
    print(f"  Trigger (creative) coherence: {trigger_coherence:.4f}")
    print(f"  Control (routine) coherence:  {control_coherence:.4f}")
    print(f"  Difference: {control_coherence - trigger_coherence:+.4f}")
    print(f"  Creative more distributed? {status}")

    # Topic analysis
    print(f"\n  TOPIC CONFOUND TEST:")
    if validated:
        print(f"  → Creative IS more distributed even with topic controls")
        print(f"  → Supports COGNITIVE MODE hypothesis")
    else:
        print(f"  → Creative NOT more distributed")
        print(f"  → Could support TOPIC CONFOUND hypothesis")

    return {
        "probe": probe_name,
        "description": probe_data["description"],
        "trigger_coherence": float(trigger_coherence),
        "control_coherence": float(control_coherence),
        "difference": float(control_coherence - trigger_coherence),
        "validated": bool(validated),
    }


def main():
    parser = argparse.ArgumentParser(description="Topic-Controlled Creative Probe")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--output", type=str, default="topic_control_results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    model, tokenizer = load_model(args.model)
    model_name = args.model.split("/")[-1]

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "experiment": "topic_controlled_creative",
        "hypothesis_test": "Does topic similarity or cognitive mode drive coherence?",
        "probes": {},
    }

    for probe_name, probe_data in PROBES.items():
        result = run_probe(model, tokenizer, probe_name, probe_data)
        results["probes"][probe_name] = result

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    validated_count = sum(1 for r in results["probes"].values() if r["validated"])
    total = len(results["probes"])

    print(f"Validated: {validated_count}/{total}")

    if validated_count >= 2:
        print("\n→ COGNITIVE MODE hypothesis supported")
        print("  Creative processing is distributed regardless of topic diversity")
    else:
        print("\n→ TOPIC CONFOUND hypothesis may have merit")
        print("  Need to investigate further")

    output_file = output_dir / f"topic_control_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
