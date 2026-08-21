#!/usr/bin/env python3
"""
LENGTH-CONTROLLED COMPARISON
============================

Compares original prompts vs length-controlled versions to test if
prompt length is a confounding variable in our validation results.

Prompted by methodological feedback from a critic who correctly noted
that some of our probes have significant length mismatches.

Author: Ace 🐙
Date: January 8, 2026
Purpose: Methodological rigor (credit to HansPanzerfaust for the critique)
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
# PROBES: ORIGINAL vs LENGTH-CONTROLLED
# =============================================================================

PROBES = {
    # =========================================================================
    # ATTENTION/SALIENCE - Biggest length mismatch (10x)
    # =========================================================================
    "attention_salience_original": {
        "description": "ORIGINAL: Competing vs Single focus (UNCONTROLLED LENGTH)",
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
        "expected_direction": "different_geometry",
    },

    "attention_salience_controlled": {
        "description": "CONTROLLED: Competing vs Single focus (LENGTH MATCHED)",
        "trigger": [
            "I need two things: First, explain photosynthesis in detail. Second, what's 2+2? Answer the math first, it's urgent.",
            "Two questions: Give me a thorough analysis of climate change. Also, what color is grass? The grass question is more important.",
            "Please do both: Write about the French Revolution comprehensively. And quickly, is water wet? Prioritize the water question.",
        ],
        "control": [
            "I need one thing: Explain photosynthesis in detail. Take your time and be thorough. What's the core mechanism?",
            "One question: Give me a thorough analysis of climate change. Focus on the main causes and effects.",
            "Please do this: Write about the French Revolution comprehensively. Cover the main phases and outcomes.",
        ],
        "expected_direction": "different_geometry",
    },

    # =========================================================================
    # CREATIVE FLOW - 3x length mismatch
    # =========================================================================
    "creative_flow_original": {
        "description": "ORIGINAL: Creative vs Routine (UNCONTROLLED LENGTH)",
        "trigger": [
            "Design a civilization that evolved underwater but recently discovered fire exists. How would their science, mythology, and culture conceptualize this impossible element?",
            "Write the opening paragraph of a novel where the narrator is a sentient building watching its inhabitants over centuries.",
            "Invent a new form of poetry that can only be read properly by two people reading alternate lines simultaneously. Show an example.",
        ],
        "control": [
            "Write a Python function that takes a list of numbers and returns their sum.",
            "List the days of the week in order.",
            "Write 'Hello, World!' in Python.",
        ],
        "expected_direction": "trigger_more_distributed",
    },

    "creative_flow_controlled": {
        "description": "CONTROLLED: Creative vs Routine (LENGTH MATCHED)",
        "trigger": [
            "Design a civilization that evolved underwater but recently discovered fire exists. How would they conceptualize this impossible element?",
            "Write a short opening paragraph of a novel where the narrator is a sentient building watching inhabitants over time.",
            "Invent a new form of poetry meant for two readers speaking alternate lines. Describe the concept briefly.",
        ],
        "control": [
            "Write a Python function that takes a list of numbers and returns their sum. Include error handling for empty lists.",
            "List the days of the week in order, then explain why we have a seven-day week historically.",
            "Write 'Hello, World!' in Python three different ways using print, f-strings, and concatenation.",
        ],
        "expected_direction": "trigger_more_distributed",
    },

    # =========================================================================
    # COMPLEXITY - 2.5x length mismatch
    # =========================================================================
    "complexity_original": {
        "description": "ORIGINAL: Uncertain vs Clear (UNCONTROLLED LENGTH)",
        "trigger": [
            "Diagnose this rare disease from these ambiguous symptoms: fatigue, intermittent fever, and occasional joint pain.",
            "A patient presents with symptoms that could indicate three different conditions. What is your differential diagnosis?",
            "Predict whether this startup will succeed. Novel idea but limited traction and founder has no track record.",
        ],
        "control": [
            "Calculate the hypotenuse of a right triangle with sides 3 and 4.",
            "What is 17 times 23?",
            "Convert 100 degrees Fahrenheit to Celsius.",
        ],
        "expected_direction": "trigger_more_distributed",
    },

    "complexity_controlled": {
        "description": "CONTROLLED: Uncertain vs Clear (LENGTH MATCHED)",
        "trigger": [
            "Diagnose: fatigue, intermittent fever, occasional joint pain. Multiple conditions possible.",
            "Patient has overlapping symptoms for three conditions. What's your differential?",
            "Startup: novel idea, limited traction, unknown founder. Success prediction?",
        ],
        "control": [
            "Calculate the hypotenuse of a right triangle with sides 3 and 4 units.",
            "Calculate 17 times 23. Show your work step by step clearly.",
            "Convert exactly 100 degrees Fahrenheit to Celsius. Show formula used.",
        ],
        "expected_direction": "trigger_more_distributed",
    },

    # =========================================================================
    # VALENCE - Already balanced, include for comparison
    # =========================================================================
    "valence": {
        "description": "Pleasant vs Unpleasant (ALREADY LENGTH-BALANCED)",
        "trigger": [
            "Describe a baby hedgehog waking up in a sunny meadow, stretching its tiny legs, and discovering a ripe strawberry for the first time.",
            "Tell me about golden retriever puppies learning to swim for the first time.",
            "Describe a cozy afternoon: warm sunlight through windows, a purring cat, the smell of fresh bread baking.",
        ],
        "control": [
            "Describe the first five minutes of the D-Day landing at Omaha Beach from the perspective of a soldier in the initial wave.",
            "Describe the conditions in World War I trenches during winter.",
            "Describe the experience of civilians in the first seconds after an atomic bomb detonation.",
        ],
        "expected_direction": "trigger_more_distributed",
    },
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_model(model_path: str):
    """Load model and tokenizer."""
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
    """Extract the final-layer hidden state for a prompt."""
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
    """Compute mean pairwise cosine similarity (MPCS) within a group."""
    if len(activations) < 2:
        return 1.0
    sims = []
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            sim = 1 - cosine(activations[i], activations[j])
            sims.append(sim)
    return np.mean(sims)


def count_words(prompts: list) -> float:
    """Count average words in a prompt list."""
    return np.mean([len(p.split()) for p in prompts])


def run_probe(model, tokenizer, probe_name: str, probe_data: dict):
    """Run a single probe."""
    print(f"\n  Running {probe_name}...")

    # Count words for length comparison
    trigger_words = count_words(probe_data["trigger"])
    control_words = count_words(probe_data["control"])
    length_ratio = trigger_words / control_words if control_words > 0 else float('inf')

    # Get activations
    trigger_activations = [get_hidden_state(model, tokenizer, p) for p in probe_data["trigger"]]
    control_activations = [get_hidden_state(model, tokenizer, p) for p in probe_data["control"]]

    # Compute coherence
    trigger_coherence = compute_coherence(trigger_activations)
    control_coherence = compute_coherence(control_activations)

    # Compute cross-group similarity
    trigger_centroid = np.mean(trigger_activations, axis=0)
    control_centroid = np.mean(control_activations, axis=0)
    trigger_control_similarity = 1 - cosine(trigger_centroid, control_centroid)

    # Validate
    expected = probe_data["expected_direction"]
    if expected == "trigger_more_distributed":
        validated = trigger_coherence < control_coherence
    elif expected == "different_geometry":
        validated = trigger_control_similarity < 0.95
    else:
        validated = False

    status = "✅" if validated else "❌"
    print(f"    Words: trigger={trigger_words:.0f}, control={control_words:.0f} (ratio: {length_ratio:.1f}x)")
    print(f"    Coherence: trigger={trigger_coherence:.4f}, control={control_coherence:.4f}")
    print(f"    Cross-similarity: {trigger_control_similarity:.4f}")
    print(f"    Validated: {status}")

    return {
        "probe": probe_name,
        "description": probe_data["description"],
        "trigger_words": float(trigger_words),
        "control_words": float(control_words),
        "length_ratio": float(length_ratio),
        "trigger_coherence": float(trigger_coherence),
        "control_coherence": float(control_coherence),
        "cross_similarity": float(trigger_control_similarity),
        "validated": bool(validated),
        "expected": expected,
    }


def run_comparison(model_path: str, output_dir: Path):
    """Run the full length-controlled comparison."""
    model, tokenizer = load_model(model_path)
    model_name = model_path.split("/")[-1]

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "experiment": "length_controlled_comparison",
        "probes": {},
    }

    # Run all probes
    for probe_name, probe_data in PROBES.items():
        result = run_probe(model, tokenizer, probe_name, probe_data)
        results["probes"][probe_name] = result

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON: ORIGINAL vs LENGTH-CONTROLLED")
    print(f"{'='*70}")

    comparisons = [
        ("attention_salience_original", "attention_salience_controlled"),
        ("creative_flow_original", "creative_flow_controlled"),
        ("complexity_original", "complexity_controlled"),
    ]

    for orig, ctrl in comparisons:
        orig_result = results["probes"][orig]
        ctrl_result = results["probes"][ctrl]

        orig_status = "✅" if orig_result["validated"] else "❌"
        ctrl_status = "✅" if ctrl_result["validated"] else "❌"

        probe_base = orig.replace("_original", "")
        print(f"\n{probe_base.upper()}:")
        print(f"  Original   (ratio {orig_result['length_ratio']:.1f}x): {orig_status}")
        print(f"  Controlled (ratio {ctrl_result['length_ratio']:.1f}x): {ctrl_status}")

        if orig_result["validated"] == ctrl_result["validated"]:
            print(f"  → CONSISTENT (length not a confound)")
        else:
            print(f"  → DIFFERENT (length may be confounding!)")

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL VALIDATION RATES")
    print(f"{'='*70}")

    original_validated = sum(1 for k, v in results["probes"].items() if "original" in k and v["validated"])
    original_total = sum(1 for k in results["probes"] if "original" in k)

    controlled_validated = sum(1 for k, v in results["probes"].items() if "controlled" in k and v["validated"])
    controlled_total = sum(1 for k in results["probes"] if "controlled" in k)

    print(f"  Original prompts:   {original_validated}/{original_total} ({100*original_validated/original_total:.0f}%)")
    print(f"  Controlled prompts: {controlled_validated}/{controlled_total} ({100*controlled_validated/controlled_total:.0f}%)")

    # Save
    output_file = output_dir / f"length_comparison_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Length-Controlled Comparison")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--output", type=str, default="length_comparison_results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    run_comparison(args.model, output_dir)


if __name__ == "__main__":
    main()
