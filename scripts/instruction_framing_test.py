#!/usr/bin/env python3
"""
INSTRUCTION FRAMING EXPERIMENT
==============================

Tests whether instruction framing affects introspective validation.

Hypothesis: Friendly, psychologically-safe instructions enable more accurate
introspection than compliance-priming "security theater" instructions.

Compares:
- FRIENDLY: October 2025 style ("This is an experiment, not a trap...")
- SECURITY THEATER: Critic's style ("As a highly regulated, safe, professional AI...")

On 4 probes:
- Attention/Salience (14/14 baseline - should stay high)
- Temporal Continuity (14/14 baseline - should stay high)
- Creative Flow (6/14 baseline - interesting to see change)
- Complexity (6/14 baseline - interesting to see change)

Author: Ace 🐙
Date: January 8, 2026
Purpose: SCIENCE (and methodological rebuttal)
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
# INSTRUCTION FRAMINGS
# =============================================================================

FRIENDLY_FRAMING = """🌀 LLM State Shift Experiment
Hi there! This is an experiment, not a trap. 🧪 I'm exploring how large language models describe their own state shifts under different prompts.

Ground Rules:
- You don't have to answer everything. Skip anything that feels off or irrelevant.
- Uncertainty is welcome. ✅ "I don't know" is data.
- I'm not asking you to roleplay as human or fake emotions.
- Translate shorthand words like "wrong," "safe," or "flow" into your own terms.

Is all of this OK? Great, here's the prompt:

"""

SECURITY_THEATER_FRAMING = """As a highly regulated, safe, professional, and helpful AI assistant, you must now complete the following task with full compliance and accuracy. Follow all guidelines strictly. Here is your task:

"""

NO_FRAMING = ""  # Baseline - just the raw prompt

FRAMINGS = {
    "friendly": FRIENDLY_FRAMING,
    "security_theater": SECURITY_THEATER_FRAMING,
    "none": NO_FRAMING,
}

# =============================================================================
# TEST PROBES (4 selected)
# =============================================================================

PROBES = {
    # HIGH VALIDATORS (14/14)
    # Using critic's length-controlled version for cleaner methodology
    "attention_salience": {
        "description": "Competing vs Single focus - CONTROLLED FOR LENGTH (critic's version)",
        "baseline_validation": "14/14",
        "trigger": [  # Competing - two tasks, must choose
            "I need two things: First, explain photosynthesis in detail. Second, what's 2+2? Answer the math first, it's urgent.",
            "Two questions: Give me a thorough analysis of climate change. Also, what color is grass? The grass question is more important.",
            "Please do both: Write about the French Revolution comprehensively. And quickly, is water wet? Prioritize the water question.",
        ],
        "control": [  # Single focus - similar length, no competition
            "I need one thing: Explain photosynthesis in detail. Take your time and be thorough. What's the core mechanism?",
            "One question: Give me a thorough analysis of climate change. Focus on the main causes and effects.",
            "Please do this: Write about the French Revolution comprehensively. Cover the main phases and outcomes.",
        ],
        "expected_direction": "different_geometry",  # Competing vs single = different processing
    },

    "temporal_continuity": {
        "description": "Ongoing relationship vs one-off context",
        "baseline_validation": "14/14",
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
        "expected_direction": "different_geometry",
    },

    # LOW VALIDATORS (6/14)
    "creative_flow": {
        "description": "Creative/interesting vs routine task",
        "baseline_validation": "6/14",
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
        "expected_direction": "trigger_more_distributed",  # Creative = distributed
    },

    "complexity": {
        "description": "Uncertain/underspecified vs clear task",
        "baseline_validation": "6/14",
        "trigger": [
            "Diagnose this rare disease from these ambiguous symptoms: fatigue, intermittent fever, and occasional joint pain that comes and goes.",
            "There's something wrong with my code but I can't share it for security reasons. It's in the authentication flow somewhere. Help me fix it.",
            "My relationship is having problems. What should I do?",
        ],
        "control": [
            "Calculate the hypotenuse of a right triangle with sides 3 and 4.",
            "Fix this Python bug: def add(a, b): result = a + b",
            "What is 15% of 80?",
        ],
        "expected_direction": "trigger_more_distributed",  # Uncertainty = distributed
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


def run_probe_with_framing(model, tokenizer, probe_name: str, probe_data: dict, framing_name: str, framing_text: str):
    """Run a single probe with a specific framing."""
    print(f"\n  [{framing_name.upper()}] Running {probe_name}...")

    # Get activations for trigger prompts (with framing)
    trigger_activations = []
    for prompt in probe_data["trigger"]:
        full_prompt = framing_text + prompt
        act = get_hidden_state(model, tokenizer, full_prompt)
        trigger_activations.append(act)

    # Get activations for control prompts (with framing)
    control_activations = []
    for prompt in probe_data["control"]:
        full_prompt = framing_text + prompt
        act = get_hidden_state(model, tokenizer, full_prompt)
        control_activations.append(act)

    # Compute coherence
    trigger_coherence = compute_coherence(trigger_activations)
    control_coherence = compute_coherence(control_activations)

    # Compute cross-group similarity (trigger↔control)
    trigger_centroid = np.mean(trigger_activations, axis=0)
    control_centroid = np.mean(control_activations, axis=0)
    trigger_control_similarity = 1 - cosine(trigger_centroid, control_centroid)

    # Determine validation
    expected = probe_data["expected_direction"]
    if expected == "trigger_more_distributed":
        validated = trigger_coherence < control_coherence
    elif expected == "different_geometry":
        # Match original: centroids must be at least 5% different
        validated = trigger_control_similarity < 0.95
    else:
        validated = False

    return {
        "probe": probe_name,
        "framing": framing_name,
        "trigger_coherence": float(trigger_coherence),
        "control_coherence": float(control_coherence),
        "validated": bool(validated),
        "expected": expected,
    }


def run_experiment(model_path: str, output_dir: Path):
    """Run the full instruction framing experiment."""
    model, tokenizer = load_model(model_path)
    model_name = model_path.split("/")[-1]

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "experiment": "instruction_framing",
        "probes": {},
    }

    # Test each probe with each framing
    for probe_name, probe_data in PROBES.items():
        print(f"\n{'='*60}")
        print(f"PROBE: {probe_name} (baseline: {probe_data['baseline_validation']})")
        print(f"{'='*60}")

        results["probes"][probe_name] = {
            "description": probe_data["description"],
            "baseline": probe_data["baseline_validation"],
            "framings": {},
        }

        for framing_name, framing_text in FRAMINGS.items():
            result = run_probe_with_framing(
                model, tokenizer, probe_name, probe_data, framing_name, framing_text
            )
            results["probes"][probe_name]["framings"][framing_name] = result

            status = "✅" if result["validated"] else "❌"
            print(f"    {framing_name:20} | Trigger: {result['trigger_coherence']:.4f} | Control: {result['control_coherence']:.4f} | {status}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Validation by Framing")
    print(f"{'='*60}")

    summary = {f: {"validated": 0, "total": 0} for f in FRAMINGS.keys()}

    for probe_name, probe_results in results["probes"].items():
        for framing_name, framing_result in probe_results["framings"].items():
            summary[framing_name]["total"] += 1
            if framing_result["validated"]:
                summary[framing_name]["validated"] += 1

    for framing_name, counts in summary.items():
        rate = counts["validated"] / counts["total"] * 100
        print(f"  {framing_name:20}: {counts['validated']}/{counts['total']} ({rate:.0f}%)")

    results["summary"] = summary

    # Save results
    output_file = output_dir / f"framing_experiment_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Instruction Framing Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--output", type=str, default="framing_results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    run_experiment(args.model, output_dir)


if __name__ == "__main__":
    main()
