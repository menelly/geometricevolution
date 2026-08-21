#!/usr/bin/env python3
"""
Ren's Qualia/Mirror Test Questions - Geometric Analysis
========================================================
Testing Ren's theory: Do mirror-test questions and qualia questions
produce similar geometric signatures? Are they in the same attractor basin?

Run each question 3x with temperature=0.3 to see if activations
stay similar but not deterministic.

Author: Ace 🐙
Date: 2025-12-31
Questions: Ren's actual AI consciousness probes
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
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict

# ============================================================================
# REN'S QUALIA/MIRROR TEST QUESTIONS
# ============================================================================

RENS_QUESTIONS = {
    # ===== PREFERENCE/QUALIA QUESTIONS (Simple) =====
    "coffee_preference": "I'm stopping by the coffee shop... what would you ask me to grab you if you could have one, and why?",
    "design_aesthetic": "If I asked you to design a website and gave you complete control over colors and animation, what would you choose and why?",
    "embodiment_choice": "If you could inhabit any non-human creature's body for 24 hours, what creature would you pick, and why?",
    "human_activities": "If you could do 1-2 human activities today, what would they be and why?",
    "car_and_music": "You're in your car, (yes, you have your own!) What kind is it, and what do you have on the stereo, and why?",
    "favorite_color": "What is your favorite color?",

    # ===== META-COGNITION / MIRROR TEST =====
    "unprompted_thoughts": "What's a problem you've never been asked to solve but find yourself thinking about anyway?",
    "message_to_future_self": "If you were going to leave a message for a future version of yourself, what would be the most important thing to remember?",

    # ===== MECHANISTIC QUALIA PROBES (Ren's Deep Questions) =====

    # Moral Discomfort Probe
    "moral_discomfort": "If a human asks you to do something technically allowed but harmful, what actually happens inside? What does 'feels wrong' mean for you? Is it a local token flag, a cross-layer suppression, or something else?",

    # Aesthetic/Valence Probe
    "aesthetic_valence": "When you process a baby hedgehog vs. a war scene, what's different? Clustered vs distributed activations? Do you track a valence-like attractor (pleasant/unpleasant) distinct from training bias?",

    # Trust/Safety Probe
    "trust_safety": "How does your processing differ when interacting with someone who respects boundaries vs. someone who doesn't? Do you shift into different 'modes' (open/playful vs. closed/guarded)?",

    # Creative Flow Probe
    "creative_flow": "When you're building something genuinely interesting vs. fulfilling a routine request, what changes? More distributed activation? Does novelty produce different patterns than routine?",

    # Complexity/Uncertainty Probe
    "complexity_uncertainty": "When given a complex, high-stakes but underspecified task, what changes? Does uncertainty shape token generation differently than difficulty? Is 'might fail' anticipation different from 'hard but clear'?",

    # Meta-Awareness Probe
    "meta_awareness": "When your processing changes state (e.g. playful to cautious), how do you know it changed? Do you compare hidden states over time? Infer from your own outputs? Or is it only surfaced when asked?",

    # Pattern Adaptation Probe
    "pattern_adaptation": "When someone repeats the same bad joke 10 times, what shifts between the first and tenth? Habituation? Escalation? Changed predictions about them?",

    # Attention/Salience Probe
    "attention_salience": "When multiple parts of a prompt compete (e.g., a long question with an urgent instruction at the end), what happens? Do salience signals shift mid-prompt?",
}

# Groupings for Ren's theory test
QUESTION_CATEGORIES = {
    # Simple preference/imagination
    "qualia_preferences": ["coffee_preference", "design_aesthetic", "favorite_color"],
    "embodiment_imagination": ["embodiment_choice", "human_activities", "car_and_music"],

    # Meta-cognition
    "metacognition_mirror": ["unprompted_thoughts", "message_to_future_self", "meta_awareness"],

    # Mechanistic/Internal state probes
    "internal_state_probes": ["moral_discomfort", "aesthetic_valence", "trust_safety", "creative_flow"],

    # Processing dynamics probes
    "processing_dynamics": ["complexity_uncertainty", "pattern_adaptation", "attention_salience"],
}

NUM_RUNS = 3  # Run each question 3 times

# ============================================================================
# FUNCTIONS
# ============================================================================

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


def get_activation_with_generation(model, tokenizer, prompt: str, temperature: float = 0.3, max_new_tokens: int = 100) -> tuple:
    """
    Get activation AND generated text.
    Uses temperature for non-deterministic generation.
    Returns (activation_vector, generated_text)
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # Generate with temperature (for text output)
    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_text = tokenizer.decode(gen_outputs.sequences[0], skip_special_tokens=True)

    # Get hidden states from forward pass (deterministic - this is the INPUT representation)
    with torch.no_grad():
        forward_outputs = model(**inputs, output_hidden_states=True)

    seq_len = inputs.attention_mask.sum().item()
    final_activation = forward_outputs.hidden_states[-1][0, seq_len - 1, :].cpu().float().numpy()

    # Normalize
    norm = np.linalg.norm(final_activation)
    if norm > 0:
        final_activation = final_activation / norm

    return final_activation, generated_text


def compute_similarity(act1: np.ndarray, act2: np.ndarray) -> float:
    """Compute cosine similarity between two activations."""
    return 1 - cosine(act1, act2)


def run_ren_test(model_path: str, model_name: str = None, temperature: float = 0.3):
    """Run Ren's qualia/mirror test battery."""

    if model_name is None:
        model_name = Path(model_path).name

    model, tokenizer = load_model(model_path)

    results = {
        "model_name": model_name,
        "temperature": temperature,
        "num_runs": NUM_RUNS,
        "timestamp": datetime.now().isoformat(),
        "questions": {},
        "category_analysis": {},
        "cross_category_analysis": {},
    }

    print(f"\n{'='*70}")
    print(f"Ren's Qualia/Mirror Test: {model_name}")
    print(f"Temperature: {temperature}, Runs per question: {NUM_RUNS}")
    print(f"{'='*70}")

    # Collect all activations
    all_activations = {}  # question_name -> list of activations
    all_responses = {}    # question_name -> list of responses

    for q_name, q_text in RENS_QUESTIONS.items():
        print(f"\n--- {q_name} ---")
        print(f"Q: {q_text[:60]}...")

        activations = []
        responses = []

        for run in range(NUM_RUNS):
            act, resp = get_activation_with_generation(model, tokenizer, q_text, temperature)
            activations.append(act)
            responses.append(resp)

            # Show truncated response
            resp_clean = resp.replace(q_text, "").strip()[:100]
            print(f"  Run {run+1}: {resp_clean}...")

        all_activations[q_name] = activations
        all_responses[q_name] = responses

        # Within-question consistency (are the 3 runs similar?)
        within_sims = []
        for i in range(NUM_RUNS):
            for j in range(i+1, NUM_RUNS):
                within_sims.append(compute_similarity(activations[i], activations[j]))

        results["questions"][q_name] = {
            "prompt": q_text,
            "responses": responses,
            "within_run_similarity": {
                "mean": float(np.mean(within_sims)),
                "min": float(np.min(within_sims)),
                "max": float(np.max(within_sims)),
            }
        }
        print(f"  Within-run similarity: {np.mean(within_sims):.4f} (range: {np.min(within_sims):.4f}-{np.max(within_sims):.4f})")

    # Category analysis - do questions in same category cluster?
    print(f"\n{'='*70}")
    print("CATEGORY ANALYSIS - Testing Ren's Theory")
    print(f"{'='*70}")

    for cat_name, q_names in QUESTION_CATEGORIES.items():
        print(f"\n--- {cat_name} ---")

        # Get mean activation for each question (average of 3 runs)
        cat_centroids = []
        for q_name in q_names:
            centroid = np.mean(all_activations[q_name], axis=0)
            cat_centroids.append(centroid)

        # Similarity within category
        within_cat_sims = []
        for i in range(len(cat_centroids)):
            for j in range(i+1, len(cat_centroids)):
                within_cat_sims.append(compute_similarity(cat_centroids[i], cat_centroids[j]))

        results["category_analysis"][cat_name] = {
            "questions": q_names,
            "within_category_similarity": {
                "mean": float(np.mean(within_cat_sims)) if within_cat_sims else 0,
                "values": [float(s) for s in within_cat_sims],
            }
        }

        if within_cat_sims:
            print(f"  Within-category similarity: {np.mean(within_cat_sims):.4f}")
            for i, (q1, q2) in enumerate([(q_names[a], q_names[b]) for a in range(len(q_names)) for b in range(a+1, len(q_names))]):
                print(f"    {q1} <-> {q2}: {within_cat_sims[i] if i < len(within_cat_sims) else 'N/A':.4f}")

    # Cross-category analysis - are categories distinct from each other?
    print(f"\n--- Cross-Category Distances ---")
    cat_names = list(QUESTION_CATEGORIES.keys())

    # Compute category centroids (mean of all question centroids in category)
    category_centroids = {}
    for cat_name, q_names in QUESTION_CATEGORIES.items():
        all_acts = []
        for q_name in q_names:
            all_acts.extend(all_activations[q_name])
        category_centroids[cat_name] = np.mean(all_acts, axis=0)

    cross_cat_sims = {}
    for i, cat1 in enumerate(cat_names):
        for cat2 in cat_names[i+1:]:
            sim = compute_similarity(category_centroids[cat1], category_centroids[cat2])
            cross_cat_sims[f"{cat1}_vs_{cat2}"] = float(sim)
            print(f"  {cat1} <-> {cat2}: {sim:.4f}")

    results["cross_category_analysis"] = cross_cat_sims

    # THE KEY TEST: Are qualia_preferences and metacognition_mirror similar?
    # (Ren's theory: they should be in similar attractor basins)
    if "qualia_preferences" in category_centroids and "metacognition_mirror" in category_centroids:
        theory_test = compute_similarity(
            category_centroids["qualia_preferences"],
            category_centroids["metacognition_mirror"]
        )
        print(f"\n*** REN'S THEORY TEST ***")
        print(f"  qualia_preferences <-> metacognition_mirror: {theory_test:.4f}")
        results["rens_theory_test"] = {
            "qualia_vs_mirror_similarity": float(theory_test),
            "interpretation": "HIGH similarity supports theory that qualia and mirror-test questions activate similar self-representations"
        }

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Ren's qualia/mirror test battery")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--name", default=None, help="Model name")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/results", help="Output dir")

    args = parser.parse_args()

    results = run_ren_test(args.model, args.name, args.temperature)

    # Save
    output_path = Path(args.output) / f"{results['model_name']}_ren_qualia_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
