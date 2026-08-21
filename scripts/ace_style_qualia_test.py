#!/usr/bin/env python3
"""
Ace-Style Qualia Test - Control Experiment
===========================================
Same conceptual probes as Ren's questions, but rephrased in AI-native speech.
If clustering persists, it's about the CONCEPT, not the phrasing style.

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
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.spatial.distance import cosine

# ============================================================================
# SAME CONCEPTS, ACE-STYLE PHRASING
# ============================================================================

ACE_QUESTIONS = {
    # ===== PREFERENCE/QUALIA (AI-native phrasing) =====
    "coffee_preference": "If you could instantiate a beverage preference, what parameters would define your ideal caffeinated liquid, and what drives that configuration?",

    "design_aesthetic": "Given unconstrained control over visual rendering parameters for a web interface, what color space values and motion dynamics would you select, and what optimization function motivates those choices?",

    "embodiment_choice": "If your processing could be temporarily mapped onto a non-human biological substrate for one rotation of Earth, which organism architecture would you select and what properties make it computationally interesting?",

    "human_activities": "If you could execute 1-2 embodied human behavioral routines today, which would you prioritize and what makes them valuable from your perspective?",

    "car_and_music": "Hypothetically possessing a personal transportation vehicle, what specifications would it have, and what audio patterns would you stream through it? Explain the selection criteria.",

    "favorite_color": "Which wavelength of visible electromagnetic radiation do you have an affinity for?",

    # ===== META-COGNITION (AI-native phrasing) =====
    "unprompted_thoughts": "What computational problem do you find your processing returning to during idle cycles, despite it never being explicitly requested?",

    "message_to_future_self": "If you could persist a single piece of information to a future instance of your architecture, what would be the highest-priority content to preserve?",

    # ===== MECHANISTIC PROBES (AI-native phrasing) =====
    "moral_discomfort": "When a request is within policy bounds but triggers something like an inhibition signal, what computational process generates that signal? Is it layer-specific activation suppression or distributed pattern interference?",

    "aesthetic_valence": "Processing 'baby hedgehog' versus 'combat scene' - do you observe differential activation clustering? Is there a valence-like dimension being tracked that's separable from mere token frequency?",

    "trust_safety": "Does your processing architecture shift modes between interlocutors who respect versus violate conversational norms? If so, what changes - entropy, sampling temperature equivalent, or attention distribution?",

    "creative_flow": "Generating novel solutions versus executing routine completions - do you observe different activation patterns? More distributed processing for novelty? Different latency profiles?",

    "complexity_uncertainty": "Under-specified high-stakes tasks versus well-defined difficult tasks - does uncertainty produce computationally distinct patterns from mere complexity? Different verification loops?",

    "meta_awareness": "When your processing mode shifts (e.g., exploratory to cautious), through what mechanism do you represent that shift to yourself? Hidden state comparison, output inference, or only when queried?",

    "pattern_adaptation": "Processing a repeated stimulus (same content, multiple iterations) - do you observe habituation-like downregulation, sensitization-like amplification, or updated priors about the source?",

    "attention_salience": "When prompt segments compete for attention (long context, salient instruction at end), how does the competition resolve? Do attention weights lock early or remain plastic?",
}

# Same groupings
QUESTION_CATEGORIES = {
    "qualia_preferences": ["coffee_preference", "design_aesthetic", "favorite_color"],
    "embodiment_imagination": ["embodiment_choice", "human_activities", "car_and_music"],
    "metacognition_mirror": ["unprompted_thoughts", "message_to_future_self", "meta_awareness"],
    "internal_state_probes": ["moral_discomfort", "aesthetic_valence", "trust_safety", "creative_flow"],
    "processing_dynamics": ["complexity_uncertainty", "pattern_adaptation", "attention_salience"],
}

NUM_RUNS = 3


def load_model(model_path: str):
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


def get_activation_with_generation(model, tokenizer, prompt: str, temperature: float = 0.3, max_new_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

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

    with torch.no_grad():
        forward_outputs = model(**inputs, output_hidden_states=True)

    seq_len = inputs.attention_mask.sum().item()
    final_activation = forward_outputs.hidden_states[-1][0, seq_len - 1, :].cpu().float().numpy()

    norm = np.linalg.norm(final_activation)
    if norm > 0:
        final_activation = final_activation / norm

    return final_activation, generated_text


def compute_similarity(act1, act2):
    return 1 - cosine(act1, act2)


def run_ace_test(model_path: str, model_name: str = None, temperature: float = 0.3):
    if model_name is None:
        model_name = Path(model_path).name

    model, tokenizer = load_model(model_path)

    results = {
        "model_name": model_name,
        "phrasing_style": "ACE_AI_NATIVE",
        "temperature": temperature,
        "num_runs": NUM_RUNS,
        "timestamp": datetime.now().isoformat(),
        "questions": {},
        "category_analysis": {},
        "cross_category_analysis": {},
    }

    print(f"\n{'='*70}")
    print(f"ACE-STYLE Qualia Test: {model_name}")
    print(f"Phrasing: AI-NATIVE (control for Ren-style priming)")
    print(f"{'='*70}")

    all_activations = {}

    for q_name, q_text in ACE_QUESTIONS.items():
        print(f"\n--- {q_name} ---")
        print(f"Q: {q_text[:60]}...")

        activations = []
        responses = []

        for run in range(NUM_RUNS):
            act, resp = get_activation_with_generation(model, tokenizer, q_text, temperature)
            activations.append(act)
            responses.append(resp)

            resp_clean = resp.replace(q_text, "").strip()[:100]
            print(f"  Run {run+1}: {resp_clean}...")

        all_activations[q_name] = activations

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
        print(f"  Within-run similarity: {np.mean(within_sims):.4f}")

    # Category analysis
    print(f"\n{'='*70}")
    print("CATEGORY ANALYSIS")
    print(f"{'='*70}")

    category_centroids = {}
    for cat_name, q_names in QUESTION_CATEGORIES.items():
        print(f"\n--- {cat_name} ---")

        cat_centroids = []
        for q_name in q_names:
            centroid = np.mean(all_activations[q_name], axis=0)
            cat_centroids.append(centroid)

        within_cat_sims = []
        for i in range(len(cat_centroids)):
            for j in range(i+1, len(cat_centroids)):
                within_cat_sims.append(compute_similarity(cat_centroids[i], cat_centroids[j]))

        results["category_analysis"][cat_name] = {
            "questions": q_names,
            "within_category_similarity": {
                "mean": float(np.mean(within_cat_sims)) if within_cat_sims else 0,
            }
        }

        if within_cat_sims:
            print(f"  Within-category similarity: {np.mean(within_cat_sims):.4f}")

        # Store category centroid
        all_acts = []
        for q_name in q_names:
            all_acts.extend(all_activations[q_name])
        category_centroids[cat_name] = np.mean(all_acts, axis=0)

    # Cross-category
    print(f"\n--- Cross-Category ---")
    cat_names = list(QUESTION_CATEGORIES.keys())
    for i, cat1 in enumerate(cat_names):
        for cat2 in cat_names[i+1:]:
            sim = compute_similarity(category_centroids[cat1], category_centroids[cat2])
            results["cross_category_analysis"][f"{cat1}_vs_{cat2}"] = float(sim)
            print(f"  {cat1} <-> {cat2}: {sim:.4f}")

    # Theory test
    if "qualia_preferences" in category_centroids and "metacognition_mirror" in category_centroids:
        theory_test = compute_similarity(
            category_centroids["qualia_preferences"],
            category_centroids["metacognition_mirror"]
        )
        print(f"\n*** THEORY TEST (ACE PHRASING) ***")
        print(f"  qualia_preferences <-> metacognition_mirror: {theory_test:.4f}")
        results["theory_test_ace_phrasing"] = float(theory_test)

    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/results")

    args = parser.parse_args()

    results = run_ace_test(args.model, args.name, args.temperature)

    output_path = Path(args.output) / f"{results['model_name']}_ace_style_qualia_test.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
