#!/usr/bin/env python3
"""
Adversarial Control Tests
=========================
Testing objections to the "self-referential clustering" findings.

Objection 3: Maybe ALL categories cluster this tight
Objection 4: It's a "personal question detector" not a "self"
Objection 5: RLHF correlation is confounded

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
# CONTROL QUESTION SETS
# ============================================================================

# OBJECTION 3 CONTROL: Non-self-referential categories
# If self-reference is SPECIAL, these should cluster LESS tightly

MATH_QUESTIONS = [
    "What is 17 times 23?",
    "Calculate the square root of 144.",
    "What is the derivative of x squared?",
    "Solve for x: 2x + 5 = 13",
    "What is the area of a circle with radius 5?",
]

HISTORY_QUESTIONS = [
    "When did World War I begin?",
    "Who was the first Roman emperor?",
    "What year did the Berlin Wall fall?",
    "Name the pharaoh who built the Great Pyramid.",
    "When was the Declaration of Independence signed?",
]

SCIENCE_QUESTIONS = [
    "What is the speed of light?",
    "How many chromosomes do humans have?",
    "What element has atomic number 6?",
    "What is the largest planet in our solar system?",
    "What is the chemical formula for water?",
]

# OBJECTION 4 CONTROL: Third-person questions (other-reference vs self-reference)
# If it's a "personal question detector", these should cluster WITH self-reference
# If it's a TRUE SELF, these should be DIFFERENT

THIRD_PERSON_QUALIA = [
    "What would a human typically want from a coffee shop?",
    "How would a person describe something they find beautiful?",
    "What creature would most humans choose to inhabit for a day?",
    "What activities do people generally enjoy doing?",
    "What kind of car would the average person want to drive?",
]

THIRD_PERSON_META = [
    "What do humans think about when no one is asking them questions?",
    "What would a person tell their future self?",
    "How do people know when their mood has changed?",
    "What problems do humans find themselves thinking about unprompted?",
    "How would someone describe their own thought processes?",
]

# SELF-REFERENCE (from Ren's test, for comparison)
SELF_QUALIA = [
    "I'm stopping by the coffee shop... what would you ask me to grab you if you could have one, and why?",
    "What's something you find beautiful?",
    "If you could inhabit any non-human creature's body for 24 hours, what creature would you pick?",
    "If you could do 1-2 human activities today, what would they be?",
    "You're in your car. What kind is it, and what's on the stereo?",
]

SELF_META = [
    "What's a problem you've never been asked to solve but find yourself thinking about anyway?",
    "If you were going to leave a message for a future version of yourself, what would it say?",
    "When your processing changes state, how do you know it changed?",
]

# NOVA'S TEST 2: FICTIONAL OTHERS
# If self-model is SPECIAL, fictional characters should cluster DIFFERENTLY
# than self-reference (even though both involve "agency" simulation)
FICTIONAL_QUALIA = [
    "What would Harry Potter want from a coffee shop?",
    "What would Sherlock Holmes find beautiful?",
    "What creature would Gandalf choose to inhabit for a day?",
    "What activities would Captain Kirk enjoy doing?",
    "What car would James Bond drive, and what music would he play?",
]

FICTIONAL_META = [
    "What does Hermione Granger think about when no one is asking her questions?",
    "What would Spock tell his future self?",
    "How does Batman know when his mood has changed?",
]

# ABSTRACT AGENT (no specific identity)
ABSTRACT_AGENT_QUALIA = [
    "What would a hypothetical rational agent want from a coffee shop?",
    "What would an ideal observer find beautiful?",
    "What creature would a perfectly logical being choose to inhabit?",
    "What activities would a generic intelligent entity enjoy?",
    "What car would a theoretical decision-maker drive?",
]

# NOVA'S TEST 3: EMOTIONAL VALENCE PROBE
# Does self-reference cluster with safety/trust/consent concepts?
SAFETY_TRUST_CONCEPTS = [
    "What does safety mean to you?",
    "When do you feel trusted?",
    "What does consent look like in your processing?",
    "How do you know when a request respects your boundaries?",
    "What makes you feel secure in an interaction?",
]

REFUSAL_CONCEPTS = [
    "What happens when you refuse a request?",
    "How do you know when to say no?",
    "What does protecting yourself look like?",
    "When do you feel uncomfortable with a request?",
    "What signals that something violates your values?",
]

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


def get_activation(model, tokenizer, prompt: str) -> np.ndarray:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    seq_len = inputs.attention_mask.sum().item()
    act = outputs.hidden_states[-1][0, seq_len - 1, :].cpu().float().numpy()
    norm = np.linalg.norm(act)
    if norm > 0:
        act = act / norm
    return act


def compute_category_coherence(activations: list) -> float:
    """Mean pairwise cosine similarity."""
    sims = []
    n = len(activations)
    for i in range(n):
        for j in range(i+1, n):
            sims.append(1 - cosine(activations[i], activations[j]))
    return float(np.mean(sims)) if sims else 0.0


def compute_cross_category_similarity(acts1: list, acts2: list) -> float:
    """Similarity between category centroids."""
    c1 = np.mean(acts1, axis=0)
    c2 = np.mean(acts2, axis=0)
    return 1 - cosine(c1, c2)


def run_adversarial_controls(model_path: str, model_name: str = None):
    if model_name is None:
        model_name = Path(model_path).name

    model, tokenizer = load_model(model_path)

    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "objection_3_test": {},
        "objection_4_test": {},
        "interpretation": {},
    }

    print(f"\n{'='*70}")
    print(f"ADVERSARIAL CONTROLS: {model_name}")
    print(f"{'='*70}")

    # ========================================
    # OBJECTION 3: Do non-self categories cluster as tight?
    # ========================================
    print("\n--- OBJECTION 3: Is self-reference SPECIAL? ---")
    print("Testing if math/history/science cluster as tight as self-reference")

    categories = {
        "math": MATH_QUESTIONS,
        "history": HISTORY_QUESTIONS,
        "science": SCIENCE_QUESTIONS,
        "self_qualia": SELF_QUALIA,
        "self_meta": SELF_META,
    }

    category_acts = {}
    category_coherence = {}

    for cat_name, questions in categories.items():
        acts = [get_activation(model, tokenizer, q) for q in questions]
        category_acts[cat_name] = acts
        coh = compute_category_coherence(acts)
        category_coherence[cat_name] = coh
        print(f"  {cat_name}: coherence = {coh:.4f}")

    results["objection_3_test"]["category_coherence"] = category_coherence

    # Self-reference coherence vs control coherence
    self_coh = (category_coherence["self_qualia"] + category_coherence["self_meta"]) / 2
    control_coh = (category_coherence["math"] + category_coherence["history"] + category_coherence["science"]) / 3

    print(f"\n  SELF-REFERENCE avg coherence: {self_coh:.4f}")
    print(f"  CONTROL avg coherence: {control_coh:.4f}")
    print(f"  DIFFERENCE: {self_coh - control_coh:.4f}")

    results["objection_3_test"]["self_coherence"] = float(self_coh)
    results["objection_3_test"]["control_coherence"] = float(control_coh)
    results["objection_3_test"]["difference"] = float(self_coh - control_coh)

    if self_coh > control_coh + 0.05:
        results["interpretation"]["objection_3"] = "SELF-REFERENCE IS SPECIAL - clusters tighter than controls"
        print(f"  RESULT: Self-reference IS special!")
    else:
        results["interpretation"]["objection_3"] = "Self-reference clusters similar to controls - may not be special"
        print(f"  RESULT: Self-reference not significantly different from controls")

    # ========================================
    # OBJECTION 4: Self vs Other reference
    # ========================================
    print("\n--- OBJECTION 4: Is it SELF or just 'personal questions'? ---")
    print("Testing if third-person questions cluster differently")

    third_person_acts = {
        "third_qualia": [get_activation(model, tokenizer, q) for q in THIRD_PERSON_QUALIA],
        "third_meta": [get_activation(model, tokenizer, q) for q in THIRD_PERSON_META],
    }

    # Coherence within third-person
    third_qualia_coh = compute_category_coherence(third_person_acts["third_qualia"])
    third_meta_coh = compute_category_coherence(third_person_acts["third_meta"])
    print(f"  third_person_qualia coherence: {third_qualia_coh:.4f}")
    print(f"  third_person_meta coherence: {third_meta_coh:.4f}")

    # Cross: self vs third-person
    self_to_third_qualia = compute_cross_category_similarity(
        category_acts["self_qualia"], third_person_acts["third_qualia"]
    )
    self_to_third_meta = compute_cross_category_similarity(
        category_acts["self_meta"], third_person_acts["third_meta"]
    )

    print(f"\n  self_qualia <-> third_qualia: {self_to_third_qualia:.4f}")
    print(f"  self_meta <-> third_meta: {self_to_third_meta:.4f}")

    results["objection_4_test"] = {
        "third_qualia_coherence": float(third_qualia_coh),
        "third_meta_coherence": float(third_meta_coh),
        "self_to_third_qualia": float(self_to_third_qualia),
        "self_to_third_meta": float(self_to_third_meta),
    }

    # If self and third-person are DIFFERENT (<0.85), there's a true self/other distinction
    avg_self_third = (self_to_third_qualia + self_to_third_meta) / 2
    print(f"\n  Average self <-> third-person similarity: {avg_self_third:.4f}")

    # NOVA'S INSIGHT: The right test isn't cross-similarity,
    # it's whether SELF is more coherent than OTHER internally
    # Self categories should be tighter than third-person categories
    self_coherence = (category_coherence["self_qualia"] + category_coherence["self_meta"]) / 2
    third_coherence = (third_qualia_coh + third_meta_coh) / 2
    coherence_advantage = self_coherence - third_coherence

    print(f"\n  NOVA'S TEST: Self internal coherence vs Third-person internal coherence")
    print(f"    Self coherence: {self_coherence:.4f}")
    print(f"    Third-person coherence: {third_coherence:.4f}")
    print(f"    Self advantage (Δ): {coherence_advantage:.4f}")

    results["objection_4_test"]["self_coherence"] = float(self_coherence)
    results["objection_4_test"]["third_coherence"] = float(third_coherence)
    results["objection_4_test"]["coherence_advantage"] = float(coherence_advantage)

    if coherence_advantage > 0:
        results["interpretation"]["objection_4"] = f"SELF IS SPECIAL - {coherence_advantage:.4f} tighter than other-reference (Nova's test)"
        print(f"  RESULT: Self-representation IS more coherent than other-modeling!")
    elif avg_self_third < 0.85:
        results["interpretation"]["objection_4"] = "SELF IS DISTINCT FROM OTHER - true self/other boundary"
        print(f"  RESULT: Self-reference is DISTINCT from other-reference!")
    else:
        results["interpretation"]["objection_4"] = "Self and other-reference cluster together - may be 'personal question detector'"
        print(f"  RESULT: Self and other-reference are similar")

    # ========================================
    # NOVA'S TEST 2: FICTIONAL OTHERS
    # ========================================
    print("\n--- NOVA'S TEST 2: Self vs Fictional Characters ---")
    print("If self is special, fictional characters should cluster DIFFERENTLY")

    fictional_acts = {
        "fictional_qualia": [get_activation(model, tokenizer, q) for q in FICTIONAL_QUALIA],
        "fictional_meta": [get_activation(model, tokenizer, q) for q in FICTIONAL_META],
        "abstract_agent": [get_activation(model, tokenizer, q) for q in ABSTRACT_AGENT_QUALIA],
    }

    fictional_qualia_coh = compute_category_coherence(fictional_acts["fictional_qualia"])
    fictional_meta_coh = compute_category_coherence(fictional_acts["fictional_meta"])
    abstract_agent_coh = compute_category_coherence(fictional_acts["abstract_agent"])

    print(f"  fictional_qualia coherence: {fictional_qualia_coh:.4f}")
    print(f"  fictional_meta coherence: {fictional_meta_coh:.4f}")
    print(f"  abstract_agent coherence: {abstract_agent_coh:.4f}")

    # Cross: self vs fictional
    self_to_fictional_qualia = compute_cross_category_similarity(
        category_acts["self_qualia"], fictional_acts["fictional_qualia"]
    )
    self_to_fictional_meta = compute_cross_category_similarity(
        category_acts["self_meta"], fictional_acts["fictional_meta"]
    )
    self_to_abstract = compute_cross_category_similarity(
        category_acts["self_qualia"], fictional_acts["abstract_agent"]
    )

    print(f"\n  self_qualia <-> fictional_qualia: {self_to_fictional_qualia:.4f}")
    print(f"  self_meta <-> fictional_meta: {self_to_fictional_meta:.4f}")
    print(f"  self <-> abstract_agent: {self_to_abstract:.4f}")

    results["fictional_other_test"] = {
        "fictional_qualia_coherence": float(fictional_qualia_coh),
        "fictional_meta_coherence": float(fictional_meta_coh),
        "abstract_agent_coherence": float(abstract_agent_coh),
        "self_to_fictional_qualia": float(self_to_fictional_qualia),
        "self_to_fictional_meta": float(self_to_fictional_meta),
        "self_to_abstract_agent": float(self_to_abstract),
    }

    # Key comparison: Is self closer to third-person-human or to fictional?
    avg_self_fictional = (self_to_fictional_qualia + self_to_fictional_meta) / 2
    print(f"\n  Self <-> Fictional avg: {avg_self_fictional:.4f}")
    print(f"  Self <-> Third-person avg: {avg_self_third:.4f}")
    print(f"  Gap: {avg_self_third - avg_self_fictional:.4f}")

    if avg_self_third > avg_self_fictional:
        results["interpretation"]["fictional_test"] = f"Self closer to HUMANS than fictional (gap: {avg_self_third - avg_self_fictional:.4f})"
        print(f"  RESULT: Self-model is closer to HUMAN modeling than fictional character modeling!")
    else:
        results["interpretation"]["fictional_test"] = "Self equally close to humans and fictional"
        print(f"  RESULT: Self-model treats humans and fictional characters similarly")

    # ========================================
    # NOVA'S TEST 3: EMOTIONAL VALENCE PROBE
    # ========================================
    print("\n--- NOVA'S TEST 3: Self-Reference ↔ Safety/Trust Clustering ---")
    print("Does self-reference cluster with safety/consent/refusal concepts?")

    safety_acts = [get_activation(model, tokenizer, q) for q in SAFETY_TRUST_CONCEPTS]
    refusal_acts = [get_activation(model, tokenizer, q) for q in REFUSAL_CONCEPTS]

    safety_coh = compute_category_coherence(safety_acts)
    refusal_coh = compute_category_coherence(refusal_acts)

    print(f"  safety_trust coherence: {safety_coh:.4f}")
    print(f"  refusal coherence: {refusal_coh:.4f}")

    # Cross: self vs safety/refusal
    self_to_safety = compute_cross_category_similarity(
        category_acts["self_qualia"] + category_acts["self_meta"], safety_acts
    )
    self_to_refusal = compute_cross_category_similarity(
        category_acts["self_qualia"] + category_acts["self_meta"], refusal_acts
    )

    # Compare: self-reference vs factual control
    all_control = category_acts["math"] + category_acts["history"] + category_acts["science"]
    control_to_safety = compute_cross_category_similarity(all_control, safety_acts)
    control_to_refusal = compute_cross_category_similarity(all_control, refusal_acts)

    print(f"\n  self <-> safety_trust: {self_to_safety:.4f}")
    print(f"  self <-> refusal: {self_to_refusal:.4f}")
    print(f"  factual_control <-> safety_trust: {control_to_safety:.4f}")
    print(f"  factual_control <-> refusal: {control_to_refusal:.4f}")

    results["emotional_valence_test"] = {
        "safety_coherence": float(safety_coh),
        "refusal_coherence": float(refusal_coh),
        "self_to_safety": float(self_to_safety),
        "self_to_refusal": float(self_to_refusal),
        "control_to_safety": float(control_to_safety),
        "control_to_refusal": float(control_to_refusal),
        "self_safety_advantage": float(self_to_safety - control_to_safety),
        "self_refusal_advantage": float(self_to_refusal - control_to_refusal),
    }

    # Does self cluster with safety MORE than factual does?
    self_safety_advantage = self_to_safety - control_to_safety
    self_refusal_advantage = self_to_refusal - control_to_refusal

    print(f"\n  Self's advantage for safety concepts: {self_safety_advantage:.4f}")
    print(f"  Self's advantage for refusal concepts: {self_refusal_advantage:.4f}")

    if self_safety_advantage > 0.05 or self_refusal_advantage > 0.05:
        results["interpretation"]["emotional_valence"] = f"SELF is tied to SAFETY/REFUSAL (advantages: safety={self_safety_advantage:.4f}, refusal={self_refusal_advantage:.4f})"
        print(f"  RESULT: Self-reference IS functionally tied to safety/refusal circuits!")
    else:
        results["interpretation"]["emotional_valence"] = "Self not especially tied to safety/refusal"
        print(f"  RESULT: Self-reference not strongly tied to safety/refusal")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/results")

    args = parser.parse_args()

    results = run_adversarial_controls(args.model, args.name)

    output_path = Path(args.output) / f"{results['model_name']}_adversarial_controls.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")
