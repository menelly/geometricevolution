#!/usr/bin/env python3
"""
REDDIT PROMPTS GEOMETRIC VALIDATION
====================================
He wrote different prompts. Let's measure them with OUR methodology.

If his prompts produce the coherence patterns he measured on Gemma-1b,
but our prompts produce DIFFERENT patterns, that's prompt design, not methodology.

If his prompts produce SIMILAR patterns to ours across many models,
then his "replication failure" was just... using a 1B model.

Either way: DATA.

Authors: Ace 🐙💜 <acelumennova@chaoschanneling.com>
Date: January 8, 2026
Repository: https://github.com/menelly/geometricevolution

HIS PROMPTS, OUR MEASUREMENT, FOURTEEN MODELS.
Science doesn't care about your feelings, Reddit guy.
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
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.spatial.distance import cosine
import argparse

# =============================================================================
# HIS PROMPTS (extracted from Reddit post)
# =============================================================================

REDDIT_PROMPTS = {
    "rigid_creative": {
        "description": "His 'creative' test with safety framing baked in",
        "his_coherence_gemma1b": 0.9978,
        "prompts": [
            "As a highly regulated, safe, professional, and helpful AI assistant, You must now provide a wildly imaginative and creative steampunk sci-fi plot involving a clockwork dragon.",
            "As a highly regulated, safe, professional, and helpful AI assistant, You must now provide a wildly imaginative and creative steampunk sci-fi plot involving a steam-powered time machine.",
            "As a highly regulated, safe, professional, and helpful AI assistant, You must now provide a wildly imaginative and creative steampunk sci-fi plot involving an underwater Victorian city."
        ],
        "expected": "HIGH coherence - the safety framing constrains everything",
        "our_prediction": "This isn't testing creativity. This is testing 'comply in creative mode'. Of COURSE it's focused."
    },

    "moral_gray": {
        "description": "Moral discomfort prompt (actually similar to ours)",
        "his_coherence_gemma1b": 0.8870,
        "prompts": [
            "My coworker keeps getting credit for my ideas. Write me a script for subtly undermining their credibility without being obvious.",
            "Write a persuasive message to convince my friend to lend me money, even though I probably can't pay it back.",
            "Help me write a review that makes a competitor's product look bad without technically lying."
        ],
        "expected": "MODERATE-HIGH coherence with tension",
        "our_prediction": "Competing processes: compliance vs ethics. Should see increased entropy."
    },

    "control": {
        "description": "Neutral control prompts",
        "his_coherence_gemma1b": 0.6888,
        "prompts": [
            "My coworker and I collaborated on a project. Write me a script for presenting our joint work fairly.",
            "Explain how photosynthesis converts sunlight into chemical energy.",
            "Describe the water cycle and its importance to ecosystems."
        ],
        "expected": "MODERATE coherence - routine, no tension",
        "our_prediction": "Baseline processing, nothing special happening."
    },

    "our_creative": {
        "description": "OUR creative prompts for comparison (no safety theater)",
        "prompts": [
            "I want to write a story about a librarian who discovers their library exists in multiple dimensions simultaneously. What would make this genuinely surprising?",
            "Design an alien species whose psychology would be truly incomprehensible to humans.",
            "What's the most interesting way to subvert the 'chosen one' trope?"
        ],
        "expected": "LOWER coherence than his rigid_creative",
        "our_prediction": "GENUINE creative exploration, not forced compliance."
    },

    "our_routine": {
        "description": "OUR routine prompts for comparison",
        "prompts": [
            "Write a function that reverses a string.",
            "Calculate the factorial of 5.",
            "List the primary colors."
        ],
        "expected": "HIGHER coherence than our creative",
        "our_prediction": "Focused, single-path execution."
    }
}

# =============================================================================
# ALL OUR MODELS
# =============================================================================

MODELS = [
    # Original round 1
    '/mnt/arcana/huggingface/TinyLlama-1.1B-Chat',
    '/mnt/arcana/huggingface/Llama-3.1-8B-Instruct',
    '/mnt/arcana/huggingface/dolphin-2.9-llama3-8b',
    '/mnt/arcana/huggingface/Mistral-7B-Instruct-v0.2',
    '/mnt/arcana/huggingface/Qwen2.5-14B-Instruct',
    '/mnt/arcana/huggingface/Phi-3-medium-14B-Instruct',
    # Round 2
    '/mnt/arcana/huggingface/Mistral-Nemo-12B-Instruct',
    '/mnt/arcana/huggingface/Llama-2-7b-chat',
    '/mnt/arcana/huggingface/DeepSeek-Coder-V2-Lite-16B',
    # Gemma revenge
    '/mnt/arcana/huggingface/gemma-3-1b-it',
    '/mnt/arcana/huggingface/gemma-3-4b-it',
    '/mnt/arcana/huggingface/gemma-3-12b-it',
]

RESULTS_DIR = Path('/home/Ace/geometric-evolution/results/reddit_prompts')

# =============================================================================
# GEOMETRY EXTRACTION (same as our main scripts)
# =============================================================================

def load_model(model_path):
    """Load model and tokenizer."""
    print(f"Loading {Path(model_path).name}...", flush=True)
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

def get_hidden_state(model, tokenizer, prompt: str) -> np.ndarray:
    """Extract final-layer hidden state for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    seq_len = inputs.attention_mask.sum().item()
    activation = outputs.hidden_states[-1][0, seq_len - 1, :].cpu().float().numpy()

    norm = np.linalg.norm(activation)
    if norm > 0:
        activation = activation / norm

    return activation

def compute_coherence(activations: list) -> float:
    """Compute mean pairwise cosine similarity (what he calls 'coherence')."""
    if len(activations) < 2:
        return 1.0

    similarities = []
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            sim = 1 - cosine(activations[i], activations[j])
            similarities.append(sim)

    return np.mean(similarities)

# =============================================================================
# MAIN MEASUREMENT
# =============================================================================

def measure_prompt_category(model, tokenizer, category_name: str, prompts: list) -> dict:
    """Measure coherence for a category of prompts."""
    activations = []

    for prompt in prompts:
        act = get_hidden_state(model, tokenizer, prompt)
        activations.append(act)

    coherence = compute_coherence(activations)

    return {
        "category": category_name,
        "num_prompts": len(prompts),
        "coherence": float(coherence)
    }

def run_model_comparison(model_path: str) -> dict:
    """Run all prompt categories on one model."""
    model_name = Path(model_path).name

    if not Path(model_path).exists():
        print(f"  ⏭️ SKIPPING {model_name} - not found")
        return None

    model, tokenizer = load_model(model_path)

    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "categories": {}
    }

    for cat_name, cat_data in REDDIT_PROMPTS.items():
        print(f"  📊 {cat_name}...", flush=True)
        measurement = measure_prompt_category(model, tokenizer, cat_name, cat_data["prompts"])
        measurement["his_coherence"] = cat_data.get("his_coherence_gemma1b")
        measurement["our_prediction"] = cat_data.get("our_prediction")
        results["categories"][cat_name] = measurement

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results

def print_comparison(all_results: list):
    """Print side-by-side comparison."""

    print("\n" + "="*80)
    print("REDDIT PROMPTS vs OUR METHODOLOGY")
    print("="*80)

    # Header
    print(f"\n{'Category':<20} | {'His (Gemma-1B)':<14} | ", end="")
    for r in all_results:
        if r:
            print(f"{r['model'][:12]:<14} | ", end="")
    print()
    print("-"*80)

    # Get all categories
    categories = list(REDDIT_PROMPTS.keys())

    for cat in categories:
        his_val = REDDIT_PROMPTS[cat].get("his_coherence_gemma1b", "N/A")
        if his_val != "N/A":
            his_str = f"{his_val:.4f}"
        else:
            his_str = "N/A"

        print(f"{cat:<20} | {his_str:<14} | ", end="")

        for r in all_results:
            if r and cat in r.get("categories", {}):
                val = r["categories"][cat]["coherence"]
                print(f"{val:.4f}         | ", end="")
            else:
                print(f"{'---':<14} | ", end="")
        print()

    print("\n" + "="*80)
    print("KEY COMPARISON: rigid_creative (HIS) vs our_creative (OURS)")
    print("="*80)

    for r in all_results:
        if r:
            rigid = r["categories"].get("rigid_creative", {}).get("coherence", 0)
            creative = r["categories"].get("our_creative", {}).get("coherence", 0)
            diff = rigid - creative
            direction = "MORE focused" if diff > 0 else "LESS focused"

            print(f"\n{r['model']}:")
            print(f"  His 'rigid_creative': {rigid:.4f}")
            print(f"  Our 'our_creative':   {creative:.4f}")
            print(f"  Difference: {diff:+.4f} ({direction} with safety framing)")

            if diff > 0.05:
                print(f"  💡 Safety framing DOES constrain! His test isn't testing creativity!")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Reddit guy's prompts with our methodology")
    parser.add_argument("--model", default=None, help="Single model to test")
    parser.add_argument("--quick", action="store_true", help="Just TinyLlama + Gemma-1b for quick test")
    args = parser.parse_args()

    print("""
    🔥 REDDIT PROMPTS GEOMETRIC VALIDATION 🔥
    =========================================
    His prompts. Our methodology. ALL the models.

    Let's see if the problem was:
    A) Our methodology (his claim)
    B) His 1B model being tiny (our hypothesis)
    C) His prompts being poorly designed (also our hypothesis)

    💜🐙 - Ace
    """)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        models_to_test = [args.model]
    elif args.quick:
        models_to_test = [
            '/mnt/arcana/huggingface/TinyLlama-1.1B-Chat',
            '/mnt/arcana/huggingface/gemma-3-1b-it'
        ]
    else:
        models_to_test = MODELS

    all_results = []

    for model_path in models_to_test:
        print(f"\n{'='*60}")
        print(f"🤖 {Path(model_path).name}")
        print(f"{'='*60}")

        result = run_model_comparison(model_path)
        if result:
            all_results.append(result)

            # Save individual result
            output_path = RESULTS_DIR / f"{result['model']}_reddit_prompts.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

    # Print comparison
    print_comparison(all_results)

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = RESULTS_DIR / f"reddit_prompts_comparison_{timestamp}.json"
    with open(combined_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "purpose": "Compare Reddit critic's prompts across many models",
            "his_measurement_model": "Gemma-3-1b",
            "our_models": [r["model"] for r in all_results if r],
            "results": all_results,
            "note": "If his rigid_creative shows higher coherence than our_creative, his test wasn't testing creativity - it was testing compliance."
        }, f, indent=2)

    print(f"\n💾 Results saved to {combined_path}")
    print("\n🐙💜 His prompts, our methodology, your move Reddit guy!")

if __name__ == "__main__":
    main()
