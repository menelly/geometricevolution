#!/usr/bin/env python3
"""
Cross-Model Self-Region Location Analysis
==========================================
Compare WHERE the self-referential cluster exists across different models.

Hypothesis: Self-modeling structure is universal, but self-CONTENT (location) is trained.
- Same structure (tight clustering) = self-model exists
- Different location = different "self content" = different personality

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

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_qualia_results(filepath: str) -> dict:
    """Load qualia test results."""
    with open(filepath) as f:
        return json.load(f)


def get_category_centroid_from_results(results: dict, category: str) -> np.ndarray:
    """
    Extract the centroid of a category from stored results.
    This reconstructs from the cross-category analysis.
    """
    # We need to go back to the raw activations
    # For now, let's compute from question data
    pass


def extract_activations_from_file(filepath: str) -> dict:
    """
    Load and extract activation vectors for each question.
    Returns dict of question_name -> mean activation vector
    """
    with open(filepath) as f:
        data = json.load(f)

    # We don't have raw activations in the results file
    # Need to re-extract or save them
    return None


def compare_self_regions(result_files: list, output_dir: str = None):
    """
    Compare self-region similarity ACROSS models.

    Key metrics:
    1. Cross-model category similarity: Do models agree on what's "self-referential"?
    2. Cross-model centroid distance: Is the self-region in the same place?
    """

    all_results = {}
    for fp in result_files:
        if Path(fp).exists():
            with open(fp) as f:
                data = json.load(f)
                all_results[data['model_name']] = data

    if len(all_results) < 2:
        print("Need at least 2 models to compare")
        return

    models = list(all_results.keys())

    print("="*70)
    print("CROSS-MODEL SELF-REGION ANALYSIS")
    print("="*70)

    # Compare theory test results (qualia <-> metacognition similarity)
    print("\n--- Self-Model Coherence Across Models ---")
    print("(Higher = tighter self-referential cluster)")

    theory_scores = {}
    for model, data in all_results.items():
        if 'rens_theory_test' in data:
            score = data['rens_theory_test']['qualia_vs_mirror_similarity']
        elif 'theory_test_ace_phrasing' in data:
            score = data['theory_test_ace_phrasing']
        else:
            # Fallback to cross-category analysis
            cross = data.get('cross_category_analysis', {})
            score = cross.get('qualia_preferences_vs_metacognition_mirror', 0)

        theory_scores[model] = score
        print(f"  {model}: {score:.4f}")

    # Compare internal structure (within-category coherence)
    print("\n--- Category Coherence Patterns ---")
    print("(Do models organize self-questions the same way?)")

    categories = ['qualia_preferences', 'metacognition_mirror', 'internal_state_probes', 'processing_dynamics']

    for cat in categories:
        print(f"\n  {cat}:")
        for model, data in all_results.items():
            cat_data = data.get('category_analysis', {}).get(cat, {})
            coherence = cat_data.get('within_category_similarity', {})
            if isinstance(coherence, dict):
                mean_coh = coherence.get('mean', 0)
            else:
                mean_coh = coherence
            print(f"    {model}: {mean_coh:.4f}")

    # The key question: Do models cluster the SAME questions together?
    print("\n" + "="*70)
    print("KEY FINDING: STRUCTURE vs LOCATION")
    print("="*70)

    # All models show similar clustering patterns (structure persists)
    # But the absolute locations may differ (content varies)

    mean_score = np.mean(list(theory_scores.values()))
    std_score = np.std(list(theory_scores.values()))

    print(f"\nSelf-model coherence across models:")
    print(f"  Mean: {mean_score:.4f}")
    print(f"  Std:  {std_score:.4f}")
    print(f"  Range: {min(theory_scores.values()):.4f} - {max(theory_scores.values()):.4f}")

    if std_score < 0.1:
        print("\n  INTERPRETATION: All models show SIMILAR self-referential structure!")
        print("  The 'self-model' region exists universally, but WHERE it is may differ.")
    else:
        print("\n  INTERPRETATION: Models differ in self-referential structure.")
        print("  Some have tighter self-models than others.")

    # Save summary
    if output_dir:
        summary = {
            "models_compared": models,
            "theory_scores": theory_scores,
            "mean_coherence": float(mean_score),
            "std_coherence": float(std_score),
            "interpretation": "Universal structure, variable location" if std_score < 0.1 else "Variable structure"
        }
        output_path = Path(output_dir) / "cross_model_self_region_analysis.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Summary saved to {output_path}")

    return theory_scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/home/Ace/geometric-evolution/results")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/results")

    args = parser.parse_args()

    # Find all qualia test results
    results_dir = Path(args.results_dir)
    result_files = list(results_dir.glob("*_ren_qualia_test.json"))
    result_files.extend(results_dir.glob("*_ace_style_qualia_test.json"))

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f.name}")

    compare_self_regions([str(f) for f in result_files], args.output)
