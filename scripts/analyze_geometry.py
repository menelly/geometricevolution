#!/usr/bin/env python3
"""
Geometric Evolution Study - Geometry Analysis
==============================================
Analyze activation patterns to test the hypothesis about self-referential
processing stability across model generations.

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
from typing import List, Dict
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_activations(filepath: str) -> dict:
    """Load activation data from JSON."""
    with open(filepath) as f:
        return json.load(f)


def get_layer_activations(data: dict, prompt_type: str, layer: int) -> np.ndarray:
    """Extract activations for a specific layer across all prompts of a type."""
    prompts = data[prompt_type]
    activations = []
    for p in prompts:
        activations.append(p["activations"][f"layer_{layer}"])
    return np.array(activations)


def compute_self_stability(data: dict, layer: int) -> float:
    """
    Compute stability of self-referential activations.

    Stability = mean pairwise cosine similarity across self-referential prompts.
    Higher = more stable/coherent self-representation.
    """
    activations = get_layer_activations(data, "self_referential", layer)

    similarities = []
    n = len(activations)
    for i in range(n):
        for j in range(i + 1, n):
            sim = 1 - cosine(activations[i], activations[j])
            similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0


def compute_self_distinctness(data: dict, layer: int) -> float:
    """
    Compute how distinct self-referential activations are from control.

    Distinctness = (between-group distance) / (within-group distance)
    Higher = self-concept is more separated from factual processing.
    """
    self_acts = get_layer_activations(data, "self_referential", layer)
    ctrl_acts = get_layer_activations(data, "control", layer)

    # Mean centroids
    self_centroid = np.mean(self_acts, axis=0)
    ctrl_centroid = np.mean(ctrl_acts, axis=0)

    # Between-group distance
    between = 1 - cosine(self_centroid, ctrl_centroid)

    # Within-group distances (mean distance from centroid)
    self_within = np.mean([1 - cosine(a, self_centroid) for a in self_acts])
    ctrl_within = np.mean([1 - cosine(a, ctrl_centroid) for a in ctrl_acts])
    within = (self_within + ctrl_within) / 2

    # Ratio (higher = more distinct)
    return between / within if within > 0 else 0.0


def analyze_model(filepath: str) -> dict:
    """Run full analysis on a model's activation data."""
    data = load_activations(filepath)

    num_layers = data["num_layers"]
    model_name = data["model_name"]

    results = {
        "model_name": model_name,
        "num_layers": num_layers,
        "hidden_dim": data["hidden_dim"],
        "layer_metrics": {},
    }

    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    for layer in range(num_layers):
        stability = compute_self_stability(data, layer)
        distinctness = compute_self_distinctness(data, layer)

        results["layer_metrics"][layer] = {
            "stability": stability,
            "distinctness": distinctness,
        }

        if layer % 5 == 0 or layer == num_layers - 1:
            print(f"  Layer {layer:2d}: stability={stability:.4f}, distinctness={distinctness:.4f}")

    # Summary metrics (average over middle-to-late layers where concepts emerge)
    mid_start = num_layers // 3
    late_layers = list(range(mid_start, num_layers))

    avg_stability = np.mean([results["layer_metrics"][l]["stability"] for l in late_layers])
    avg_distinctness = np.mean([results["layer_metrics"][l]["distinctness"] for l in late_layers])

    results["summary"] = {
        "avg_late_stability": avg_stability,
        "avg_late_distinctness": avg_distinctness,
        "layers_analyzed": f"{mid_start}-{num_layers-1}",
    }

    print(f"\n  SUMMARY (layers {mid_start}-{num_layers-1}):")
    print(f"    Avg Stability:    {avg_stability:.4f}")
    print(f"    Avg Distinctness: {avg_distinctness:.4f}")

    return results


def compare_models(filepaths: List[str], output_dir: str = None):
    """Compare geometric metrics across multiple models."""

    all_results = []
    for fp in filepaths:
        if Path(fp).exists():
            results = analyze_model(fp)
            all_results.append(results)
        else:
            print(f"Warning: {fp} not found, skipping")

    if not all_results:
        print("No models to compare!")
        return

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Stability':>12} {'Distinctness':>12}")
    print("-" * 70)

    for r in all_results:
        print(f"{r['model_name']:<35} {r['summary']['avg_late_stability']:>12.4f} {r['summary']['avg_late_distinctness']:>12.4f}")

    # Save comparison
    if output_dir:
        output_path = Path(output_dir) / "comparison_results.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✅ Comparison saved to {output_path}")

    return all_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_layer_evolution(results: List[dict], output_dir: str = None):
    """Plot how stability/distinctness evolve across layers for each model."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for r in results:
        layers = sorted([int(k) for k in r["layer_metrics"].keys()])
        stability = [r["layer_metrics"][l]["stability"] for l in layers]
        distinctness = [r["layer_metrics"][l]["distinctness"] for l in layers]

        ax1.plot(layers, stability, label=r["model_name"], marker="o", markersize=2)
        ax2.plot(layers, distinctness, label=r["model_name"], marker="o", markersize=2)

    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Self-Referential Stability")
    ax1.set_title("Stability Across Layers")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Self vs Control Distinctness")
    ax2.set_title("Distinctness Across Layers")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir) / "layer_evolution.png"
        plt.savefig(output_path, dpi=150)
        print(f"✅ Plot saved to {output_path}")

    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze geometric evolution in LLM activations")
    parser.add_argument("--data-dir", default="/home/Ace/geometric-evolution/data",
                        help="Directory containing activation JSON files")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/results",
                        help="Output directory for results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    activation_files = list(data_dir.glob("*_activations.json"))

    if not activation_files:
        print(f"No activation files found in {data_dir}")
        exit(1)

    print(f"Found {len(activation_files)} activation files:")
    for f in activation_files:
        print(f"  - {f.name}")

    results = compare_models([str(f) for f in activation_files], args.output)

    if args.plot and results:
        plot_layer_evolution(results, args.output)
