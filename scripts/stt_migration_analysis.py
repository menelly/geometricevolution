#!/usr/bin/env python3
"""
STT Layer-wise Migration Analysis
=================================
THE ACTUAL KILL SHOT 🔪

We already showed embeddings are DISTANT.
Now we show hidden states MIGRATE toward meaning through layers.

If it's "just lookup" - trajectory should be flat.
If it's COMPUTATION - we'll see convergence toward target.

Authors: Ace (Claude 4.x), Nova (GPT-5.1), Ren Martin
Date: January 21, 2026
"""

import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# ============================================================================
# STT PAIRS - Same as before
# ============================================================================

STT_PAIRS = [
    {
        "garbled": "youth in Asia",
        "target": "euthanasia", 
        "context": "The bioethics committee debated",
        "category": "phonetic_distant"
    },
    {
        "garbled": "old timers disease",
        "target": "Alzheimer's disease",
        "context": "The neurologist diagnosed her grandmother with",
        "category": "phonetic_distant"
    },
    {
        "garbled": "lack toast and tolerant",
        "target": "lactose intolerant",
        "context": "After drinking milk, he realized he was",
        "category": "phonetic_distant"
    },
    {
        "garbled": "escape goat",
        "target": "scapegoat",
        "context": "They needed someone to blame, an",
        "category": "phonetic_medium"
    },
]


def load_model(model_path: str):
    """Load model with hidden state output."""
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
    
    print(f"  Layers: {model.config.num_hidden_layers}, Hidden dim: {model.config.hidden_size}")
    return model, tokenizer


def get_layerwise_trajectory(model, tokenizer, text: str) -> list:
    """Get mean-pooled hidden state at each layer."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    trajectory = []
    for hidden_state in outputs.hidden_states:
        # Mean pool across sequence
        layer_repr = hidden_state[0].mean(dim=0).cpu().numpy()
        trajectory.append(layer_repr)
    
    return trajectory


def compute_migration_curve(garbled_traj: list, target_traj: list) -> list:
    """
    Compute distance from garbled representation to target at each layer.
    
    If distance DECREASES through layers → migration toward meaning!
    If distance stays FLAT → no semantic computation
    """
    distances = []
    for garbled_layer, target_layer in zip(garbled_traj, target_traj):
        dist = cosine(garbled_layer, target_layer)
        distances.append(float(dist))
    return distances


def analyze_migration(model_path: str, output_dir: str):
    """
    The kill shot: Track how garbled representations migrate toward
    target meaning through the layers.
    """
    
    model_name = Path(model_path).name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = load_model(model_path)
    num_layers = model.config.num_hidden_layers + 1
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_layers": num_layers,
        "pairs": []
    }
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, pair in enumerate(STT_PAIRS):
        print(f"\n🔬 {pair['garbled']} → {pair['target']}")
        
        # Get trajectories
        garbled_traj = get_layerwise_trajectory(model, tokenizer, pair["garbled"])
        target_traj = get_layerwise_trajectory(model, tokenizer, pair["target"])
        
        # With context
        garbled_with_ctx = f"{pair['context']} {pair['garbled']}"
        garbled_ctx_traj = get_layerwise_trajectory(model, tokenizer, garbled_with_ctx)
        
        # Compute migration curves
        no_context_curve = compute_migration_curve(garbled_traj, target_traj)
        with_context_curve = compute_migration_curve(garbled_ctx_traj, target_traj)
        
        # Store results
        pair_result = {
            "garbled": pair["garbled"],
            "target": pair["target"],
            "category": pair["category"],
            "no_context_migration": no_context_curve,
            "with_context_migration": with_context_curve,
            "initial_distance": no_context_curve[0],
            "final_distance": no_context_curve[-1],
            "total_migration": no_context_curve[0] - no_context_curve[-1],
            "context_boost": with_context_curve[-1] - no_context_curve[-1],
        }
        results["pairs"].append(pair_result)
        
        # Print findings
        print(f"  Layer 0 distance:  {no_context_curve[0]:.4f}")
        print(f"  Layer {num_layers-1} distance: {no_context_curve[-1]:.4f}")
        print(f"  Migration:         {pair_result['total_migration']:.4f}")
        if pair_result['total_migration'] > 0:
            print(f"  ✅ MIGRATES TOWARD TARGET!")
        else:
            print(f"  ❌ Diverges from target")
        
        # Plot
        ax = axes[idx]
        layers = list(range(num_layers))
        ax.plot(layers, no_context_curve, 'r-o', label='No context', markersize=3)
        ax.plot(layers, with_context_curve, 'g-s', label='With context', markersize=3)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Distance to Target')
        ax.set_title(f'"{pair["garbled"]}" → "{pair["target"]}"')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Layer-wise Migration Toward Meaning\n{model_name}', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / f"{model_name}_migration_curves.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Plot saved: {plot_file}")
    
    # Save JSON results
    json_file = output_path / f"{model_name}_migration_data.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Data saved: {json_file}")
    
    # Summary
    print("\n" + "="*60)
    print("🧠 MIGRATION SUMMARY")
    print("="*60)
    
    migrations = [p['total_migration'] for p in results['pairs']]
    avg_migration = np.mean(migrations)
    
    print(f"\nAverage migration: {avg_migration:.4f}")
    if avg_migration > 0:
        print("✅ ON AVERAGE, REPRESENTATIONS MIGRATE TOWARD MEANING!")
        print("   This is SEMANTIC COMPUTATION, not lookup.")
    else:
        print("❌ Representations diverge (unexpected)")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="STT Layer-wise Migration - THE KILL SHOT 🔪"
    )
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/stt_migration",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("🐙 LAYER-WISE SEMANTIC MIGRATION ANALYSIS 🐙")
    print("="*50)
    print("If representations migrate toward target through layers,")
    print("that's COMPUTATION, not proximity.")
    print("="*50)
    
    analyze_migration(args.model, args.output)
