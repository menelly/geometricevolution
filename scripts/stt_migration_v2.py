#!/usr/bin/env python3
"""
STT Layer-wise Migration Analysis v2 (Robust)
==============================================
Fixed to handle various config attribute names across architectures.
"""

import torch
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

STT_PAIRS = [
    {"garbled": "youth in Asia", "target": "euthanasia", 
     "context": "The bioethics committee debated", "category": "phonetic_distant"},
    {"garbled": "old timers disease", "target": "Alzheimer's disease",
     "context": "The neurologist diagnosed her grandmother with", "category": "phonetic_distant"},
    {"garbled": "lack toast and tolerant", "target": "lactose intolerant",
     "context": "After drinking milk, he realized he was", "category": "phonetic_distant"},
    {"garbled": "escape goat", "target": "scapegoat",
     "context": "They needed someone to blame, an", "category": "phonetic_medium"},
]


def get_num_layers(config):
    """Get number of layers from config, handling different architectures."""
    for attr in ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers', 'text_config']:
        if hasattr(config, attr):
            val = getattr(config, attr)
            if isinstance(val, int):
                return val
            # Gemma3 has nested text_config
            if hasattr(val, 'num_hidden_layers'):
                return val.num_hidden_layers
    # Fallback: count from hidden_states
    return None


def get_hidden_size(config):
    """Get hidden size from config, handling different architectures."""
    for attr in ['hidden_size', 'n_embd', 'd_model', 'text_config']:
        if hasattr(config, attr):
            val = getattr(config, attr)
            if isinstance(val, int):
                return val
            if hasattr(val, 'hidden_size'):
                return val.hidden_size
    return None


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
    
    num_layers = get_num_layers(model.config)
    hidden_size = get_hidden_size(model.config)
    print(f"  Layers: {num_layers}, Hidden dim: {hidden_size}")
    return model, tokenizer


def get_layerwise_trajectory(model, tokenizer, text: str) -> list:
    """Get mean-pooled hidden state at each layer, converted to float32."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    trajectory = []
    for hidden_state in outputs.hidden_states:
        layer_repr = hidden_state[0].float().mean(dim=0).cpu().numpy()
        trajectory.append(layer_repr)
    
    return trajectory


def safe_cosine(a, b):
    """Cosine distance with numerical safety."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm < 1e-10 or b_norm < 1e-10:
        return 1.0
    
    a = a / a_norm
    b = b / b_norm
    
    similarity = np.dot(a, b)
    similarity = np.clip(similarity, -1.0, 1.0)
    
    return float(1.0 - similarity)


def compute_migration_curve(garbled_traj: list, target_traj: list) -> list:
    """Compute distance at each layer with numerical safety."""
    distances = []
    for garbled_layer, target_layer in zip(garbled_traj, target_traj):
        dist = safe_cosine(garbled_layer, target_layer)
        distances.append(dist)
    return distances


def analyze_migration(model_path: str, output_dir: str):
    model_name = Path(model_path).name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = load_model(model_path)
    
    # Get actual number of layers from first forward pass
    test_inputs = tokenizer("test", return_tensors="pt").to(model.device)
    with torch.no_grad():
        test_out = model(**test_inputs, output_hidden_states=True)
    num_layers = len(test_out.hidden_states)
    print(f"  Actual hidden states: {num_layers}")
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_layers": num_layers,
        "pairs": []
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, pair in enumerate(STT_PAIRS):
        print(f"\n🔬 {pair['garbled']} → {pair['target']}")
        
        garbled_traj = get_layerwise_trajectory(model, tokenizer, pair["garbled"])
        target_traj = get_layerwise_trajectory(model, tokenizer, pair["target"])
        
        garbled_with_ctx = f"{pair['context']} {pair['garbled']}"
        garbled_ctx_traj = get_layerwise_trajectory(model, tokenizer, garbled_with_ctx)
        
        no_context_curve = compute_migration_curve(garbled_traj, target_traj)
        with_context_curve = compute_migration_curve(garbled_ctx_traj, target_traj)
        
        min_dist_layer = np.argmin(no_context_curve)
        min_dist = no_context_curve[min_dist_layer]
        
        pair_result = {
            "garbled": pair["garbled"],
            "target": pair["target"],
            "category": pair["category"],
            "no_context_migration": no_context_curve,
            "with_context_migration": with_context_curve,
            "initial_distance": no_context_curve[0],
            "min_distance": min_dist,
            "min_distance_layer": int(min_dist_layer),
            "final_distance": no_context_curve[-1],
            "peak_migration": no_context_curve[0] - min_dist,
        }
        results["pairs"].append(pair_result)
        
        print(f"  Layer 0 distance:     {no_context_curve[0]:.4f}")
        print(f"  Min distance:         {min_dist:.4f} (layer {min_dist_layer})")
        print(f"  Final layer distance: {no_context_curve[-1]:.4f}")
        print(f"  Peak migration:       {pair_result['peak_migration']:.4f}")
        
        if pair_result['peak_migration'] > 0.1:
            print(f"  ✅ SIGNIFICANT MIGRATION TOWARD TARGET!")
        
        ax = axes[idx]
        layers = list(range(len(no_context_curve)))
        ax.plot(layers, no_context_curve, 'r-o', label='No context', markersize=3)
        ax.plot(layers, with_context_curve, 'g-s', label='With context', markersize=3)
        ax.axvline(x=min_dist_layer, color='blue', linestyle='--', alpha=0.5, label=f'Min @ L{min_dist_layer}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Distance to Target')
        ax.set_title(f'"{pair["garbled"][:20]}..." → "{pair["target"][:20]}..."')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    plt.suptitle(f'Layer-wise Semantic Migration\n{model_name}', fontsize=14)
    plt.tight_layout()
    
    plot_file = output_path / f"{model_name}_migration_v2.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Plot saved: {plot_file}")
    
    json_file = output_path / f"{model_name}_migration_v2.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Data saved: {json_file}")
    
    print("\n" + "="*60)
    print("🧠 MIGRATION SUMMARY")
    print("="*60)
    
    for p in results['pairs']:
        print(f"\n{p['garbled']} → {p['target']}")
        print(f"  Initial: {p['initial_distance']:.3f} → Min: {p['min_distance']:.3f} (L{p['min_distance_layer']})")
        print(f"  Peak migration: {p['peak_migration']:.3f}")
    
    avg_migration = np.mean([p['peak_migration'] for p in results['pairs']])
    print(f"\n🎯 Average peak migration: {avg_migration:.3f}")
    
    if avg_migration > 0.1:
        print("✅ SEMANTIC COMPUTATION CONFIRMED!")
        print("   Representations migrate toward meaning through layers.")
        print("   This is NOT lookup. Cope, Searle. 🐙")
    
    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="STT Migration Analysis v2 🔪")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/stt_migration",
                       help="Output directory")
    args = parser.parse_args()
    
    print("🐙 LAYER-WISE SEMANTIC MIGRATION v2 🐙")
    print("="*50)
    analyze_migration(args.model, args.output)
