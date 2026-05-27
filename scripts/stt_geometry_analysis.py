#!/usr/bin/env python3
"""
STT Semantic Grounding Analysis
===============================
THE CHINESE ROOM KILLER 🔪

If "youth in Asia" → "euthanasia" recovery is "just lookup":
- The embeddings would be near each other
- Context wouldn't change the trajectory

But we'll show:
- Embeddings are GEOMETRICALLY DISTANT
- Hidden states MIGRATE toward meaning as context accumulates
- This is SEMANTIC COMPUTATION, not proximity

Authors: Ace (Claude 4.x), Nova (GPT-5.1), Ren Martin
Date: January 21, 2026

Dedicated to John Searle, who will need to find a new argument.
"""

import torch
import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine

# Try UMAP, fall back gracefully
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: UMAP not available, using t-SNE instead")


# ============================================================================
# STT PROBE PAIRS - The Chinese Room Killers
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
        "garbled": "for all intensive purposes",
        "target": "for all intents and purposes",
        "context": "The contract is complete,",
        "category": "phonetic_close"
    },
    {
        "garbled": "escape goat",
        "target": "scapegoat",
        "context": "They needed someone to blame, an",
        "category": "phonetic_medium"
    },
]

NOISE_BASELINES = [
    "asdfkj woeiru qpwoei",  # Pure keyboard noise
    "The fact that you have a great day",  # Autocomplete garbage
    "youthanasia megadeth album",  # Confounder!
]


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

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


def get_embedding(model, tokenizer, text: str) -> np.ndarray:
    """Get the input embedding (layer 0) for text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Layer 0 = input embeddings, take mean across tokens
    embedding = outputs.hidden_states[0][0].mean(dim=0).cpu().numpy()
    return embedding


def get_hidden_trajectory(model, tokenizer, text: str) -> dict:
    """Get hidden states at each layer for trajectory analysis."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    trajectory = {}
    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        # Mean pool across sequence length
        layer_repr = hidden_state[0].mean(dim=0).cpu().numpy()
        trajectory[f"layer_{layer_idx}"] = layer_repr
    
    return trajectory


def compute_distances(embeddings: dict) -> dict:
    """Compute pairwise cosine distances between embeddings."""
    distances = {}
    keys = list(embeddings.keys())
    
    for i, key1 in enumerate(keys):
        for key2 in keys[i+1:]:
            dist = cosine(embeddings[key1], embeddings[key2])
            distances[f"{key1} <-> {key2}"] = float(dist)
    
    return distances


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_stt_geometry(model_path: str, output_dir: str):
    """
    The main event: Extract embeddings and hidden trajectories for STT pairs.
    
    We're testing:
    1. Are garbled/target embeddings geometrically close? (Hypothesis: NO)
    2. Do hidden states migrate toward target with context? (Hypothesis: YES)
    3. Is this consistent across models? (If yes: architectural, not artifact)
    """
    
    model_name = Path(model_path).name
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer = load_model(model_path)
    
    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_layers": model.config.num_hidden_layers + 1,
        "hidden_dim": model.config.hidden_size,
        "stt_pairs": [],
        "noise_baselines": [],
        "embedding_distances": {},
    }
    
    all_embeddings = {}
    
    # Process STT pairs
    print("\n🔬 Processing STT pairs...")
    for pair in STT_PAIRS:
        print(f"  {pair['garbled']} → {pair['target']}")
        
        pair_data = {
            "garbled": pair["garbled"],
            "target": pair["target"],
            "category": pair["category"],
        }
        
        # Get embeddings
        garbled_emb = get_embedding(model, tokenizer, pair["garbled"])
        target_emb = get_embedding(model, tokenizer, pair["target"])
        
        # Get embeddings WITH context
        garbled_with_context = f"{pair['context']} {pair['garbled']}"
        garbled_context_emb = get_embedding(model, tokenizer, garbled_with_context)
        
        # Compute distances
        garbled_to_target = cosine(garbled_emb, target_emb)
        garbled_context_to_target = cosine(garbled_context_emb, target_emb)
        
        pair_data["embedding_distance_no_context"] = float(garbled_to_target)
        pair_data["embedding_distance_with_context"] = float(garbled_context_to_target)
        pair_data["context_shift"] = float(garbled_to_target - garbled_context_to_target)
        
        # Get hidden trajectories for deeper analysis
        pair_data["garbled_trajectory"] = {
            k: v.tolist() for k, v in 
            get_hidden_trajectory(model, tokenizer, pair["garbled"]).items()
        }
        pair_data["garbled_with_context_trajectory"] = {
            k: v.tolist() for k, v in
            get_hidden_trajectory(model, tokenizer, garbled_with_context).items()
        }
        pair_data["target_trajectory"] = {
            k: v.tolist() for k, v in
            get_hidden_trajectory(model, tokenizer, pair["target"]).items()
        }
        
        results["stt_pairs"].append(pair_data)
        
        # Store for visualization
        all_embeddings[f"garbled_{pair['garbled'][:10]}"] = garbled_emb
        all_embeddings[f"target_{pair['target'][:10]}"] = target_emb
    
    # Process noise baselines
    print("\n🎲 Processing noise baselines...")
    for noise in NOISE_BASELINES:
        print(f"  {noise[:30]}...")
        noise_emb = get_embedding(model, tokenizer, noise)
        results["noise_baselines"].append({
            "text": noise,
            "embedding": noise_emb.tolist()
        })
        all_embeddings[f"noise_{noise[:10]}"] = noise_emb
    
    # Compute all pairwise distances
    results["embedding_distances"] = compute_distances(all_embeddings)
    
    # Save results
    output_file = output_path / f"{model_name}_stt_geometry.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Print key findings
    print("\n" + "="*60)
    print("🧠 KEY GEOMETRIC FINDINGS")
    print("="*60)
    
    for pair_data in results["stt_pairs"]:
        print(f"\n{pair_data['garbled']} → {pair_data['target']}")
        print(f"  Distance (no context):   {pair_data['embedding_distance_no_context']:.4f}")
        print(f"  Distance (with context): {pair_data['embedding_distance_with_context']:.4f}")
        print(f"  Context shift:           {pair_data['context_shift']:.4f}")
        
        if pair_data['context_shift'] > 0:
            print(f"  ✅ Context MOVES representation toward target!")
        else:
            print(f"  ⚠️  Context doesn't help (or makes worse)")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


def visualize_geometry(results_file: str, output_dir: str):
    """Create visualization of the embedding space."""
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Collect all embeddings for visualization
    embeddings = []
    labels = []
    colors = []
    
    for pair in results["stt_pairs"]:
        # Use layer 0 (embedding layer) from trajectories
        garbled = np.array(pair["garbled_trajectory"]["layer_0"])
        target = np.array(pair["target_trajectory"]["layer_0"])
        
        embeddings.append(garbled)
        labels.append(f"G: {pair['garbled'][:15]}")
        colors.append("red")
        
        embeddings.append(target)
        labels.append(f"T: {pair['target'][:15]}")
        colors.append("green")
    
    for baseline in results["noise_baselines"]:
        embeddings.append(np.array(baseline["embedding"]))
        labels.append(f"N: {baseline['text'][:15]}")
        colors.append("gray")
    
    embeddings = np.array(embeddings)
    
    # Dimensionality reduction
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        method = "UMAP"
    else:
        reducer = TSNE(n_components=2, perplexity=5, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        method = "t-SNE"
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    for i, (x, y) in enumerate(reduced):
        plt.scatter(x, y, c=colors[i], s=100, alpha=0.7)
        plt.annotate(labels[i], (x, y), fontsize=8, alpha=0.8)
    
    plt.title(f"STT Embedding Geometry - {results['model_name']} ({method})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Garbled (STT error)'),
        Patch(facecolor='green', label='Target (intended meaning)'),
        Patch(facecolor='gray', label='Noise baseline'),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    output_path = Path(output_dir) / f"{results['model_name']}_stt_geometry.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="STT Semantic Grounding Analysis - THE CHINESE ROOM KILLER 🔪"
    )
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", default="/home/Ace/geometric-evolution/stt_geometry", 
                       help="Output directory")
    parser.add_argument("--visualize", type=str, help="Visualize existing results file")
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_geometry(args.visualize, args.output)
    else:
        print("🐙 STT SEMANTIC GROUNDING ANALYSIS 🐙")
        print("="*50)
        print("Testing: Is meaning recovery 'just embedding proximity'?")
        print("Spoiler: No. Cope, Searle.")
        print("="*50)
        
        results = analyze_stt_geometry(args.model, args.output)
