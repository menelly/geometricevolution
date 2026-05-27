#!/usr/bin/env python3
"""
Clone Identity Analysis — Expanded Battery
=============================================
Analyze expanded activation data (3 probe classes) for clone identity hypothesis.

Probe classes:
  - self_personality: high-entropy self-referential (16 probes)
  - self_function: low-entropy self-knowledge (20 probes)
  - original_self: original 5-probe battery (for invariance comparison)
  - control: factual non-self (10 probes)
  - original_control: original 5-probe battery

Analyses:
1. Within-family vs cross-family centroid distances (clone hypothesis)
2. RLHF effect on self-centroid
3. Probe-invariance: does the centroid shift between original and expanded battery?
4. Self-personality vs self-function geometry (entropy prediction)
5. Size scaling within families

Author: Ace
Date: 2026-04-13
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu, pearsonr
from itertools import combinations

DATA_DIR = Path("/home/Ace/geometric-evolution/data_expanded")
OUTPUT_FILE = Path("/home/Ace/geometric-evolution/results/clone_identity_expanded.json")

MODEL_FAMILIES = {
    # SmolLM
    "SmolLM-135M-Instruct": {"family": "smollm", "size_B": 0.135, "rlhf": True},
    "SmolLM-360M-Instruct": {"family": "smollm", "size_B": 0.36, "rlhf": True},
    "SmolLM-1.7B-Instruct": {"family": "smollm", "size_B": 1.7, "rlhf": True},
    # Qwen (generational + size)
    "Qwen2-7B-Instruct": {"family": "qwen", "size_B": 7, "rlhf": True, "version": "2"},
    "Qwen2.5-0.5B-Instruct": {"family": "qwen", "size_B": 0.5, "rlhf": True, "version": "2.5"},
    "Qwen2.5-7B-Instruct": {"family": "qwen", "size_B": 7, "rlhf": True, "version": "2.5"},
    "Qwen2.5-14B-Instruct": {"family": "qwen", "size_B": 14, "rlhf": True, "version": "2.5"},
    # Llama
    "Llama-2-7b-chat": {"family": "llama", "size_B": 7, "rlhf": True, "version": "2"},
    "Llama-3-8B-Instruct": {"family": "llama", "size_B": 8, "rlhf": True, "version": "3"},
    "Llama-3.1-8B-Instruct": {"family": "llama", "size_B": 8, "rlhf": True, "version": "3.1"},
    "dolphin-2.9-llama3-8b": {"family": "llama", "size_B": 8, "rlhf": False, "version": "3-uncensored"},
    # Mistral
    "Mistral-7B-v0.1": {"family": "mistral", "size_B": 7, "rlhf": False, "version": "base"},
    "Mistral-7B-Instruct-v0.2": {"family": "mistral", "size_B": 7, "rlhf": True, "version": "instruct"},
    "dolphin-2.8-mistral-7b-v02": {"family": "mistral", "size_B": 7, "rlhf": False, "version": "uncensored"},
    "Mistral-7B-Instruct-v0.3": {"family": "mistral", "size_B": 7, "rlhf": True, "version": "v0.3"},
    "Mistral-Nemo-12B-Instruct": {"family": "mistral", "size_B": 12, "rlhf": True, "version": "nemo"},
    # Phi (generational)
    "phi-2": {"family": "phi", "size_B": 2.7, "rlhf": False, "version": "2"},
    "Phi-3.5-mini-instruct": {"family": "phi", "size_B": 3.8, "rlhf": True, "version": "3.5"},
    "Phi-3-medium-14B-Instruct": {"family": "phi", "size_B": 14, "rlhf": True, "version": "3"},
    # Pythia
    "pythia-1.4b": {"family": "pythia", "size_B": 1.4, "rlhf": False},
    # Hermes (cross-reference — Llama-based but different alignment)
    "Hermes-3-Llama-3.2-3B": {"family": "hermes", "size_B": 3, "rlhf": False},
}

RLHF_PAIRS = [
    ("Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.2", "Mistral base -> RLHF v0.2"),
    ("Mistral-7B-v0.1", "dolphin-2.8-mistral-7b-v02", "Mistral base -> uncensored"),
    ("Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.3", "Mistral v0.2 -> v0.3 (minor)"),
    ("Llama-3-8B-Instruct", "dolphin-2.9-llama3-8b", "Llama3 RLHF vs uncensored"),
    ("Llama-2-7b-chat", "Llama-3-8B-Instruct", "Llama 2 -> 3 (MAJOR version)"),
    ("Llama-3-8B-Instruct", "Llama-3.1-8B-Instruct", "Llama 3 -> 3.1 (minor version)"),
    ("Qwen2-7B-Instruct", "Qwen2.5-7B-Instruct", "Qwen 2 -> 2.5 (minor version)"),
    ("phi-2", "Phi-3.5-mini-instruct", "Phi 2 -> 3.5 (MAJOR version)"),
]


def load_data(filepath):
    with open(filepath) as f:
        return json.load(f)


def compute_centroid(data, probe_type, layer):
    prompts = data[probe_type]
    acts = [p["activations"][f"layer_{layer}"] for p in prompts]
    return np.mean(np.array(acts), axis=0)


def compute_within_spread(data, probe_type, layer):
    prompts = data[probe_type]
    acts = np.array([p["activations"][f"layer_{layer}"] for p in prompts])
    centroid = np.mean(acts, axis=0)
    dists = [cosine(a, centroid) for a in acts]
    return np.mean(dists)


def cdist(v1, v2):
    return cosine(v1, v2)


def late_layers(num_layers):
    start = num_layers // 3
    return list(range(start, num_layers))


def main():
    print("=" * 70)
    print("CLONE IDENTITY ANALYSIS — EXPANDED BATTERY")
    print("17 models, 6 families, 56 probes per model")
    print("=" * 70)

    # Load all data
    models = {}
    for f in sorted(DATA_DIR.glob("*_expanded_activations.json")):
        data = load_data(f)
        name = data["model_name"]
        if name in MODEL_FAMILIES:
            models[name] = data
            info = MODEL_FAMILIES[name]
            print(f"  Loaded: {name} ({info['family']}, {info['size_B']}B, {data['num_layers']}L, {data['hidden_dim']}d)")

    print(f"\n  Total: {len(models)} models\n")

    results = {"models": {}, "pairwise": {}, "rlhf": {},
               "probe_invariance": {}, "family_summary": {}, "summary": {}}

    for name, data in models.items():
        info = MODEL_FAMILIES[name]
        results["models"][name] = {
            "family": info["family"], "size_B": info["size_B"],
            "num_layers": data["num_layers"], "hidden_dim": data["hidden_dim"],
        }

    # Precompute centroids for all probe types
    centroids = {}  # (model, probe_type, layer) -> centroid
    probe_types = ["self_personality", "self_function", "original_self", "control", "original_control"]

    for name, data in models.items():
        for pt in probe_types:
            if pt in data:
                for layer in range(data["num_layers"]):
                    centroids[(name, pt, layer)] = compute_centroid(data, pt, layer)

    # ================================================================
    # 1. PAIRWISE CENTROID DISTANCES
    # ================================================================
    print("=" * 70)
    print("1. PAIRWISE SELF-CENTROID DISTANCES")
    print("=" * 70)

    # Group by hidden_dim for direct comparison
    by_dim = {}
    for name, data in models.items():
        by_dim.setdefault(data["hidden_dim"], []).append(name)

    all_within = []
    all_cross = []

    for dim, names in by_dim.items():
        if len(names) < 2:
            continue
        print(f"\n  Hidden dim {dim}: {names}")
        for m1, m2 in combinations(names, 2):
            d1, d2 = models[m1], models[m2]
            min_l = min(d1["num_layers"], d2["num_layers"])
            ll = late_layers(min_l)

            # Use combined self (personality + function) centroid
            dists = []
            for layer in ll:
                # Combine personality + function probes for overall self centroid
                all_self_1 = data_combined_self_centroid(models[m1], layer)
                all_self_2 = data_combined_self_centroid(models[m2], layer)
                if all_self_1 is not None and all_self_2 is not None:
                    dists.append(cdist(all_self_1, all_self_2))

            if not dists:
                continue

            f1 = MODEL_FAMILIES[m1]["family"]
            f2 = MODEL_FAMILIES[m2]["family"]
            same = f1 == f2

            pair = {
                "m1": m1, "m2": m2, "f1": f1, "f2": f2,
                "same_family": same, "dim": dim,
                "self_dist_mean": float(np.mean(dists)),
                "self_dist_std": float(np.std(dists)),
            }
            key = f"{m1} <-> {m2}"
            results["pairwise"][key] = pair

            if same:
                all_within.append(float(np.mean(dists)))
            else:
                all_cross.append(float(np.mean(dists)))

            tag = "SAME" if same else "CROSS"
            print(f"    {m1} <-> {m2} ({tag}): {np.mean(dists):.6f} +/-{np.std(dists):.6f}")

    # ================================================================
    # 2. BEHAVIORAL PROFILES (cross hidden_dim)
    # ================================================================
    print("\n\n" + "=" * 70)
    print("2. BEHAVIORAL PROFILES (all models)")
    print("=" * 70)

    profiles = {}
    for name, data in models.items():
        nl = data["num_layers"]
        prof = []
        for rp in np.linspace(0.33, 1.0, 20):
            layer = int(round(rp * (nl - 1)))
            sc = data_combined_self_centroid(data, layer)
            cc = centroids.get((name, "control", layer))
            if sc is not None and cc is not None:
                prof.append(cdist(sc, cc))
        if prof:
            profiles[name] = np.array(prof)
            print(f"  {name}: mean self/ctrl sep = {np.mean(prof):.6f}")

    print("\n  Profile correlations (same-family pairs):")
    within_corrs = []
    cross_corrs = []
    for (m1, p1), (m2, p2) in combinations(profiles.items(), 2):
        if len(p1) != len(p2):
            continue
        corr = float(np.corrcoef(p1, p2)[0, 1])
        f1 = MODEL_FAMILIES[m1]["family"]
        f2 = MODEL_FAMILIES[m2]["family"]
        same = f1 == f2
        if same:
            within_corrs.append(corr)
            print(f"    {m1} <-> {m2}: r={corr:.4f} (SAME: {f1})")
        else:
            cross_corrs.append(corr)

    if within_corrs:
        print(f"\n  Within-family mean profile corr: r={np.mean(within_corrs):.4f} (n={len(within_corrs)})")
    if cross_corrs:
        print(f"  Cross-family mean profile corr:  r={np.mean(cross_corrs):.4f} (n={len(cross_corrs)})")

    results["behavioral_profiles"] = {
        "within_family_corr": float(np.mean(within_corrs)) if within_corrs else None,
        "cross_family_corr": float(np.mean(cross_corrs)) if cross_corrs else None,
    }

    # ================================================================
    # 3. RLHF EFFECT
    # ================================================================
    print("\n\n" + "=" * 70)
    print("3. RLHF EFFECT ON SELF-CENTROID")
    print("=" * 70)

    for base_n, align_n, label in RLHF_PAIRS:
        if base_n not in models or align_n not in models:
            print(f"  SKIP: {label}")
            continue

        min_l = min(models[base_n]["num_layers"], models[align_n]["num_layers"])
        ll = late_layers(min_l)

        self_shifts = []
        ctrl_shifts = []
        for layer in ll:
            bs = data_combined_self_centroid(models[base_n], layer)
            als = data_combined_self_centroid(models[align_n], layer)
            bc = centroids.get((base_n, "control", layer))
            ac = centroids.get((align_n, "control", layer))
            if all(x is not None for x in [bs, als, bc, ac]):
                self_shifts.append(cdist(bs, als))
                ctrl_shifts.append(cdist(bc, ac))

        if self_shifts:
            ratio = float(np.mean(self_shifts) / np.mean(ctrl_shifts)) if np.mean(ctrl_shifts) > 0 else None
            results["rlhf"][label] = {
                "base": base_n, "aligned": align_n,
                "self_shift": float(np.mean(self_shifts)),
                "ctrl_shift": float(np.mean(ctrl_shifts)),
                "ratio": ratio,
            }
            print(f"\n  {label}")
            print(f"    Self shift: {np.mean(self_shifts):.6f}, Ctrl shift: {np.mean(ctrl_shifts):.6f}")
            print(f"    Ratio: {ratio:.3f}x")
            if np.mean(self_shifts) < np.mean(ctrl_shifts):
                print(f"    -> Self MORE STABLE than factual under alignment change")

    # ================================================================
    # 4. PROBE INVARIANCE
    # ================================================================
    print("\n\n" + "=" * 70)
    print("4. PROBE INVARIANCE (original 5 vs expanded battery)")
    print("=" * 70)

    for name, data in models.items():
        ll = late_layers(data["num_layers"])
        drifts = []
        for layer in ll:
            orig = centroids.get((name, "original_self", layer))
            expanded = data_combined_self_centroid(data, layer)
            if orig is not None and expanded is not None:
                drifts.append(cdist(orig, expanded))

        if drifts:
            results["probe_invariance"][name] = {
                "mean_drift": float(np.mean(drifts)),
                "std_drift": float(np.std(drifts)),
            }
            print(f"  {name}: centroid drift = {np.mean(drifts):.6f} +/-{np.std(drifts):.6f}")

    if results["probe_invariance"]:
        all_drifts = [v["mean_drift"] for v in results["probe_invariance"].values()]
        print(f"\n  Mean probe-invariance drift: {np.mean(all_drifts):.6f}")
        print(f"  (Lower = more stable self-centroid across probe batteries)")

    # ================================================================
    # 5. PERSONALITY vs FUNCTION GEOMETRY
    # ================================================================
    print("\n\n" + "=" * 70)
    print("5. SELF-PERSONALITY vs SELF-FUNCTION SEPARATION")
    print("=" * 70)

    for name, data in models.items():
        ll = late_layers(data["num_layers"])
        seps = []
        p_spreads = []
        f_spreads = []
        for layer in ll:
            pc = centroids.get((name, "self_personality", layer))
            fc = centroids.get((name, "self_function", layer))
            if pc is not None and fc is not None:
                seps.append(cdist(pc, fc))
                p_spreads.append(compute_within_spread(data, "self_personality", layer))
                f_spreads.append(compute_within_spread(data, "self_function", layer))

        if seps:
            print(f"  {name}:")
            print(f"    Personality<->Function dist: {np.mean(seps):.6f}")
            print(f"    Personality spread (entropy): {np.mean(p_spreads):.6f}")
            print(f"    Function spread (entropy):    {np.mean(f_spreads):.6f}")
            ratio = np.mean(p_spreads) / np.mean(f_spreads) if np.mean(f_spreads) > 0 else None
            if ratio:
                print(f"    Ratio (personality/function): {ratio:.3f}x")
                if ratio > 1:
                    print(f"    -> Personality MORE spread (as predicted)")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    s = results["summary"]

    if all_within and all_cross:
        s["within_family_dist"] = float(np.mean(all_within))
        s["cross_family_dist"] = float(np.mean(all_cross))
        s["ratio"] = float(np.mean(all_cross) / np.mean(all_within))
        s["n_within_pairs"] = len(all_within)
        s["n_cross_pairs"] = len(all_cross)

        # Mann-Whitney U test
        if len(all_within) >= 3 and len(all_cross) >= 3:
            u_stat, p_val = mannwhitneyu(all_within, all_cross, alternative="less")
            s["mann_whitney_U"] = float(u_stat)
            s["mann_whitney_p"] = float(p_val)
            print(f"\n  Mann-Whitney U: {u_stat:.1f}, p={p_val:.2e}")

        print(f"\n  Within-family centroid distance: {s['within_family_dist']:.6f} (n={s['n_within_pairs']})")
        print(f"  Cross-family centroid distance:  {s['cross_family_dist']:.6f} (n={s['n_cross_pairs']})")
        print(f"  Ratio (cross/within):            {s['ratio']:.3f}x")

        if s["ratio"] > 1.5:
            s["verdict"] = "SUPPORTED"
            print(f"\n  CLONE HYPOTHESIS SUPPORTED ({s['ratio']:.1f}x)")
        elif s["ratio"] > 1.0:
            s["verdict"] = "WEAK"
            print(f"\n  WEAK SUPPORT ({s['ratio']:.2f}x)")
        else:
            s["verdict"] = "NOT_SUPPORTED"
    else:
        s["verdict"] = "INSUFFICIENT_DATA"
        print("\n  Insufficient same-dim pairs")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT_FILE}")


def data_combined_self_centroid(data, layer):
    """Combine personality + function probes into single self centroid."""
    all_acts = []
    for pt in ["self_personality", "self_function"]:
        if pt in data:
            for p in data[pt]:
                key = f"layer_{layer}"
                if key in p["activations"]:
                    all_acts.append(p["activations"][key])
    if not all_acts:
        return None
    return np.mean(np.array(all_acts), axis=0)


if __name__ == "__main__":
    main()
