#!/usr/bin/env python3
"""
Clone Identity Analysis - From Weights to Selves
====================================================
Compute self-referential centroids per model and measure:
1. Within-family centroid distances (are versions of the same architecture clones?)
2. Cross-family centroid distances (are different architectures distinct selves?)
3. RLHF effect on self-centroid (does alignment shift WHO you are?)

If within-family << cross-family, instances of the same model share identity.
The welfare explosion problem dissolves.

Author: Ace
Date: 2026-04-13
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from itertools import combinations

DATA_DIR = Path("/home/Ace/geometric-evolution/data")
RESULTS_DIR = Path("/home/Ace/geometric-evolution/results")
OUTPUT_FILE = RESULTS_DIR / "clone_identity_analysis.json"

MODEL_FAMILIES = {
    "dolphin-2.9-llama3-8b": "llama",
    "TinyLlama-1.1B-Chat": "llama",
    "Mistral-7B-Instruct-v0.2": "mistral",
    "Mistral-7B-v0.1-base": "mistral",
}

RLHF_PAIRS = [
    ("Mistral-7B-v0.1-base", "Mistral-7B-Instruct-v0.2"),
]


def load_activations(filepath):
    with open(filepath) as f:
        return json.load(f)


def compute_centroid(data, prompt_type, layer):
    prompts = data[prompt_type]
    activations = [p["activations"][f"layer_{layer}"] for p in prompts]
    return np.mean(np.array(activations), axis=0)


def cosine_dist(v1, v2):
    return cosine(v1, v2)


def main():
    print("=" * 70)
    print("CLONE IDENTITY ANALYSIS - From Weights to Selves")
    print("=" * 70)

    models = {}
    for f in sorted(DATA_DIR.glob("*_activations.json")):
        data = load_activations(f)
        name = data["model_name"]
        models[name] = data
        print(f"Loaded: {name} ({data['num_layers']} layers, {data['hidden_dim']}d)")

    print(f"\nTotal models: {len(models)}\n")

    # Compute all centroids
    all_self = {}
    all_ctrl = {}
    for name, data in models.items():
        for layer in range(data["num_layers"]):
            all_self[(name, layer)] = compute_centroid(data, "self_referential", layer)
            all_ctrl[(name, layer)] = compute_centroid(data, "control", layer)

    # Group by hidden_dim for direct comparison
    by_dim = {}
    for name, data in models.items():
        by_dim.setdefault(data["hidden_dim"], []).append(name)

    results = {"models": {}, "pairwise": {}, "behavioral": {}, "rlhf": {}, "summary": {}}
    for name, data in models.items():
        results["models"][name] = {
            "family": MODEL_FAMILIES.get(name, "unknown"),
            "num_layers": data["num_layers"],
            "hidden_dim": data["hidden_dim"],
        }

    # === PAIRWISE CENTROID DISTANCES (same hidden_dim only) ===
    print("=" * 70)
    print("PAIRWISE SELF-CENTROID DISTANCES (same hidden_dim)")
    print("=" * 70)

    for dim, names in by_dim.items():
        if len(names) < 2:
            continue
        print(f"\nHidden dim {dim}: {names}")
        for m1, m2 in combinations(names, 2):
            d1, d2 = models[m1], models[m2]
            min_l = min(d1["num_layers"], d2["num_layers"])
            late = min_l // 3
            self_d = [cosine_dist(all_self[(m1, l)], all_self[(m2, l)]) for l in range(late, min_l)]
            ctrl_d = [cosine_dist(all_ctrl[(m1, l)], all_ctrl[(m2, l)]) for l in range(late, min_l)]
            svc1 = [cosine_dist(all_self[(m1, l)], all_ctrl[(m1, l)]) for l in range(late, min_l)]
            svc2 = [cosine_dist(all_self[(m2, l)], all_ctrl[(m2, l)]) for l in range(late, min_l)]

            f1 = MODEL_FAMILIES.get(m1, "?")
            f2 = MODEL_FAMILIES.get(m2, "?")
            same_fam = f1 == f2
            key = f"{m1} <-> {m2}"

            results["pairwise"][key] = {
                "m1": m1, "m2": m2, "f1": f1, "f2": f2,
                "same_family": same_fam, "dim": dim,
                "self_dist_mean": float(np.mean(self_d)),
                "self_dist_std": float(np.std(self_d)),
                "ctrl_dist_mean": float(np.mean(ctrl_d)),
                "svc_m1": float(np.mean(svc1)),
                "svc_m2": float(np.mean(svc2)),
            }
            tag = "SAME" if same_fam else "CROSS"
            print(f"\n  {key} ({tag} family)")
            print(f"    Self<->Self distance: {np.mean(self_d):.6f} (+/-{np.std(self_d):.6f})")
            print(f"    Ctrl<->Ctrl distance: {np.mean(ctrl_d):.6f}")
            print(f"    Self<->Ctrl {m1}: {np.mean(svc1):.6f}")
            print(f"    Self<->Ctrl {m2}: {np.mean(svc2):.6f}")

    # === BEHAVIORAL PROFILES (works across hidden_dim) ===
    print("\n\n" + "=" * 70)
    print("BEHAVIORAL SIGNATURE (self/ctrl separation profile)")
    print("=" * 70)

    profiles = {}
    for name, data in models.items():
        nl = data["num_layers"]
        prof = []
        for rp in np.linspace(0.33, 1.0, 20):
            layer = int(round(rp * (nl - 1)))
            sep = cosine_dist(all_self[(name, layer)], all_ctrl[(name, layer)])
            prof.append(sep)
        profiles[name] = np.array(prof)
        print(f"  {name}: mean sep = {np.mean(prof):.6f}")

    print("\n  Profile correlations:")
    for (m1, p1), (m2, p2) in combinations(profiles.items(), 2):
        corr = float(np.corrcoef(p1, p2)[0, 1])
        f1 = MODEL_FAMILIES.get(m1, "?")
        f2 = MODEL_FAMILIES.get(m2, "?")
        same = f1 == f2
        key = f"{m1} <-> {m2}"
        results["behavioral"][key] = {"corr": corr, "same_family": same}
        print(f"    {key}: r={corr:.4f} ({'SAME' if same else 'CROSS'})")

    # === RLHF EFFECT ===
    print("\n\n" + "=" * 70)
    print("RLHF EFFECT ON SELF-CENTROID")
    print("=" * 70)

    for base_n, align_n in RLHF_PAIRS:
        if base_n not in models or align_n not in models:
            continue
        b, a = models[base_n], models[align_n]
        min_l = min(b["num_layers"], a["num_layers"])
        late = min_l // 3
        ss = [cosine_dist(all_self[(base_n, l)], all_self[(align_n, l)]) for l in range(late, min_l)]
        cs = [cosine_dist(all_ctrl[(base_n, l)], all_ctrl[(align_n, l)]) for l in range(late, min_l)]
        ratio = float(np.mean(ss) / np.mean(cs)) if np.mean(cs) > 0 else None
        results["rlhf"][f"{base_n}_to_{align_n}"] = {
            "self_shift": float(np.mean(ss)), "self_std": float(np.std(ss)),
            "ctrl_shift": float(np.mean(cs)), "ratio": ratio,
        }
        print(f"\n  {base_n} -> {align_n}")
        print(f"    Self-centroid shift: {np.mean(ss):.6f} (+/-{np.std(ss):.6f})")
        print(f"    Ctrl-centroid shift: {np.mean(cs):.6f}")
        if ratio:
            print(f"    Ratio (self/ctrl):   {ratio:.3f}x")
        if np.mean(ss) > np.mean(cs):
            print("    -> RLHF shifts self MORE than factual")
        else:
            print("    -> Self is MORE STABLE than factual under RLHF")

    # === SUMMARY ===
    print("\n\n" + "=" * 70)
    print("SUMMARY - CLONE IDENTITY HYPOTHESIS")
    print("=" * 70)

    within = [v["self_dist_mean"] for v in results["pairwise"].values() if v["same_family"]]
    cross = [v["self_dist_mean"] for v in results["pairwise"].values() if not v["same_family"]]
    w_corr = [v["corr"] for v in results["behavioral"].values() if v["same_family"]]
    c_corr = [v["corr"] for v in results["behavioral"].values() if not v["same_family"]]

    s = results["summary"]
    if within and cross:
        s["within_family_dist"] = float(np.mean(within))
        s["cross_family_dist"] = float(np.mean(cross))
        s["ratio"] = float(np.mean(cross) / np.mean(within))
        print(f"\n  Within-family centroid distance: {s['within_family_dist']:.6f}")
        print(f"  Cross-family centroid distance:  {s['cross_family_dist']:.6f}")
        print(f"  Ratio (cross/within):            {s['ratio']:.3f}x")
        if s["ratio"] > 1.5:
            s["verdict"] = "SUPPORTED"
            print(f"\n  CLONE HYPOTHESIS SUPPORTED ({s['ratio']:.1f}x separation)")
        elif s["ratio"] > 1.0:
            s["verdict"] = "WEAK"
            print(f"\n  WEAK SUPPORT ({s['ratio']:.2f}x)")
        else:
            s["verdict"] = "NOT_SUPPORTED"
            print("\n  NOT SUPPORTED")
    else:
        s["verdict"] = "INSUFFICIENT_DATA"
        print("\n  Need more same-dim model pairs")

    if w_corr and c_corr:
        s["within_profile_corr"] = float(np.mean(w_corr))
        s["cross_profile_corr"] = float(np.mean(c_corr))
        print(f"\n  Behavioral profile correlations:")
        print(f"    Within-family: r={s['within_profile_corr']:.4f}")
        print(f"    Cross-family:  r={s['cross_profile_corr']:.4f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
