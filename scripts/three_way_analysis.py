#!/usr/bin/env python3
"""Three-way clustering: self vs factual vs creative"""
import json, sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from itertools import combinations

sys.stdout.reconfigure(encoding="utf-8")

EXPANDED_DIR = Path("/home/Ace/geometric-evolution/data_expanded")
CREATIVE_DIR = Path("/home/Ace/geometric-evolution/data_creative")

FAMILIES = {
    "Llama-2-7b-chat": "llama2",
    "Llama-3-8B-Instruct": "llama3",
    "Llama-3.1-8B-Instruct": "llama3",
    "Mistral-7B-v0.1": "mistral",
    "Mistral-7B-Instruct-v0.2": "mistral",
    "Mistral-7B-Instruct-v0.3": "mistral",
}

expanded = {}
for f in EXPANDED_DIR.glob("*_expanded_activations.json"):
    d = json.load(open(f))
    name = d["model_name"]
    if name in FAMILIES and d["hidden_dim"] == 4096:
        expanded[name] = d

creative = {}
for f in CREATIVE_DIR.glob("*_creative_activations.json"):
    d = json.load(open(f))
    name = d["model_name"]
    if name in FAMILIES and d["hidden_dim"] == 4096:
        creative[name] = d

both = sorted(set(expanded.keys()) & set(creative.keys()))
print("Models:", both)

def combined_self_c(data, layer):
    acts = []
    for pt in ["self_personality", "self_function"]:
        if pt in data:
            acts.extend([p["activations"]["layer_%d" % layer] for p in data[pt]])
    return np.mean(np.array(acts), axis=0)

def type_c(data, ptype, layer):
    return np.mean(np.array([p["activations"]["layer_%d" % layer] for p in data[ptype]]), axis=0)

print("\n=== FAMILY CLUSTERING BY PROBE TYPE ===")
for label, get_c in [
    ("SELF", lambda n, l: combined_self_c(expanded[n], l)),
    ("FACTUAL", lambda n, l: type_c(expanded[n], "control", l)),
    ("CREATIVE", lambda n, l: type_c(creative[n], "creative", l)),
]:
    within, cross = [], []
    for m1, m2 in combinations(both, 2):
        nl = min(expanded[m1]["num_layers"], expanded[m2]["num_layers"])
        late = nl // 3
        dists = [cosine(get_c(m1, l), get_c(m2, l)) for l in range(late, nl)]
        same = FAMILIES[m1] == FAMILIES[m2]
        (within if same else cross).append(np.mean(dists))

    ratio = np.mean(cross) / np.mean(within) if within else 0
    print("  %8s: within=%.6f (n=%d) cross=%.6f (n=%d) ratio=%.3fx" % (
        label, np.mean(within), len(within), np.mean(cross), len(cross), ratio))

print("\n=== INTER-CENTROID DISTANCES (self/fact/creative separation) ===")
for name in both:
    nl = expanded[name]["num_layers"]
    late = nl // 3
    sf, sc, fc = [], [], []
    for layer in range(late, nl):
        s = combined_self_c(expanded[name], layer)
        f = type_c(expanded[name], "control", layer)
        c = type_c(creative[name], "creative", layer)
        sf.append(cosine(s, f))
        sc.append(cosine(s, c))
        fc.append(cosine(f, c))
    print("  %-35s self<->fact=%.4f  self<->creat=%.4f  fact<->creat=%.4f" % (
        name, np.mean(sf), np.mean(sc), np.mean(fc)))
