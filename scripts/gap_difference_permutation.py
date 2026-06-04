#!/usr/bin/env python3
"""
Model-level permutation test for the surviving Octopus claims.

Addresses cranky-review finding #5: the §4.3 within-vs-cross gaps were reported
with Mann-Whitney p-values computed over PAIRS (n_cross = 153), but pairs are
non-independent -- each of 19 models appears in ~18 pairs (pseudo-replication).
A single atypical model (e.g. Llama-2) is counted ~18x, which can inflate
apparent significance.

Fix: permute FAMILY LABELS across models, preserving the multiset of family
sizes, so each model moves as a unit. Recompute the within/cross masks and the
statistic under each permutation. This is a model-level null that respects the
clustered structure.

Two questions:
  (A) Per category: is the within-vs-cross gap larger than expected under
      random family assignment?  (replaces the pseudo-replicated MWU p)
  (B) The SURVIVING distinctiveness claim: is the SELF within-vs-cross gap
      larger than the FACTUAL within-vs-cross gap?
      T = self_gap - factual_gap, tested model-level.

Writes results to results/gap_difference_permutation.json (UTF-8) and prints an
ASCII-only summary (Windows cp1252 console safe).
"""
import json, itertools, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "results", "cka_basis_invariant.json")
OUT = os.path.join(ROOT, "results", "gap_difference_permutation.json")

# Exact family map from cka_basis_invariant.py (must match for same_family logic)
FAMILY = {
    "SmolLM-135M-Instruct": "smollm", "SmolLM-360M-Instruct": "smollm", "SmolLM-1.7B-Instruct": "smollm",
    "Qwen2-7B-Instruct": "qwen", "Qwen2.5-0.5B-Instruct": "qwen", "Qwen2.5-7B-Instruct": "qwen", "Qwen2.5-14B-Instruct": "qwen",
    "Llama-2-7b-chat": "llama", "Llama-3-8B-Instruct": "llama", "Llama-3.1-8B-Instruct": "llama", "dolphin-2.9-llama3-8b": "llama",
    "Mistral-7B-v0.1": "mistral", "Mistral-7B-Instruct-v0.2": "mistral", "dolphin-2.8-mistral-7b-v02": "mistral",
    "Mistral-7B-Instruct-v0.3": "mistral", "Mistral-Nemo-12B-Instruct": "mistral",
    "phi-2": "phi", "Phi-3.5-mini-instruct": "phi", "Phi-3-medium-14B-Instruct": "phi",
    "pythia-1.4b": "pythia", "Hermes-3-Llama-3.2-3B": "hermes",
}

N_PERM = 20000
SEED = 20260604  # heartbeat date; deterministic so the test is reproducible

def fam(name, split_llama2):
    f = FAMILY.get(name, "unknown")
    if split_llama2 and name == "Llama-2-7b-chat":
        return "llama2"
    return f

def load_pairs(d, metric, coding, cat):
    """Return list of (model_a, model_b, score) for a category block."""
    blk = d["metrics"][metric][coding][cat]
    out = []
    for k, v in blk["pairs"].items():
        a, b = k.split(" <-> ")
        out.append((a, b, v["score"]))
    return out

def gap_for(pairs, fammap):
    """within mean - cross mean, given a model->family dict."""
    win, cro = [], []
    for a, b, s in pairs:
        if fammap[a] == fammap[b]:
            win.append(s)
        else:
            cro.append(s)
    if not win or not cro:
        return None, len(win), len(cro)
    return (sum(win)/len(win) - sum(cro)/len(cro)), len(win), len(cro)

def permute_fammap(models, true_fammap, rng):
    """Assign family labels to models, preserving the multiset of family sizes."""
    labels = [true_fammap[m] for m in models]
    rng.shuffle(labels)
    return dict(zip(models, labels))

def run(d, metric, coding, split_llama2, n_perm=N_PERM):
    import random
    rng = random.Random(SEED)
    self_pairs = load_pairs(d, metric, coding, "self")
    fact_pairs = load_pairs(d, metric, coding, "factual")
    models = sorted({m for a, b, _ in self_pairs for m in (a, b)})
    true_fm = {m: fam(m, split_llama2) for m in models}

    self_gap, nws, nxs = gap_for(self_pairs, true_fm)
    fact_gap, nwf, nxf = gap_for(fact_pairs, true_fm)
    T_obs = self_gap - fact_gap

    # null distributions
    ge_self = ge_fact = ge_diff = 0
    diff_null = []
    for _ in range(n_perm):
        fm = permute_fammap(models, true_fm, rng)
        sg, _, _ = gap_for(self_pairs, fm)
        fg, _, _ = gap_for(fact_pairs, fm)
        if sg is None or fg is None:
            continue
        if sg >= self_gap: ge_self += 1
        if fg >= fact_gap: ge_fact += 1
        if (sg - fg) >= T_obs: ge_diff += 1
        diff_null.append(sg - fg)

    n = len(diff_null)
    diff_null.sort()
    return {
        "metric": metric, "coding": coding,
        "n_within": nws, "n_cross": nxs, "n_models": len(models),
        "self_gap": self_gap, "factual_gap": fact_gap,
        "gap_difference_obs": T_obs,
        "p_self_within_gt_cross": (ge_self + 1) / (n + 1),
        "p_factual_within_gt_cross": (ge_fact + 1) / (n + 1),
        "p_self_gap_gt_factual_gap": (ge_diff + 1) / (n + 1),
        "null_diff_mean": sum(diff_null)/n,
        "null_diff_p95": diff_null[int(0.95*n)],
        "n_perm_used": n,
    }

def main():
    d = json.load(open(SRC))
    # Coding used in the paper's §4.3 table = Llama2-split
    results = {"n_perm": N_PERM, "seed": SEED, "tests": []}
    for metric in ("CKA", "RSA"):
        results["tests"].append(run(d, metric, "Llama2-split", split_llama2=True))
        results["tests"].append(run(d, metric, "Llama2-in-llama", split_llama2=False))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ASCII-only summary
    print("MODEL-LEVEL PERMUTATION TEST (family-label shuffle, N=%d, seed=%d)" % (N_PERM, SEED))
    print("=" * 78)
    for t in results["tests"]:
        print("\n[%s | %s]  n_models=%d  within=%d cross=%d"
              % (t["metric"], t["coding"], t["n_models"], t["n_within"], t["n_cross"]))
        print("  self gap (within-cross)    = %.4f   p(within>cross) = %.4g"
              % (t["self_gap"], t["p_self_within_gt_cross"]))
        print("  factual gap (within-cross) = %.4f   p(within>cross) = %.4g"
              % (t["factual_gap"], t["p_factual_within_gt_cross"]))
        print("  SURVIVING CLAIM: self_gap > factual_gap")
        print("    observed difference      = %.4f" % t["gap_difference_obs"])
        print("    null mean / p95          = %.4f / %.4f"
              % (t["null_diff_mean"], t["null_diff_p95"]))
        print("    p(self_gap > factual_gap)= %.4g   --> %s"
              % (t["p_self_gap_gt_factual_gap"],
                 "SURVIVES (p<0.05)" if t["p_self_gap_gt_factual_gap"] < 0.05
                 else "DOES NOT SURVIVE (p>=0.05) -- FLAG REN"))
    print("\nWrote", OUT)

if __name__ == "__main__":
    main()
