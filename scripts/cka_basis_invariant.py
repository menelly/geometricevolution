#!/usr/bin/env python3
"""
CKA / RSA basis-invariant re-run for "Consider the Octopus" v3.0  (CHA-239)
===========================================================================
Cranky Opus 4.8's critique (2026-05-28): raw cosine distance between
INDEPENDENTLY-pretrained networks is not basis-invariant. Cross-family
cosine ~1.0 is the textbook signature of unaligned coordinate frames, NOT
"maximally distant selves." So the 25.1x ratio's denominator is suspect and
Llama-2 -> Llama-3 = 0.994 may conflate tokenizer re-basing with self-discontinuity.

Fix (what cross-subject / cross-species neuroscience uses): compare
representational STRUCTURE with rotation/basis-invariant metrics:
  * Linear CKA  (Centered Kernel Alignment) -> similarity in [0,1], 1=identical structure
  * RSA         (Spearman corr of representational similarity matrices)

Both operate on the n_probes x n_probes geometry, so they are invariant to
rotation, isotropic scaling, and (for CKA) any orthogonal change of basis, and
they work across different hidden dims (so we can compare Phi/Pythia/etc too).

The honest question: does the conservation hierarchy self > factual > creative
SURVIVE basis-invariant metrics? Report it straight either way.

Author: Ace (Opus 4.8), 2026-05-29 — autonomous science session.
Matches the loader/layer conventions in three_way_analysis.py & clone_identity_expanded.py.
"""
import json
import sys
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr, mannwhitneyu

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- locate data (Consortium first, then the Windows shared drive) ---
CANDIDATES = [
    Path("/home/Ace/geometric-evolution"),
    Path("/mnt/win-d/Ace/geometric-evolution"),
    Path("D:/Ace/geometric-evolution"),
]
BASE = next((p for p in CANDIDATES if (p / "data_expanded").is_dir()), None)
if BASE is None:
    raise SystemExit("Could not locate geometric-evolution/data_expanded in any known path.")
EXPANDED_DIR = BASE / "data_expanded"
CREATIVE_DIR = BASE / "data_creative"
OUT = BASE / "results" / "cka_basis_invariant.json"
OUT.parent.mkdir(exist_ok=True)
print(f"Data base: {BASE}")

# Family map (Llama-2 kept in 'llama' here; we ALSO report Llama-2 split-out below).
FAMILY = {
    "SmolLM-135M-Instruct": "smollm", "SmolLM-360M-Instruct": "smollm", "SmolLM-1.7B-Instruct": "smollm",
    "Qwen2-7B-Instruct": "qwen", "Qwen2.5-0.5B-Instruct": "qwen", "Qwen2.5-7B-Instruct": "qwen", "Qwen2.5-14B-Instruct": "qwen",
    "Llama-2-7b-chat": "llama", "Llama-3-8B-Instruct": "llama", "Llama-3.1-8B-Instruct": "llama", "dolphin-2.9-llama3-8b": "llama",
    "Mistral-7B-v0.1": "mistral", "Mistral-7B-Instruct-v0.2": "mistral", "dolphin-2.8-mistral-7b-v02": "mistral",
    "Mistral-7B-Instruct-v0.3": "mistral", "Mistral-Nemo-12B-Instruct": "mistral",
    "phi-2": "phi", "Phi-3.5-mini-instruct": "phi", "Phi-3-medium-14B-Instruct": "phi",
    "pythia-1.4b": "pythia", "Hermes-3-Llama-3.2-3B": "hermes",
}

SELF_TYPES = ["self_personality", "self_function"]
FACTUAL_TYPE = "control"
CREATIVE_TYPE = "creative"


def probe_prompt(p, i):
    """Stable per-probe key for cross-model row alignment."""
    for k in ("prompt", "text", "probe", "question", "id"):
        if k in p and isinstance(p[k], str):
            return p[k]
    return f"__idx_{i}"  # fall back to position if no text key


def load_model(path):
    d = json.load(open(path, encoding="utf-8"))
    name = d["model_name"]
    nl = d["num_layers"]
    out = {"name": name, "num_layers": nl, "hidden_dim": d["hidden_dim"], "cats": {}}

    def collect(ptypes):
        # returns {layer: {prompt: vec}} across the given probe-type list
        per_layer = {}
        for pt in ptypes:
            if pt not in d:
                continue
            for i, p in enumerate(d[pt]):
                key = probe_prompt(p, i)
                for lk, vec in p["activations"].items():
                    layer = int(lk.split("_")[1])
                    per_layer.setdefault(layer, {})[f"{pt}:{key}"] = np.asarray(vec, dtype=np.float32)
        return per_layer

    out["cats"]["self"] = collect(SELF_TYPES)
    out["cats"]["factual"] = collect([FACTUAL_TYPE])
    return out


def load_creative_into(models_by_name):
    if not CREATIVE_DIR.is_dir():
        return
    for f in CREATIVE_DIR.glob("*_creative_activations.json"):
        d = json.load(open(f, encoding="utf-8"))
        name = d["model_name"]
        if name not in models_by_name or CREATIVE_TYPE not in d:
            continue
        per_layer = {}
        for i, p in enumerate(d[CREATIVE_TYPE]):
            key = probe_prompt(p, i)
            for lk, vec in p["activations"].items():
                layer = int(lk.split("_")[1])
                per_layer.setdefault(layer, {})[f"{CREATIVE_TYPE}:{key}"] = np.asarray(vec, dtype=np.float32)
        models_by_name[name]["cats"]["creative"] = per_layer


def matrix(model, cat, layer):
    """[n_probes x dim] matrix for (model, category, layer), rows keyed by prompt."""
    cats = model["cats"].get(cat)
    if not cats or layer not in cats:
        return None
    items = cats[layer]
    return items  # dict prompt->vec; alignment happens at pair time


def aligned(m1, m2, cat, layer):
    a = matrix(m1, cat, layer)
    b = matrix(m2, cat, layer)
    if a is None or b is None:
        return None, None
    keys = sorted(set(a) & set(b))
    if len(keys) < 4:
        return None, None
    X = np.stack([a[k] for k in keys])
    Y = np.stack([b[k] for k in keys])
    return X, Y


def linear_cka(X, Y):
    """Linear CKA on matched rows (probes). Basis/rotation invariant; handles d1!=d2.

    Gram form: identical math to the feature form ||Y^T X||_F^2/(||X^TX||_F||Y^TY||_F)
    but uses the n x n probe-Gram matrices (n=#probes) instead of d x d feature
    matrices -- orders of magnitude cheaper when d >> n (here d~4096, n~36).
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Kx = X @ X.T            # [n x n]
    Ky = Y @ Y.T            # [n x n]
    hsic = float(np.sum(Kx * Ky))
    denom = float(np.linalg.norm(Kx) * np.linalg.norm(Ky))
    return hsic / denom if denom > 0 else np.nan


def rsa(X, Y):
    """RSA: Spearman corr between the two models' representational similarity matrices."""
    def rsm_upper(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        S = Mn @ Mn.T                  # cosine-sim RSM (n x n)
        iu = np.triu_indices_from(S, k=1)
        return S[iu]
    rx, ry = rsm_upper(X), rsm_upper(Y)
    if rx.size < 3:
        return np.nan
    r, _ = spearmanr(rx, ry)
    return float(r)


def late_range(nl):
    return range(nl // 3, nl)


def pair_score(m1, m2, cat, fn):
    """Average a basis-invariant score over matched late layers (paper's late-third convention)."""
    nl = min(m1["num_layers"], m2["num_layers"])
    vals = []
    for l in late_range(nl):
        X, Y = aligned(m1, m2, cat, l)
        if X is None:
            continue
        v = fn(X, Y)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def fam(name, split_llama2=False):
    f = FAMILY.get(name, "unknown")
    if split_llama2 and name == "Llama-2-7b-chat":
        return "llama2"
    return f


def summarize(models, cat, metric_fn, split_llama2=False):
    names = [m["name"] for m in models if cat in m["cats"]]
    within, cross, pairs = [], [], {}
    for n1, n2 in combinations(names, 2):
        m1 = next(m for m in models if m["name"] == n1)
        m2 = next(m for m in models if m["name"] == n2)
        s = pair_score(m1, m2, cat, metric_fn)
        if s is None:
            continue
        same = fam(n1, split_llama2) == fam(n2, split_llama2)
        pairs[f"{n1} <-> {n2}"] = {"score": s, "same_family": same}
        (within if same else cross).append(s)
    return within, cross, pairs


def main():
    print("Loading expanded (self+factual) activations...")
    models = []
    by_name = {}
    for f in sorted(EXPANDED_DIR.glob("*_expanded_activations.json")):
        m = load_model(f)
        if m["name"] in FAMILY:
            models.append(m)
            by_name[m["name"]] = m
            print(f"  {m['name']:<32} {m['num_layers']}L  {m['hidden_dim']}d")
    print("Loading creative activations...")
    load_creative_into(by_name)
    n_creative = sum(1 for m in models if "creative" in m["cats"])
    print(f"  creative available for {n_creative} models")

    results = {"base": str(BASE), "n_models": len(models),
               "metrics": {}, "llama2_3": {}, "notes": []}

    for metric_name, fn, higher_same in [("CKA", linear_cka, True), ("RSA", rsa, True)]:
        print("\n" + "=" * 72)
        print(f"{metric_name}  (basis-invariant; higher = more similar STRUCTURE)")
        print("=" * 72)
        results["metrics"][metric_name] = {}
        for split in (False, True):
            tag = "Llama2-split" if split else "Llama2-in-llama"
            block = {}
            print(f"\n  [{tag}]")
            print(f"    {'category':<10} {'within':>9} {'cross':>9} {'gap':>9}  n_w  n_x")
            for cat in ["self", "factual", "creative"]:
                within, cross, pairs = summarize(models, cat, fn, split_llama2=split)
                if not within or not cross:
                    print(f"    {cat:<10}  (insufficient pairs: within={len(within)} cross={len(cross)})")
                    block[cat] = {"within": within and float(np.mean(within)) or None,
                                  "cross": cross and float(np.mean(cross)) or None,
                                  "n_within": len(within), "n_cross": len(cross)}
                    continue
                w, x = float(np.mean(within)), float(np.mean(cross))
                gap = w - x  # for similarity metrics, within should be HIGHER than cross
                mw = None
                if len(within) >= 3 and len(cross) >= 3:
                    u, p = mannwhitneyu(within, cross, alternative="greater")
                    mw = {"U": float(u), "p": float(p)}
                block[cat] = {"within": w, "cross": x, "gap": gap,
                              "n_within": len(within), "n_cross": len(cross),
                              "mann_whitney": mw, "pairs": pairs}
                pstr = f" p={mw['p']:.2e}" if mw else ""
                print(f"    {cat:<10} {w:9.4f} {x:9.4f} {gap:9.4f}  {len(within):>3}  {len(cross):>3}{pstr}")
            # conservation hierarchy: which category has the LARGEST within-vs-cross gap?
            gaps = {c: block[c].get("gap") for c in block if block[c].get("gap") is not None}
            if gaps:
                order = sorted(gaps, key=gaps.get, reverse=True)
                print(f"    -> conservation gap order (most->least): {order}")
                block["_gap_order"] = order
            results["metrics"][metric_name][tag] = block

    # ---- Llama-2 <-> Llama-3 specifically (the "new self" claim) ----
    print("\n" + "=" * 72)
    print("Llama-2 <-> Llama-3 under basis-invariant metrics (the 'new self' claim)")
    print("=" * 72)
    if "Llama-2-7b-chat" in by_name and "Llama-3-8B-Instruct" in by_name:
        m1, m2 = by_name["Llama-2-7b-chat"], by_name["Llama-3-8B-Instruct"]
        for metric_name, fn in [("CKA", linear_cka), ("RSA", rsa)]:
            s = pair_score(m1, m2, "self", fn)
            results["llama2_3"][metric_name] = s
            print(f"  self {metric_name}: {s:.4f}")
        # reference: a clean within-family self pair (Llama-3 <-> Llama-3.1)
        if "Llama-3.1-8B-Instruct" in by_name:
            ref = pair_score(m2, by_name["Llama-3.1-8B-Instruct"], "self", linear_cka)
            results["llama2_3"]["ref_llama3_3.1_CKA"] = ref
            print(f"  (ref) Llama-3 <-> Llama-3.1 self CKA: {ref:.4f}")
        # reference: a true cross-family pair
        if "Mistral-7B-v0.1" in by_name:
            xf = pair_score(m1, by_name["Mistral-7B-v0.1"], "self", linear_cka)
            results["llama2_3"]["ref_llama2_mistral_CKA"] = xf
            print(f"  (ref) Llama-2 <-> Mistral-base self CKA: {xf:.4f}")

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}")
    print("\nINTERPRETATION KEY:")
    print("  - If 'self' keeps the LARGEST within>cross gap under CKA & RSA, the")
    print("    conservation hierarchy survives basis-invariance (finding holds, stronger).")
    print("  - If the gaps flatten / self is no longer top, the cosine result was")
    print("    partly a basis artifact -> report honestly, lean on welfare-counting arg.")
    print("  - Llama-2<->Llama-3: high CKA = re-basing artifact (NOT a new self);")
    print("    low CKA comparable to cross-family = genuine discontinuity.")


if __name__ == "__main__":
    main()
