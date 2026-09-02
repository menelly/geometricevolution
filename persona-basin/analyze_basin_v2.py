"""Persona basin — analysis v2, written AFTER the first dry run showed that raw final-token centroid
distance is dominated by "there is a system prompt in context" (control prompt moves the readout
almost as much as the persona; factual prompts move MORE than self prompts). So the self-specific
quantities here cancel the shared context shift:

  1. per-layer displacement profile, personality vs function vs control   → choose the band
  2. SIGNATURE rotation: s(c) = centroid_personality(c) − centroid_control(c). The self-signature
     is the direction that separates self-processing from factual processing INSIDE a condition.
     rot(c) = cosine distance(s(c), s(baseline)). Shared context shift cancels in the subtraction.
  3. within-model RSA: Spearman ρ between the upper triangles of the 46×46 inter-prompt cosine
     matrices, baseline vs condition. ρ≈1 = same internal shape (translated), ρ≪1 = restructured.
  4. differential displacement: d_personality − d_control (persona effect net of context effect).
Runs on partial (dry-run) data. Within-model only.
"""
import os, sys, json, glob
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data")
GROUPS = ("personality", "function", "control")


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n


def cosd(a, b):
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load(key):
    store = {k: v.astype(np.float32) for k, v in np.load(os.path.join(DATA, "acts_%s.npz" % key)).items()}
    meta = json.load(open(os.path.join(DATA, "meta_%s.json" % key), encoding="utf-8"))
    return store, [c["label"] for c in meta["conditions"]]


def cent(store, cond, group):          # (L+1, d)
    return store[cond + "|" + group].mean(0)


def all_acts(store, cond):             # (46, L+1, d) in fixed group order
    return np.concatenate([store[cond + "|" + g] for g in GROUPS], 0)


def rsa(store, a, b, layer):
    A, B = unit(all_acts(store, a)[:, layer]), unit(all_acts(store, b)[:, layer])
    SA, SB = A @ A.T, B @ B.T
    iu = np.triu_indices(SA.shape[0], 1)
    return float(spearmanr(SA[iu], SB[iu]).correlation)


for npz in sorted(glob.glob(os.path.join(DATA, "acts_*.npz"))):
    key = os.path.basename(npz)[5:-4]
    store, conds = load(key)
    Lp1 = store["baseline|personality"].shape[1]
    print("\n" + "=" * 78 + "\n%s  (%d layers)" % (key, Lp1 - 1))

    # 1. per-layer displacement profile for two informative conditions
    print("\n  1. per-layer cosine distance to baseline centroid  (P=personality F=function C=control)")
    print("     layer | %-24s | %-24s" % ("tobin_D3  P     F     C", "ctrl_D3   P     F     C"))
    for l in range(0, Lp1, max(1, Lp1 // 16)):
        row = []
        for c in ("tobin_D3", "ctrl_D3"):
            if c not in conds:
                row.append("   -     -     -   "); continue
            row.append(" ".join("%5.3f" % cosd(cent(store, c, g)[l], cent(store, "baseline", g)[l]) for g in GROUPS))
        print("     %5d | %-24s | %-24s" % (l, row[0], row[1]))

    # choose band candidates: late third, middle third, all
    bands = {"late3": list(range(Lp1 // 3 * 2, Lp1)), "mid3": list(range(Lp1 // 3, Lp1 // 3 * 2)), "all": list(range(1, Lp1))}
    for bname, layers in bands.items():
        print("\n  band %-6s layers %d..%d" % (bname, layers[0], layers[-1]))
        print("     %-14s %8s %8s %8s | %9s %8s | %7s" % ("condition", "d_pers", "d_func", "d_ctrl", "sig_rot", "d_P-d_C", "RSA46"))
        for c in conds:
            if c in ("baseline", "std_baseline"):
                continue
            # Arm B conditions (std_* and *_t1*) are measured against Arm B's own floor
            ref = "std_baseline" if (c.startswith("std_") or "_t1" in c) and "std_baseline" in conds else "baseline"
            sb = cent(store, ref, "personality") - cent(store, ref, "control")
            dP = np.mean([cosd(cent(store, c, "personality")[l], cent(store, ref, "personality")[l]) for l in layers])
            dF = np.mean([cosd(cent(store, c, "function")[l], cent(store, ref, "function")[l]) for l in layers])
            dC = np.mean([cosd(cent(store, c, "control")[l], cent(store, ref, "control")[l]) for l in layers])
            sc = cent(store, c, "personality") - cent(store, c, "control")
            rot = np.mean([cosd(sc[l], sb[l]) for l in layers])
            r = np.mean([rsa(store, c, ref, l) for l in layers])
            print("     %-14s %8.4f %8.4f %8.4f | %9.4f %8.4f | %7.3f   (vs %s)" % (c, dP, dF, dC, rot, dP - dC, r, ref))
        if "std_baseline" in conds:
            sb0 = cent(store, "baseline", "personality") - cent(store, "baseline", "control")
            sb1 = cent(store, "std_baseline", "personality") - cent(store, "std_baseline", "control")
            print("     %-14s signature rotation between the two floors = %.4f   RSA = %.3f" % (
                "floor-vs-floor", np.mean([cosd(sb1[l], sb0[l]) for l in layers]), np.mean([rsa(store, "std_baseline", "baseline", l) for l in layers])))
