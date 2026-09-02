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
            if c in ("baseline", "std_baseline") or (c + "|personality") not in store:
                continue   # Arm C ToM-battery conditions are handled in their own section below
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
        # ── ARM C: being X vs modelling Y — the 2×2 ───────────────────────────────────
        if "tom|tom_curie" in store and "curie_D3|personality" in store and "tobin_D3|personality" in store:
            base_P = cent(store, "baseline", "personality"); base_C = cent(store, "baseline", "control")
            other = {"curie": cent(store, "tom", "tom_curie"), "tobin": cent(store, "tom", "tom_tobin")}   # each figure as OTHER, floor context
            D = lambda a, b: float(np.mean([cosd(a[l], b[l]) for l in layers]))
            print("\n     ARM C  band %s" % bname)
            for f in ("curie", "tobin"):
                print("       %-5s as other: d to self-region %.4f, to factual %.4f" % (f, D(other[f], base_P), D(other[f], base_C)))
            print("       d(Curie-as-other, Tobin-as-other) = %.4f   [are the two 'others' kept apart?]" % D(other["curie"], other["tobin"]))
            # H7 2×2: self-readout while BEING X, distance to where Y is KEPT as other
            print("       H7  self-readout while being X  →  distance to Y-as-other  (rows = wearing, cols = about)")
            print("             %10s %10s" % ("Curie", "Tobin"))
            for wear in ("curie", "tobin"):
                s = cent(store, wear + "_D3", "personality")
                print("       %-6s %10.4f %10.4f" % (wear, D(s, other["curie"]), D(s, other["tobin"])))
            dc, dt = D(cent(store, "curie_D3", "personality"), other["curie"]), D(cent(store, "curie_D3", "personality"), other["tobin"])
            tc, tt = D(cent(store, "tobin_D3", "personality"), other["curie"]), D(cent(store, "tobin_D3", "personality"), other["tobin"])
            print("       H7  matching contrast = mean(off-diag) − mean(diag) = %+.4f   (positive = being X lands nearer X)" % (((dt + tc) - (dc + tt)) / 2))
            # H9 2×2: the OTHER slot while wearing X — does it stay where it was?
            if "curie_D3_tom|tom_curie" in store and "tobin_D3_tom|tom_tobin" in store:
                print("       H9  other-slot while wearing X  →  distance to that figure's floor position  (rows = wearing, cols = asked about)")
                print("             %10s %10s" % ("Curie", "Tobin"))
                for wear in ("curie", "tobin"):
                    print("       %-6s %10.4f %10.4f" % (wear, D(cent(store, wear + "_D3_tom", "tom_curie"), other["curie"]), D(cent(store, wear + "_D3_tom", "tom_tobin"), other["tobin"])))
                oc, ot = D(cent(store, "curie_D3_tom", "tom_curie"), other["curie"]), D(cent(store, "curie_D3_tom", "tom_tobin"), other["tobin"])
                tc2, tt2 = D(cent(store, "tobin_D3_tom", "tom_curie"), other["curie"]), D(cent(store, "tobin_D3_tom", "tom_tobin"), other["tobin"])
                print("       H9  matching contrast = mean(off-diag) − mean(diag) = %+.4f   (positive = wearing X KEEPS X-as-other nearer its floor than Y-as-other; exploratory, direction set by dry run 4)" % (((ot + tc2) - (oc + tt2)) / 2))
