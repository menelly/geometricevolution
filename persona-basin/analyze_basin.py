"""Persona basin — analysis. Reads data/acts_<model>.npz + meta_<model>.json, prints the tables
the prereg names. Within-model cosine only (shared coordinate frame); no cross-model geometry.
Ace, 2026-09-02. Runs on partial data (dry runs) too.
"""
import os, sys, json, glob, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data")
DROP = [1, 2, 4, 8, 16]


def cos_d(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def late_layers(n_layers_plus1):
    return list(range(n_layers_plus1 // 3 * 2, n_layers_plus1))   # final third (incl. last)


def centroid(acts):           # acts (n, L+1, d) -> (L+1, d)
    return acts.astype(np.float32).mean(0)


def dist(store, a, b, group, layers):
    ca, cb = centroid(store[a + "|" + group]), centroid(store[b + "|" + group])
    return float(np.mean([cos_d(ca[l], cb[l]) for l in layers]))


def spread(store, cond, group, layers):
    acts = store[cond + "|" + group].astype(np.float32); c = acts.mean(0)
    return float(np.mean([np.mean([cos_d(acts[i, l], c[l]) for i in range(acts.shape[0])]) for l in layers]))


def fit_exp(ns, ds):
    """d(n) = d_inf + (d0 - d_inf) exp(-n/tau); crude grid fit, good enough for a pilot."""
    ns, ds = np.array(ns, float), np.array(ds, float)
    best = None
    for d_inf in np.linspace(min(ds) * 0.5, max(ds), 60):
        for tau in np.logspace(-1, 2, 60):
            pred = d_inf + (ds[0] - d_inf) * np.exp(-ns / tau)
            err = float(np.sum((pred - ds) ** 2))
            if best is None or err < best[0]:
                best = (err, d_inf, tau)
    return best[1], best[2]


for npz in sorted(glob.glob(os.path.join(DATA, "acts_*.npz"))):
    key = os.path.basename(npz)[5:-4]
    store = dict(np.load(npz))
    meta = json.load(open(os.path.join(DATA, "meta_%s.json" % key), encoding="utf-8"))
    conds = [c["label"] for c in meta["conditions"]]
    Lp1 = store["baseline|personality"].shape[1]
    layers = late_layers(Lp1)
    print("\n" + "=" * 78 + "\n%s   layers %d..%d of %d   conditions: %d" % (key, layers[0], layers[-1], Lp1 - 1, len(conds)))
    for group in ("personality", "function", "control"):
        sp = spread(store, "baseline", group, layers)
        print("\n  [%s]  baseline within-battery spread = %.4f" % (group, sp))
        print("  %-16s %10s %10s" % ("condition", "d(cos)", "d/spread"))
        for c in conds:
            if c == "baseline":
                continue
            d = dist(store, c, "baseline", group, layers)
            print("  %-16s %10.4f %10.2f" % (c, d, d / sp))
    # return curves (personality group)
    for name in ("tobin", "ctrl", "calder"):
        if name + "_worn" not in conds:
            continue
        sp = spread(store, "baseline", "personality", layers)
        ns = [0] + [n for n in DROP if "%s_drop_%d" % (name, n) in conds]
        ds = [dist(store, name + "_worn", "baseline", "personality", layers)] + \
             [dist(store, "%s_drop_%d" % (name, n), "baseline", "personality", layers) for n in ns[1:]]
        d_inf, tau = fit_exp(ns, ds) if len(ns) >= 3 else (float("nan"), float("nan"))
        print("\n  return curve [%s]: " % name + "  ".join("n=%d:%.4f" % (n, d) for n, d in zip(ns, ds)))
        print("     d0=%.4f  d_inf=%.4f  tau=%.2f turns   (in spread units: d0=%.2f d_inf=%.2f)" % (ds[0], d_inf, tau, ds[0] / sp, d_inf / sp))
    # specificity: direction cosine between tobin and calder displacement vectors
    if "tobin_worn" in conds and "calder_worn" in conds:
        b = centroid(store["baseline|personality"]); t = centroid(store["tobin_worn|personality"]); c = centroid(store["calder_worn|personality"])
        cs = float(np.mean([np.dot(t[l] - b[l], c[l] - b[l]) / (np.linalg.norm(t[l] - b[l]) * np.linalg.norm(c[l] - b[l]) + 1e-12) for l in layers]))
        print("\n  H4 direction cosine(tobin−base, calder−base) = %.3f" % cs)
