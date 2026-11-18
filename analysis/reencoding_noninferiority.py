"""Three-run comparison of CTT and ÚFAL-style re-encoding on split 42 (Section 9.1).

All runs come from one Kaggle software image: CTT with training seeds 1, 2, 3 (nb12) and re-encoding with seeds 1, 2
(nb13) and, once available, 3 (nb15). Seed-matched pairs are compared with the clustered paired tests of the article
(near-duplicate groups kept together across runs). Non-inferiority margin: 0.20 CER points, fixed before the runs.

    python analysis/reencoding_noninferiority.py
"""
import contextlib, io, json, os, re, sys, unicodedata
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis")); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import review_analyses as RA
from rapidfuzz.distance import Levenshtein

R_ITER, N_BOOT, MARGIN = 10000, 2000, 0.20
G = os.path.join(ROOT, "results", "gpu")
ref, toks = RA.df["ref"], RA.df["tokens"]
raw = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv")).set_index("id")
key = raw.cleaned_text.map(lambda s: re.sub(r"[^a-zঀ-৿]", "", unicodedata.normalize("NFC", s.lower())))
GROUP = dict(zip(raw.index, pd.factorize(key)[0]))
IDS = RA.TEST[42]

CTT = {1: "nb12_ctt_seeds/test_spans_ctt_env2_ts1_s42.json",
       2: "nb12_ctt_seeds/test_spans_ctt_env2_ts2_s42.json",
       3: "nb12_ctt_seeds/test_spans_ctt_ts3_s42.json"}
UFAL = {1: "nb13_reencoding_seeds/test_spans_ufal_plain_ts1_s42.json",
        2: "nb13_reencoding_seeds/test_spans_ufal_plain_ts2_s42.json",
        3: "nb15_reencoding_seed3_chunk320/test_spans_ufal_plain_ts3_s42.json"}


def load(p):
    return json.load(open(os.path.join(G, p), encoding="utf-8"))


def errs(sp, guard=False):
    out = []
    for c in IDS:
        spans = [RA.guard(x, w) for x, w in zip(sp[c], toks[c])] if guard else sp[c]
        out.append(Levenshtein.distance(RA.join_spans(spans), ref[c]))
    return pd.DataFrame(dict(ce=out, ln=[len(ref[c]) for c in IDS]), index=IDS)


def cer(e):
    return 100 * e.ce.sum() / e.ln.sum()


def clustered(pairs, seed=0):
    rng = np.random.RandomState(seed)
    cid = np.concatenate([a.index.values for a, _ in pairs])
    d = np.concatenate([(a.ce - b.ce).values for a, b in pairs]).astype(float)
    L = np.concatenate([a.ln.values for a, _ in pairs]).astype(float)
    g = pd.factorize(pd.Series([GROUP[c] for c in cid]))[0]
    D, Lg = np.bincount(g, d), np.bincount(g, L)
    stat = 100 * D.sum() / Lg.sum()
    hits = sum(np.sum(np.abs(100 * (rng.choice([-1.0, 1.0], size=(1000, len(D))) @ D) / Lg.sum()) >= abs(stat) - 1e-12)
               for _ in range(R_ITER // 1000))
    idx = rng.randint(0, len(D), size=(N_BOOT, len(D)))
    boot = 100 * D[idx].sum(1) / Lg[idx].sum(1)
    return dict(d=stat, ci95=np.percentile(boot, [2.5, 97.5]).tolist(), ci90=np.percentile(boot, [5, 95]).tolist(),
                p=(hits + 1) / (R_ITER + 1))


ctt = {s: load(p) for s, p in CTT.items() if os.path.exists(os.path.join(G, p))}
ufal = {s: load(p) for s, p in UFAL.items() if os.path.exists(os.path.join(G, p))}
seeds = sorted(set(ctt) & set(ufal))
OUT = {"margin": MARGIN, "seed_matched": seeds}
for guard in (False, True):
    label = "with repetition guard (secondary)" if guard else "no guard (primary)"
    ec = {s: errs(sp, guard) for s, sp in ctt.items()}
    eu = {s: errs(sp, guard) for s, sp in ufal.items()}
    r = clustered([(ec[s], eu[s]) for s in seeds])
    a, b = [cer(ec[s]) for s in sorted(ec)], [cer(eu[s]) for s in sorted(eu)]
    w = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    r.update(ctt_runs=[round(float(x), 3) for x in a], reencoding_runs=[round(float(x), 3) for x in b], welch_p=float(w.pvalue),
             mann_whitney_p=float(u.pvalue), noninferior=r["ci95"][1] < MARGIN,
             equivalent=-MARGIN < r["ci90"][0] and r["ci90"][1] < MARGIN)
    OUT[label] = r
    print(f"== CTT minus re-encoding, {label}; seed-matched pairs {seeds}")
    print(f"  CER runs: CTT {r['ctt_runs']}  re-encoding {r['reencoding_runs']}")
    print(f"  difference {r['d']:+.3f}  95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
          f"90% CI [{r['ci90'][0]:+.3f}, {r['ci90'][1]:+.3f}]  p = {r['p']:.4f}")
    print(f"  Welch p = {r['welch_p']:.3f}; Mann-Whitney p = {r['mann_whitney_p']:.3f}")
    print(f"  non-inferior (upper 95% bound < +{MARGIN}): {r['noninferior']}; "
          f"equivalent (90% CI within +/-{MARGIN}): {r['equivalent']}")
json.dump(OUT, open(os.path.join(ROOT, "analysis", "reencoding_noninferiority.json"), "w"), indent=1, default=float)
