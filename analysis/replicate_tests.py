import contextlib, io, json, os, sys, unicodedata
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis")); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import review_analyses as RA
from rapidfuzz.distance import Levenshtein

G = os.path.join(ROOT, "results", "gpu")
ref, S = RA.df["ref"], RA.SEEDS
OUT = {}


def cer(sp, ids):
    return 100 * sum(Levenshtein.distance(RA.join_spans(sp[c]), ref[c]) for c in ids) / sum(len(ref[c]) for c in ids)


def load(p):
    return json.load(open(os.path.join(G, p), encoding="utf-8"))


def paired(name, a, b):
    d = np.array(a) - np.array(b)
    t, p = stats.ttest_rel(a, b)
    print(f"  {name:52s} differences {np.round(d, 2).tolist()}  mean {d.mean():+.2f}  t = {t:+.2f}  p = {p:.3f}  "
          f"first lower in {int((d < 0).sum())} of {len(d)}")
    OUT[name] = dict(diffs=d.tolist(), mean=float(d.mean()), t=float(t), p=float(p))


def unpaired(name, a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    print(f"  {name:52s} means {np.mean(a):.2f} vs {np.mean(b):.2f}  Welch t = {t:+.2f}  p = {p:.3f}  "
          f"Mann-Whitney exact p = {u.pvalue:.3f}  (runs {np.round(a, 2).tolist()} vs {np.round(b, 2).tolist()})")
    OUT[name] = dict(a=list(map(float, a)), b=list(map(float, b)), welch_t=float(t), welch_p=float(p), mw_p=float(u.pvalue))


print("== A. split level: one training run per split, so the three differences include training and split variation")
C = {v: [cer(RA.SPANS[(v, s)], RA.TEST[s]) for s in S] for v in RA.SYSTEMS}
for v in RA.SYSTEMS:
    print(f"  {v:22s} CER per split {np.round(C[v], 2).tolist()}")
paired("CTT vs re-encoding", C["ctx_plain_aks_heads"], C["ufal_plain"])
paired("context-only vs re-encoding", C["ctx_plain"], C["ufal_plain"])
paired("CTT vs word_crf", C["ctx_plain_aks_heads"], C["word_crf"])
paired("context-only vs word_crf", C["ctx_plain"], C["word_crf"])

print("\n== B. split 42: three training runs per system, compared as independent replicates")
ids42 = RA.TEST[42]
RUNS = {"CTT": ["nb5_ctt/stage2_three_splits/test_spans_ctx_plain_aks_heads_s42.json",
                "nb11_train_seeds/test_spans_ctt_ts1_s42.json", "nb11_train_seeds/test_spans_ctt_ts2_s42.json"],
        "ctx": ["nb5_ctt/stage1_seed42/test_spans_ctx_plain_s42.json",
                "nb11_train_seeds/test_spans_ctx_plain_ts1_s42.json", "nb11_train_seeds/test_spans_ctx_plain_ts2_s42.json"],
        "word": ["nb8_oracle_tags/test_spans_word_plain_align_s42.json",
                 "nb11_train_seeds/test_spans_word_plain_align_ts1_s42.json",
                 "nb11_train_seeds/test_spans_word_plain_align_ts2_s42.json"]}
RC = {k: [cer(load(p), ids42) for p in v] for k, v in RUNS.items()}
unpaired("CTT vs matched word level", RC["CTT"], RC["word"])
unpaired("context-only vs matched word level", RC["ctx"], RC["word"])
ufal42 = cer(RA.SPANS[("ufal_plain", 42)], ids42)
print(f"  re-encoding, single split-42 run: {ufal42:.2f}; CTT runs {np.round(RC['CTT'], 2).tolist()}; "
      f"context-only runs {np.round(RC['ctx'], 2).tolist()}")
OUT["ufal42"] = ufal42

print("\n== C. MultiLexNorm: three training seeds per system, compared as independent replicates (ERR, %)")
M = os.path.join(G, "nb9_multilexnorm")
for Lg in ("iden", "trde"):
    d = pd.read_csv(os.path.join(M, f"mln_{Lg}.csv"), keep_default_na=False); d = d[d.id.str.startswith(f"{Lg}_test_")]
    data = [(c, [t.rsplit("/", 1)[0] for t in tl.split(" ")], json.loads(sp)) for c, tl, sp in zip(d.id, d.token_labels, d.target_spans)]

    def err_rate(sysn, seed):
        sp = json.load(open(os.path.join(M, f"test_spans_{Lg}_{sysn}_s{seed}.json"), encoding="utf-8"))
        nfc = lambda x: unicodedata.normalize("NFC", x)
        cor = sum(nfc(a) == nfc(b) for c, _, g in data for a, b in zip(g, sp[c]))
        base = sum(r == g_ for _, t, g in data for r, g_ in zip(t, g))
        n = sum(len(g) for _, _, g in data)
        return 100 * (cor - base) / (n - base)
    R = {s_: [err_rate(s_, sd) for sd in (13, 42, 2026)] for s_ in ("ctx_copy", "ufal", "ctx", "word")}
    unpaired(f"{Lg}: CTT vs re-encoding", R["ctx_copy"], R["ufal"])
    unpaired(f"{Lg}: context-only vs word level", R["ctx"], R["word"])

json.dump(OUT, open(os.path.join(ROOT, "analysis", "replicate_tests.json"), "w"), indent=1)
