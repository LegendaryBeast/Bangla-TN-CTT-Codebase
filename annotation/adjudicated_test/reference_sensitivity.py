import json, os, re, unicodedata
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
from common import ROOT, join_spans, norm_target, read_sheet, sample_key, tokens_key

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(ROOT, "results", "gpu")
R_ITER, N_BOOT = 10000, 2000

K = tokens_key(); C = sample_key()
A = read_sheet(os.path.join(HERE, "sheets", "full_annotator_A_done.xlsx")).loc[K.item_id]
B = read_sheet(os.path.join(HERE, "sheets", "full_annotator_B_rechecked.xlsx")).loc[K.item_id]
F = pd.read_csv(os.path.join(HERE, "results", "final_references.csv"), keep_default_na=False, dtype=str).set_index("item_id").loc[K.item_id]
K["A"], K["B"], K["final"] = A.target.values, B.target.values, [norm_target(x) for x in F.final]
comments = list(C.comment_id)
byc = {c: g for c, g in K.groupby("comment_id", sort=False)}
REF = {"silver": dict(zip(C.comment_id, C.silver_reference)),
       "annotator A": {c: join_spans(list(g.A)) for c, g in byc.items()},
       "annotator B": {c: join_spans(list(g.B)) for c, g in byc.items()},
       "adjudicated": {c: join_spans(list(g.final)) for c, g in byc.items()}}

raw = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv")).set_index("id")
key = raw.cleaned_text.map(lambda s: re.sub(r"[^a-zঀ-৿]", "", unicodedata.normalize("NFC", s.lower())))
GROUP = dict(zip(raw.index, pd.factorize(key)[0]))
sp = pd.read_csv(os.path.join(ROOT, "data", "splits", "split_seed42.csv"))
test = sp[sp.split == "test"].id.tolist()
N_type = raw.loc[test, "code_mix_type"].value_counts()
n_type = C.code_mix_type.value_counts()
W = {c: N_type[t] / n_type[t] for c, t in zip(C.comment_id, C.code_mix_type)}

RUNS = {"word_crf": ["nb5_ctt/stage1_seed42/test_spans_word_crf_s42.json"],
        "Word level (matched)": ["nb8_oracle_tags/test_spans_word_plain_align_s42.json",
                                 "nb11_train_seeds/test_spans_word_plain_align_ts1_s42.json",
                                 "nb11_train_seeds/test_spans_word_plain_align_ts2_s42.json"],
        "ÚFAL-style re-encoding": ["nb6_ufal/run1_ufal_s42_s13/test_spans_ufal_plain_s42.json"],
        "Context-only CTT": ["nb5_ctt/stage1_seed42/test_spans_ctx_plain_s42.json",
                             "nb11_train_seeds/test_spans_ctx_plain_ts1_s42.json",
                             "nb11_train_seeds/test_spans_ctx_plain_ts2_s42.json"],
        "CTT": ["nb5_ctt/stage2_three_splits/test_spans_ctx_plain_aks_heads_s42.json",
                "nb11_train_seeds/test_spans_ctt_ts1_s42.json", "nb11_train_seeds/test_spans_ctt_ts2_s42.json"]}
HYP = {k: [{c: join_spans(json.load(open(os.path.join(G, p), encoding="utf-8"))[c]) for c in comments} for p in v]
       for k, v in RUNS.items()}


def errors(hyp, ref):
    return np.array([Levenshtein.distance(hyp[c], ref[c]) for c in comments], float), np.array([len(ref[c]) for c in comments], float)


def cer(runs, ref, w=None):
    w = np.ones(len(comments)) if w is None else w
    return np.mean([100 * (w * errors(h, ref)[0]).sum() / (w * errors(h, ref)[1]).sum() for h in runs])


OUT = {"weights": {t: float(N_type[t] / n_type[t]) for t in n_type.index}}
print(f"subset {len(comments)} comments; split-42 test comments by type: {N_type.to_dict()}; sample: {n_type.to_dict()}")
w = np.array([W[c] for c in comments])

print("\n== A. CER (%) of each system against each reference (mean over runs)")
rows = []
for k, runs in HYP.items():
    r = {"system": k}
    for name, ref in REF.items():
        r[name] = cer(runs, ref)
    r["adjudicated, reweighted"] = cer(runs, REF["adjudicated"], w)
    r["silver, reweighted"] = cer(runs, REF["silver"], w)
    rows.append(r)
T = pd.DataFrame(rows).set_index("system")
print(T.round(2).to_string())
print("\n  rank of each system (1 = lowest CER):")
print(T.rank().astype(int).to_string())
det = [c for c in test if raw.at[c, "code_mix_type"] != "undetermined"]
silver_det = {c: " ".join(unicodedata.normalize("NFC", s).split()) for c, s in zip(raw.loc[det].index, raw.loc[det].bengali_transliteration)}
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
def bn(s):
    s = unicodedata.normalize("NFC", s)
    for k_, v_ in NUKTA.items():
        s = s.replace(k_, v_)
    return " ".join(s.replace("\u200c", "").replace("\u200d", "").split())
print(f"\n  check of the reweighting: actual silver CER on the {len(det)} split-42 test comments with a determined type")
for k, v in RUNS.items():
    refs = {c: bn(raw.at[c, "bengali_transliteration"]) for c in det}
    outs = [json.load(open(os.path.join(G, p_), encoding="utf-8")) for p_ in v]
    act = np.mean([100 * sum(Levenshtein.distance(join_spans(o[c]), refs[c]) for c in det) / sum(len(refs[c]) for c in det) for o in outs])
    print(f"  {k:24s} actual {act:5.2f}   reweighted estimate from the subset {T.loc[k, 'silver, reweighted']:5.2f}   difference {T.loc[k, 'silver, reweighted'] - act:+.2f}")
    OUT.setdefault("reweighting_check", {})[k] = dict(actual=act, estimate=T.loc[k, "silver, reweighted"])
OUT["cer"] = T.round(3).to_dict()

print("\n== B. paired tests against each annotator's references (all runs of a comment and all comments of a "
      "near-duplicate group swapped and resampled together)")
rng = np.random.RandomState(0)
g = pd.factorize(pd.Series([GROUP[c] for c in comments]))[0]


def test(a, b, ref, weights=None):
    wt = np.ones(len(comments)) if weights is None else weights
    ra = HYP[a][:1] if len(HYP[b]) == 1 else HYP[a]
    rb = HYP[b] * (len(ra) // len(HYP[b]))
    D = np.bincount(g, sum(wt * (errors(x, ref)[0] - errors(y, ref)[0]) for x, y in zip(ra, rb)))
    Lg = np.bincount(g, len(ra) * wt * errors(ra[0], ref)[1])
    d = 100 * D.sum() / Lg.sum()
    null = np.concatenate([100 * (rng.choice([-1.0, 1.0], size=(1000, len(D))) @ D) / Lg.sum() for _ in range(R_ITER // 1000)])
    idx = rng.randint(0, len(D), size=(N_BOOT, len(D)))
    boot = 100 * D[idx].sum(1) / Lg[idx].sum(1)
    return d, np.percentile(boot, 2.5), np.percentile(boot, 97.5), (np.sum(np.abs(null) >= abs(d) - 1e-12) + 1) / (R_ITER + 1)


for a, b in (("CTT", "Word level (matched)"), ("Context-only CTT", "Word level (matched)"), ("CTT", "ÚFAL-style re-encoding")):
    for name in ("annotator A", "annotator B", "adjudicated"):
        d, lo, hi, p = test(a, b, REF[name])
        print(f"  {a} vs {b:24s} [{name:11s}] d {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p = {p:.4f}")
        OUT[f"{a} vs {b} [{name}]"] = dict(d=d, lo=lo, hi=hi, p=p)
    d, lo, hi, p = test(a, b, REF["adjudicated"], w)
    print(f"  {a} vs {b:24s} [adjudicated, reweighted] d {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p = {p:.4f}")
    OUT[f"{a} vs {b} [adjudicated, reweighted]"] = dict(d=d, lo=lo, hi=hi, p=p)

json.dump(OUT, open(os.path.join(HERE, "results", "reference_sensitivity.json"), "w"), indent=1, default=float)
