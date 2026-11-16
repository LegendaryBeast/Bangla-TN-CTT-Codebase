import contextlib, io, json, os, re, sys, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis")); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import review_analyses as RA
from rapidfuzz.distance import Levenshtein

R_ITER, N_BOOT, BUDGET = 10000, 2000, 160
G = os.path.join(ROOT, "results", "gpu")
df, ref, toks, S = RA.df, RA.df["ref"], RA.df["tokens"], RA.SEEDS
raw = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv")).set_index("id")
key = raw.cleaned_text.map(lambda s: re.sub(r"[^a-zঀ-৿]", "", unicodedata.normalize("NFC", s.lower())))
GROUP = dict(zip(raw.index, pd.factorize(key)[0]))
SYS = {"word_crf": "word_crf", "ufal_plain": "re-encoding", "ctx_plain": "context-only", "ctx_plain_aks_heads": "CTT"}
OUT = {}


def load(p):
    return json.load(open(os.path.join(G, p), encoding="utf-8"))


def errs(sp, ids):
    ce, ln = [], []
    for c in ids:
        ce.append(Levenshtein.distance(RA.join_spans(sp[c]), ref[c])); ln.append(len(ref[c]))
    return pd.DataFrame(dict(ce=ce, ln=ln), index=ids)


def clustered(pairs, num="ce", den="ln", seed=0):
    rng = np.random.RandomState(seed)
    cid = np.concatenate([a.index.values for a, _ in pairs])
    d = np.concatenate([(a[num] - b[num]).values for a, b in pairs]).astype(float)
    L = np.concatenate([a[den].values for a, _ in pairs]).astype(float)
    g = pd.factorize(pd.Series([GROUP[c] for c in cid]))[0]
    D, Lg = np.bincount(g, d), np.bincount(g, L)
    stat = 100 * D.sum() / Lg.sum()
    hits = 0
    for k in range(0, R_ITER, 1000):
        sg = rng.choice([-1.0, 1.0], size=(min(1000, R_ITER - k), len(D)))
        hits += np.sum(np.abs(100 * (sg @ D) / Lg.sum()) >= abs(stat) - 1e-12)
    idx = rng.randint(0, len(D), size=(N_BOOT, len(D)))
    boot = 100 * D[idx].sum(1) / Lg[idx].sum(1)
    return dict(d=stat, lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), p=(hits + 1) / (R_ITER + 1))


def show(name, r):
    print(f"  {name:62s} d {r['d']:+.2f}  95% CI [{r['lo']:+.2f}, {r['hi']:+.2f}]  p = {r['p']:.4f}")
    OUT[name] = {k: float(v) for k, v in r.items()}


def cer(pairs):
    return 100 * sum(f.loc[i, "ce"].sum() for f, i in pairs) / sum(f.loc[i, "ln"].sum() for f, i in pairs)


print(f"== A. chunking of comments ({BUDGET}-byte budget unless stated)")
for b in (160, 320, 480, 640):
    n = df.tokens.map(lambda t: len(RA.chunk_words(t, b)))
    w = df.tokens.map(len)
    print(f"  budget {b:3d}: {100 * (n > 1).mean():5.1f}% of comments cut, holding {100 * w[n > 1].sum() / w.sum():5.1f}% of words")
nch = {s: {c: len(RA.chunk_words(toks[c], BUDGET)) for c in RA.TEST[s]} for s in S}
multi = {s: [c for c in RA.TEST[s] if nch[s][c] > 1] for s in S}
single = {s: [c for c in RA.TEST[s] if nch[s][c] == 1] for s in S}
aligned = {s: [c for c in RA.TEST[s] if len(toks[c]) == len(ref[c].split())] for s in S}
unaligned = {s: [c for c in RA.TEST[s] if len(toks[c]) != len(ref[c].split())] for s in S}
print(f"  test comments (three splits): {sum(map(len, multi.values()))} multi-chunk, {sum(map(len, single.values()))} single-chunk; "
      f"{sum(map(len, unaligned.values()))} not token-aligned, {sum(map(len, aligned.values()))} aligned")

E = {(v, s): errs(RA.SPANS[(v, s)], RA.TEST[s]) for v in SYS for s in S}
RUNS = {"CTT": ["nb5_ctt/stage2_three_splits/test_spans_ctx_plain_aks_heads_s42.json",
                "nb11_train_seeds/test_spans_ctt_ts1_s42.json", "nb11_train_seeds/test_spans_ctt_ts2_s42.json"],
        "word": ["nb8_oracle_tags/test_spans_word_plain_align_s42.json",
                 "nb11_train_seeds/test_spans_word_plain_align_ts1_s42.json",
                 "nb11_train_seeds/test_spans_word_plain_align_ts2_s42.json"]}
RS = {k: [load(p) for p in v] for k, v in RUNS.items()}
ids42 = RA.TEST[42]
RE = {k: [errs(sp, ids42) for sp in v] for k, v in RS.items()}

for title, A_, B_, an, bn in (("B. single-chunk vs multi-chunk comments", single, multi, "single-chunk", "multi-chunk"),
                              ("D. token-aligned vs not token-aligned comments", aligned, unaligned, "aligned", "not aligned")):
    print(f"\n== {title}: CER (%), three splits pooled, one run each")
    for v, name in SYS.items():
        print(f"  {name:13s} {an} {cer([(E[(v, s)], A_[s]) for s in S]):5.2f}   {bn} {cer([(E[(v, s)], B_[s]) for s in S]):5.2f}")
    for sub, lab in ((A_, an), (B_, bn)):
        for a, b in (("ctx_plain_aks_heads", "word_crf"), ("ctx_plain_aks_heads", "ufal_plain")):
            show(f"{lab}: {SYS[a]} vs {SYS[b]} (3 splits)", clustered([(E[(a, s)].loc[sub[s]], E[(b, s)].loc[sub[s]]) for s in S]))
        show(f"{lab}: CTT vs matched word (split 42, 3 runs)",
             clustered([(x.loc[sub[42]], y.loc[sub[42]]) for x, y in zip(RE["CTT"], RE["word"])]))

print("\n== C. token accuracy by position relative to chunk boundaries (token-aligned test comments, no guard)")
rows = []
for s in S:
    for c in aligned[s]:
        ch = RA.chunk_words(toks[c], BUDGET); r_ = ref[c].split(); where = {}
        for k, idx in enumerate(ch):
            for j, i in enumerate(idx):
                cut = (j == 0 and k > 0) or (j == len(idx) - 1 and k < len(ch) - 1)
                where[i] = "boundary" if cut else ("inside multi-chunk" if len(ch) > 1 else "single chunk")
        for i, w in enumerate(toks[c]):
            row = dict(split=s, cid=c, position=where[i])
            for v in SYS:
                row[v] = float(RA.bn(RA.SPANS[(v, s)][c][i]) == r_[i])
            if s == 42:
                for k_, runs in RS.items():
                    row[k_ + "_runs"] = float(np.mean([RA.bn(sp[c][i]) == r_[i] for sp in runs]))
            rows.append(row)
T = pd.DataFrame(rows)
acc = T.groupby("position")[list(SYS)].mean().mul(100).rename(columns=SYS)
acc["tokens"] = T.groupby("position").size()
acc["share %"] = 100 * acc["tokens"] / acc["tokens"].sum()
acc["CTT - word_crf"] = acc["CTT"] - acc["word_crf"]; acc["CTT - re-encoding"] = acc["CTT"] - acc["re-encoding"]
print(acc.round(2).to_string())
for pos in ("boundary", "inside multi-chunk", "single chunk"):
    x = T[T.position == pos].set_index("cid")
    one = pd.Series(1.0, index=x.index)
    for a, b in (("ctx_plain_aks_heads", "word_crf"), ("ctx_plain_aks_heads", "ufal_plain")):
        show(f"token accuracy {pos}: {SYS[a]} vs {SYS[b]}", clustered([(pd.DataFrame(dict(ce=x[a], ln=one)), pd.DataFrame(dict(ce=x[b], ln=one)))]))
lab = {(s, c): df.at[c, "labels"] for s in S for c in aligned[s]}
T["tag"] = [lab[(s, c)][i] for (s, c), grp in T.groupby(["split", "cid"], sort=False) for i in range(len(grp))]
T["cm"] = [bool(raw.at[c, "likely_code_mixed"]) for c in T.cid]
T["chunks"] = np.where(T.position == "single chunk", "single", "multi")
comp = pd.crosstab(T.chunks, T.tag, normalize="index").mul(100)
comp["code-mixed comments %"] = T.groupby("chunks").cm.mean().mul(100)
print("  tag mix (% of tokens) and share of tokens in code-mixed comments:"); print(comp.round(1).to_string())
T["gain"] = T.ctx_plain_aks_heads - T.word_crf
gt = T.groupby(["tag", "chunks"]).gain.agg(["mean", "size"]); gt["mean"] *= 100
print("  CTT - word_crf token-accuracy gain by tag:"); print(gt.round(2).to_string())
w = T[T.chunks == "single"].tag.value_counts(normalize=True)
adj = sum(w[t_] * 100 * T[(T.chunks == "multi") & (T.tag == t_)].gain.mean() for t_ in w.index)
print(f"  multi-chunk gain reweighted to the single-chunk tag mix: {adj:.2f} points")
OUT["composition"] = comp.round(2).to_dict(); OUT["multi_gain_reweighted"] = float(adj)
T42 = T[T.split == 42]
a42 = T42.groupby("position")[["CTT_runs", "word_runs"]].mean().mul(100)
a42["tokens"] = T42.groupby("position").size(); a42["gain"] = a42.CTT_runs - a42.word_runs
print("  split 42, mean over three runs, CTT vs matched word level:"); print(a42.round(2).to_string())
T42 = T42.assign(g42=T42.CTT_runs - T42.word_runs)
g42 = T42.groupby(["tag", "chunks"]).g42.agg(["mean", "size"]); g42["mean"] *= 100
print("  split 42, gain over matched word level by tag:"); print(g42.round(2).to_string())
w42 = T42[T42.chunks == "single"].tag.value_counts(normalize=True)
adj42 = sum(w42[t_] * 100 * T42[(T42.chunks == "multi") & (T42.tag == t_)].g42.mean() for t_ in w42.index)
print(f"  split 42: single-chunk gain {100 * T42[T42.chunks == 'single'].g42.mean():.2f}; multi-chunk gain "
      f"{100 * T42[T42.chunks == 'multi'].g42.mean():.2f}, reweighted to the single-chunk tag mix {adj42:.2f}")
OUT["positions"] = acc.round(2).to_dict(); OUT["positions42"] = a42.round(2).to_dict(); OUT["multi_gain_reweighted_42"] = float(adj42)

print("\n== E. training with and without aligner-recovered comments (split 42, single runs, CRF-tagged word level)")
st1 = "nb5_ctt/stage1_seed42/test_spans_{}_s42.json"
wa, wn = errs(load(st1.format("word_crf_align")), ids42), errs(load(st1.format("word_crf")), ids42)
wa2 = errs(load("nb8_oracle_tags/test_spans_word_crf_align_rerun_s42.json"), ids42)
for lab, sub in (("all", ids42), ("aligned", aligned[42]), ("not aligned", unaligned[42])):
    print(f"  {lab:12s} with aligner {cer([(wa, sub)]):5.2f}  without {cer([(wn, sub)]):5.2f}  ({len(sub)} comments)")
    show(f"{lab}: word level with vs without recovered training comments", clustered([(wa.loc[sub], wn.loc[sub])]))
    print(f"  {'':12s} second run with aligner {cer([(wa2, sub)]):5.2f}")
    show(f"{lab}: two runs with vs one run without recovered comments", clustered([(wa.loc[sub], wn.loc[sub]), (wa2.loc[sub], wn.loc[sub])]))

json.dump(OUT, open(os.path.join(ROOT, "analysis", "chunk_alignment_sensitivity.json"), "w"), indent=1, default=float)
