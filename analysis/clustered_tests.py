import contextlib, io, json, os, pickle, re, sys, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis")); sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import review_analyses as RA
from rapidfuzz.distance import Levenshtein

R_ITER, N_BOOT = 10000, 2000
G = os.path.join(ROOT, "results", "gpu")
df, ref, toks, S = RA.df, RA.df["ref"], RA.df["tokens"], RA.SEEDS
raw = pd.read_csv(os.path.join(ROOT, "data", "BanglaMix.csv")).set_index("id")
key = raw.cleaned_text.map(lambda s: re.sub(r"[^a-zঀ-৿]", "", unicodedata.normalize("NFC", s.lower())))
GROUP = dict(zip(raw.index, pd.factorize(key)[0]))


def load(p):
    return json.load(open(os.path.join(G, p), encoding="utf-8"))


def errs(sp, ids, guarded=False):
    ce, ln, we, wn, em = [], [], [], [], []
    for c in ids:
        s = [RA.guard(x, w) for x, w in zip(sp[c], toks[c])] if guarded else sp[c]
        h, r = RA.join_spans(s), ref[c]
        ce.append(Levenshtein.distance(h, r)); ln.append(len(r))
        we.append(Levenshtein.distance(h.split(), r.split())); wn.append(len(r.split())); em.append(float(h == r))
    return pd.DataFrame(dict(ce=ce, ln=ln, we=we, wn=wn, em=em), index=ids)


def text_errs(path, ids):
    tx = load(path)
    ce = [Levenshtein.distance(" ".join(RA.bn(tx[c]).split()), ref[c]) for c in ids]
    return pd.DataFrame(dict(ce=ce, ln=[len(ref[c]) for c in ids]), index=ids)


def clustered(pairs, num="ce", den="ln", seed=0):
    rng = np.random.RandomState(seed)
    cid = np.concatenate([a.index.values for a, _ in pairs])
    d = np.concatenate([(a[num] - b[num]).values for a, b in pairs]).astype(float)
    L = np.concatenate([a[den].values for a, _ in pairs]).astype(float)
    g = pd.factorize(pd.Series([GROUP.get(c, c) for c in cid]))[0]
    D, Lg = np.bincount(g, d), np.bincount(g, L)
    stat = 100 * D.sum() / Lg.sum()
    hits = 0
    for k in range(0, R_ITER, 1000):
        sg = rng.choice([-1.0, 1.0], size=(min(1000, R_ITER - k), len(D)))
        hits += np.sum(np.abs(100 * (sg @ D) / Lg.sum()) >= abs(stat) - 1e-12)
    idx = rng.randint(0, len(D), size=(N_BOOT, len(D)))
    boot = 100 * D[idx].sum(1) / Lg[idx].sum(1)
    return dict(d=stat, lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), p=(hits + 1) / (R_ITER + 1),
                n_obs=len(d), n_clusters=len(D))


def show(name, r):
    print(f"  {name:58s} d {r['d']:+.2f}  95% CI [{r['lo']:+.2f}, {r['hi']:+.2f}]  p = {r['p']:.4f}"
          f"  ({r['n_obs']} observations, {r['n_clusters']} clusters)")
    return r


OUT = {}
print("== 0. repeated items")
app = [c for s in S for c in RA.TEST[s]]
print(f"  test appearances over three splits {len(app)}, unique comments {len(set(app))}, "
      f"comments in more than one split {sum(v > 1 for v in pd.Series(app).value_counts())}, "
      f"near-duplicate groups {len(set(GROUP[c] for c in app))}")
print(f"  split 42: {len(RA.TEST[42])} comments in {len(set(GROUP[c] for c in RA.TEST[42]))} near-duplicate groups")

print("\n== A. three splits, one run each (unguarded); clusters = near-duplicate groups across splits")
E = {(v, s): errs(RA.SPANS[(v, s)], RA.TEST[s]) for v in RA.SYSTEMS for s in S}
for a, b in [("ctx_plain_aks_heads", "word_crf"), ("ctx_plain", "word_crf"), ("ctx_plain_aks_heads", "ufal_plain"),
             ("ctx_plain", "ufal_plain"), ("ctx_plain_aks_heads", "ctx_plain")]:
    OUT[f"{a} vs {b}"] = show(f"{a} vs {b}", clustered([(E[(a, s)], E[(b, s)]) for s in S]))


def stage(name, s):
    d = "stage1_seed42" if os.path.exists(os.path.join(G, "nb5_ctt/stage1_seed42", name.format(s))) else "stage2_three_splits"
    return f"nb5_ctt/{d}/" + name.format(s)


CP = {s: text_errs(stage("test_text_ctx_plain_aks_heads_s{}_copy.json", s), RA.TEST[s]) for s in S}
FR = {s: text_errs(stage("test_text_ctx_plain_aks_heads_s{}_free.json", s), RA.TEST[s]) for s in S}
OUT["copy vs free"] = show("copy gate vs free decoding (CTT)", clustered([(CP[s], FR[s]) for s in S]))
E8 = {s: errs(load(f"nb5_ctt/stage3a_8_epochs/test_spans_ctx_plain_aks_heads_e8_s{s}.json"), RA.TEST[s]) for s in S}
OUT["8 vs 5"] = show("8 epochs vs 5 epochs (CTT)", clustered([(E8[s], E[("ctx_plain_aks_heads", s)]) for s in S]))
EB = {s: errs(load(f"nb5_ctt/stage3b_byt5_base/test_spans_ctx_plain_aks_heads_byt5-base_s{s}.json"), RA.TEST[s]) for s in (13, 42)}
OUT["base vs small"] = show("ByT5-base vs ByT5-small (splits 13, 42)",
                            clustered([(EB[s], E[("ctx_plain_aks_heads", s)]) for s in (13, 42)]))

print("\n== B. split 42, three training runs pooled (unguarded); clusters = near-duplicate groups, all runs together")
ids42 = RA.TEST[42]
RUNS = {"CTT": ["nb5_ctt/stage2_three_splits/test_spans_ctx_plain_aks_heads_s42.json",
                "nb11_train_seeds/test_spans_ctt_ts1_s42.json", "nb11_train_seeds/test_spans_ctt_ts2_s42.json"],
        "ctx": ["nb5_ctt/stage1_seed42/test_spans_ctx_plain_s42.json",
                "nb11_train_seeds/test_spans_ctx_plain_ts1_s42.json", "nb11_train_seeds/test_spans_ctx_plain_ts2_s42.json"],
        "word": ["nb8_oracle_tags/test_spans_word_plain_align_s42.json",
                 "nb11_train_seeds/test_spans_word_plain_align_ts1_s42.json",
                 "nb11_train_seeds/test_spans_word_plain_align_ts2_s42.json"]}
RS = {k: [load(p) for p in v] for k, v in RUNS.items()}
RE = {k: [errs(sp, ids42) for sp in v] for k, v in RS.items()}
for a, b in [("CTT", "word"), ("ctx", "word"), ("CTT", "ctx")]:
    OUT[f"{a} vs {b} runs CER"] = show(f"{a} vs {b} CER", clustered(list(zip(RE[a], RE[b]))))
for a, b in [("CTT", "word"), ("ctx", "word")]:
    OUT[f"{a} vs {b} runs WER"] = show(f"{a} vs {b} WER", clustered(list(zip(RE[a], RE[b])), "we", "wn"))
    for x in RE[a] + RE[b]:
        x["one"] = 1.0
    OUT[f"{a} vs {b} runs EM"] = show(f"{a} vs {b} EM", clustered(list(zip(RE[a], RE[b])), "em", "one"))

print("\n== C. ablations on split 42 (single runs); clusters = near-duplicate groups")
st1 = "nb5_ctt/stage1_seed42/test_spans_{}_s42.json"
A = {v: load(st1.format(v)) for v in ["word_crf", "word_crf_align", "ctx_crf", "ctx_plain", "ctx_joint"]}
for a, b, lab in [("word_crf_align", "word_crf", "recover unaligned data"), ("ctx_crf", "word_crf_align", "add context"),
                  ("ctx_plain", "ctx_crf", "remove CRF tags"), ("ctx_plain", "ctx_joint", "remove joint LID")]:
    for gd in (False, True):
        OUT[f"abl {lab} {gd}"] = show(f"{lab} {'guarded' if gd else 'raw'}",
                                      clustered([(errs(A[a], ids42, gd), errs(A[b], ids42, gd))]))

print("\n== D. code-mixed and monolingual comments")
df["cm"] = raw["likely_code_mixed"].astype(bool)
cm = {s: [c for c in RA.TEST[s] if df.at[c, "cm"]] for s in S}
mono = {s: [c for c in RA.TEST[s] if not df.at[c, "cm"] and raw.at[c, "code_mix_type"] != "undetermined"] for s in S}
for nm, sub in (("code-mixed", cm), ("monolingual", mono)):
    for a, b in [("ctx_plain_aks_heads", "ufal_plain"), ("ctx_plain_aks_heads", "word_crf"), ("ctx_plain", "ufal_plain")]:
        OUT[f"{nm} {a} vs {b}"] = show(f"{nm}: {a} vs {b} (3 splits)",
                                       clustered([(E[(a, s)].loc[sub[s]], E[(b, s)].loc[sub[s]]) for s in S]))
    for a in ("CTT", "ctx"):
        OUT[f"{nm} {a} vs word runs"] = show(f"{nm}: {a} vs matched word (3 runs)",
                                             clustered([(x.loc[sub[42]], y.loc[sub[42]]) for x, y in zip(RE[a], RE["word"])]))

print("\n== E. language-switch tokens, Bengali script and Romanized Bengali counted as one language")
LANG = {"BAN": "BEN", "ROM_BAN": "BEN", "ENG": "ENG"}


def flags(labels, collapse):
    idx = [i for i, l in enumerate(labels) if l in LANG]
    lang = [LANG[labels[i]] if collapse else labels[i] for i in idx]
    return {i: any(lang[k] != lang[j] for j in (k - 1, k + 1) if 0 <= j < len(idx)) for k, i in enumerate(idx)}


SYS = ["word_crf", "ufal_plain", "ctx_plain", "ctx_plain_aks_heads"]
for collapse in (False, True):
    rows = []
    for s in S:
        for c in cm[s]:
            r_ = ref[c].split()
            if len(toks[c]) != len(r_):
                continue
            for i, f in flags(df.at[c, "labels"], collapse).items():
                row = dict(split=s, cid=c, switch=f, tag=df.at[c, "labels"][i])
                for v in SYS:
                    row[v] = float(RA.bn(RA.SPANS[(v, s)][c][i]) == r_[i])
                rows.append(row)
    T = pd.DataFrame(rows)
    acc = T.groupby("switch")[SYS].mean().mul(100)
    acc["tokens"] = T.groupby("switch").size()
    acc["gain CTT - word_crf"] = acc["ctx_plain_aks_heads"] - acc["word_crf"]
    print(f"  {'collapsed' if collapse else 'original (BAN and ROM_BAN differ)'}; three splits pooled")
    print(acc.round(2).to_string())
    if collapse:
        T["gain"] = T.ctx_plain_aks_heads - T.word_crf
        by = T.groupby(["tag", "switch"]).gain.agg(["mean", "size"])
        by["mean"] *= 100
        print("  gain by tag (points):"); print(by.round(2).to_string())
        w = T[T.switch].tag.value_counts(normalize=True)
        adj = sum(w[t] * 100 * T[(~T.switch) & (T.tag == t)].gain.mean() for t in w.index)
        print(f"  non-switch gain reweighted to the tag mix of switch tokens: {adj:.2f} points")
        OUT["switch"] = dict(tokens=acc["tokens"].to_dict(), gain=acc["gain CTT - word_crf"].to_dict(), adjusted_other=adj,
                             acc=acc[SYS].round(2).to_dict())
        rows = []
        for c in cm[42]:
            r_ = ref[c].split()
            if len(toks[c]) != len(r_):
                continue
            for i, f in flags(df.at[c, "labels"], True).items():
                row = dict(switch=f)
                for k in ("CTT", "ctx", "word"):
                    row[k] = np.mean([RA.bn(sp[c][i]) == r_[i] for sp in RS[k]])
                rows.append(row)
        T2 = pd.DataFrame(rows)
        a2 = T2.groupby("switch")[["word", "ctx", "CTT"]].mean().mul(100)
        a2["tokens"] = T2.groupby("switch").size(); a2["gain CTT - word"] = a2.CTT - a2.word
        print("  split 42, three training runs (mean accuracy over runs):"); print(a2.round(2).to_string())
        OUT["switch42"] = a2.round(2).to_dict()

print("\n== F. MultiLexNorm ERR, three seeds pooled; clusters = test sentences across seeds")
M = os.path.join(G, "nb9_multilexnorm")
for Lg in ("iden", "trde"):
    d = pd.read_csv(os.path.join(M, f"mln_{Lg}.csv"), keep_default_na=False); d = d[d.id.str.startswith(f"{Lg}_test_")]
    data = [(c, [t.rsplit("/", 1)[0] for t in tl.split(" ")], json.loads(sp)) for c, tl, sp in zip(d.id, d.token_labels, d.target_spans)]

    def per(sysn, seed):
        sp = json.load(open(os.path.join(M, f"test_spans_{Lg}_{sysn}_s{seed}.json"), encoding="utf-8"))
        cor = [sum(unicodedata.normalize("NFC", a) == unicodedata.normalize("NFC", b) for a, b in zip(g, sp[c])) for c, _, g in data]
        chg = [sum(r != g_ for r, g_ in zip(t, g)) for _, t, g in data]
        return pd.DataFrame(dict(cor=cor, chg=chg), index=[c for c, _, _ in data])
    for a, b in [("ctx_copy", "ufal"), ("ctx", "word"), ("ctx_copy", "ctx")]:
        pairs = [(per(a, s), per(b, s)) for s in (13, 42, 2026)]
        OUT[f"mln {Lg} {a} vs {b}"] = show(f"{Lg} {a} vs {b} (dERR)", clustered(pairs, "cor", "chg"))

print("\n== G. LID macro-F1, XLM-R minus CRF, mean over three splits; clusters = near-duplicate groups across splits")
LAB = ["BAN", "ENG", "ROM_BAN", "OTHER", "UNK"]; LI = {l: i for i, l in enumerate(LAB)}
cms, gids = [], []
for s in S:
    sp = pd.read_csv(os.path.join(ROOT, "data", "splits", f"split_seed{s}.csv")); te = sp[sp.split == "test"].id.tolist()
    X = load(f"nb1_lid_transformers/lid_test_pred_xlm-roberta-base_s{s}_ce.json")
    C = pickle.load(open(os.path.join(ROOT, "results", "cpu", f"crf_pred_seed{s}.pkl"), "rb"))
    for c in te:
        m = np.zeros((2, 5, 5))
        for k, P in enumerate((X[c], C[c])):
            for gl, pl in zip(df.at[c, "labels"], P):
                m[k, LI[gl], LI[pl]] += 1
        cms.append((s, m)); gids.append(GROUP[c])
split_of = np.array([s for s, _ in cms]); Mx = np.stack([m for _, m in cms])
gidx = pd.factorize(pd.Series(gids))[0]; nG = gidx.max() + 1


def macro(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1); fp = cm.sum(-2) - tp; fn = cm.sum(-1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0).mean(-1)


def lid_stat(swap, w):
    out = 0.0
    for s in S:
        k = split_of == s
        sw, ww = swap[gidx[k]], w[gidx[k]]
        a = (ww[:, None, None] * np.where(sw[:, None, None], Mx[k, 1], Mx[k, 0])).sum(0)
        b = (ww[:, None, None] * np.where(sw[:, None, None], Mx[k, 0], Mx[k, 1])).sum(0)
        out += 100 * (macro(a) - macro(b)) / len(S)
    return out


rng = np.random.RandomState(0)
obs = lid_stat(np.zeros(nG, bool), np.ones(nG))
null = np.array([lid_stat(rng.rand(nG) < 0.5, np.ones(nG)) for _ in range(2000)])
boot = np.array([lid_stat(np.zeros(nG, bool), np.bincount(rng.randint(0, nG, nG), minlength=nG).astype(float)) for _ in range(2000)])
OUT["lid"] = dict(d=obs, lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), p=(np.sum(np.abs(null) >= abs(obs)) + 1) / 2001)
print(f"  XLM-R - CRF macro-F1 {obs:+.2f}  95% CI [{OUT['lid']['lo']:+.2f}, {OUT['lid']['hi']:+.2f}]  p = {OUT['lid']['p']:.4f}"
      f"  (2,000 permutations and resamples, {nG} clusters)")

json.dump({k: {kk: (float(vv) if isinstance(vv, (np.floating, float, int, np.integer)) else vv) for kk, vv in v.items()}
           for k, v in OUT.items()}, open(os.path.join(ROOT, "analysis", "clustered_tests.json"), "w"), indent=1, default=str)
