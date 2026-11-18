import argparse, json, os, sys, time, unicodedata
import numpy as np, pandas as pd, torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--ctt_weights", required=True)
ap.add_argument("--ufal_weights", required=True)
ap.add_argument("--device", default="mps")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--budgets", type=int, nargs="+", default=[2048, 4096])
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--dump", default="")
args = ap.parse_args()
sys.path.insert(0, os.path.join(args.root, "kaggle", "src"))
from banglish_align import join_spans
from rapidfuzz.distance import Levenshtein

EOS, OFF, MARK_L, MARK_R, NO_TAG = 1, 3, 259, 260, 5
SRC_BUDGET, MAX_WORD, MAX_SRC, WORDS, CHUNKS, GEN_BS, MAX_TGT, TAU = 160, 64, 320, 48, 8, 48, 160, 0.9
NUKTA = {"ড়": "ড়", "ঢ়": "ঢ়", "য়": "য়"}
DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
df = pd.read_csv(os.path.join(args.root, "data", "BanglaMix.csv"), keep_default_na=False).set_index("id")
sp = pd.read_csv(os.path.join(args.root, "data", "splits", "split_seed42.csv"))
test = sp[sp.split == "test"].id.tolist()
test = test[:args.limit] if args.limit else test
TOKS = {c: [t.rsplit("/", 1)[0] for t in df.at[c, "token_labels"].split(" ")] for c in test}
dev = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")


def bn(s):
    s = unicodedata.normalize("NFC", s)
    for k, v in NUKTA.items():
        s = s.replace(k, v)
    return s.replace("‌", "").replace("‍", "")


REF = {c: " ".join(bn(df.at[c, "bengali_transliteration"]).split()) for c in test}


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    cfg = T5Config.from_pretrained("google/byt5-small", local_files_only=True)
    m = T5ForConditionalGeneration(cfg)
    m.load_state_dict({k[3:]: v.float() for k, v in sd.items() if k.startswith("t5.")})
    extra = {k: v.float().to(dev) for k, v in sd.items() if not k.startswith("t5.") and v.is_floating_point()}
    return m.to(dev).eval(), extra


MODELS = {"focus": load(args.ctt_weights), "ufal": load(args.ufal_weights)}


def chunk_words(words):
    chunks, cur, size = [], [], 0
    for i, w in enumerate(words):
        b = len(w.encode()) + (1 if cur else 0)
        if cur and size + b > SRC_BUDGET:
            chunks.append(cur); cur, size, b = [], 0, len(w.encode())
        cur.append(i); size += b
    if cur:
        chunks.append(cur)
    return chunks


def prepare(mode):
    ex = []
    for c in test:
        words = TOKS[c]
        for idx in chunk_words(words):
            ws = [words[i] for i in idx]
            if mode == "ufal":
                for p in range(len(ws)):
                    ids, span = [], (0, 0)
                    for j, w in enumerate(ws):
                        if j:
                            ids.append(ord(" ") + OFF)
                        b = [x + OFF for x in w.encode()[:MAX_WORD]]
                        if j == p:
                            ids.append(MARK_L); span = (len(ids), len(ids) + len(b)); ids += b; ids.append(MARK_R)
                        else:
                            ids += b
                    ids = ids[:MAX_SRC - 1] + [EOS]
                    ex.append(dict(cid=c, widx=[idx[p]], words=[ws[p]], ids=ids,
                                   spans=[(span[0], min(span[1], len(ids) - 1))], nodec=[len(ws[p].encode()) > MAX_WORD]))
                continue
            ids, spans = [], []
            for j, w in enumerate(ws):
                if j:
                    ids.append(ord(" ") + OFF)
                b = [x + OFF for x in w.encode()[:MAX_WORD]]
                spans.append((len(ids), len(ids) + len(b))); ids += b
            ids = ids[:MAX_SRC - 1] + [EOS]
            ex.append(dict(cid=c, widx=idx, words=ws, ids=ids, spans=[(s, min(e, len(ids) - 1)) for s, e in spans],
                           nodec=[len(w.encode()) > MAX_WORD for w in ws]))
    return ex


def batches(ex, budget):
    cap = 2 * (WORDS if all(len(e["words"]) == 1 for e in ex[:200]) else CHUNKS)
    out = []
    for m in range(0, len(ex), 2000):
        mega = sorted(range(m, min(m + 2000, len(ex))), key=lambda i: len(ex[i]["ids"]))
        cur, words, T = [], 0, 0
        for i in mega:
            k, T2 = len(ex[i]["words"]), max(T, len(ex[i]["ids"]))
            if cur and (words + k > 2 * WORDS or len(cur) >= cap or (len(cur) + 1) * T2 > 2 * budget):
                out.append(cur); cur, words, T2 = [], 0, len(ex[i]["ids"])
            cur.append(i); words += k; T = T2
        if cur:
            out.append(cur)
    return out


def decode_bytes(seq):
    out = []
    for x in seq.tolist():
        if x == EOS:
            break
        if OFF <= x < OFF + 256:
            out.append(x - OFF)
    return bytes(out).decode("utf-8", errors="ignore").strip(" ")


@torch.no_grad()
def run(mode, budget, gated):
    model, extra = MODELS[mode]
    enc = model.get_encoder()
    t = dict.fromkeys(["prepare", "batch", "collate_transfer", "encoder", "gate", "decoder", "to_text"], 0.0)
    t0 = time.perf_counter(); ex = prepare(mode); t["prepare"] = time.perf_counter() - t0
    t0 = time.perf_counter(); plan = batches(ex, budget); t["batch"] = time.perf_counter() - t0
    res, n_dec = {}, 0
    for idx in plan:
        B = [ex[i] for i in idx]
        t0 = time.perf_counter()
        T = max(len(e["ids"]) for e in B); Wn = max(len(e["spans"]) for e in B)
        ids = torch.zeros(len(B), T, dtype=torch.long); att = torch.zeros(len(B), T, dtype=torch.long)
        assign = torch.zeros(len(B), Wn, T)
        for i, e in enumerate(B):
            ids[i, :len(e["ids"])] = torch.tensor(e["ids"]); att[i, :len(e["ids"])] = 1
            for w, (s, e_) in enumerate(e["spans"]):
                if e_ > s:
                    assign[i, w, s:e_] = 1.0
        ids, att, assign = ids.to(dev), att.to(dev), assign.to(dev)
        sync(); t["collate_transfer"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        h0 = enc(input_ids=ids, attention_mask=att).last_hidden_state
        h = h0 + assign.sum(1).unsqueeze(-1) * extra["ext.weight"][NO_TAG]
        sync(); t["encoder"] += time.perf_counter() - t0
        allw = [(i, w) for i, e in enumerate(B) for w in range(len(e["spans"]))]
        t0 = time.perf_counter()
        if gated:
            u = torch.bmm(assign / assign.sum(-1, keepdim=True).clamp_min(1.0), h0)
            cp = torch.softmax(u @ extra["copy.weight"].T + extra["copy.bias"], -1).cpu()
        sel = []
        for n, (i, w) in enumerate(allw):
            e = B[i]; s, e_ = e["spans"][w]
            c = bn(e["words"][w]); d = c.translate(DIGITS)
            if e_ <= s or e["nodec"][w]:
                res[(e["cid"], e["widx"][w])] = c; continue
            if gated and float(cp[i, w, 1:].sum()) >= TAU:
                res[(e["cid"], e["widx"][w])] = c if cp[i, w, 1] >= cp[i, w, 2] else d; continue
            sel.append(n)
        sync(); t["gate"] += time.perf_counter() - t0
        n_dec += len(sel)
        for g0 in range(0, len(sel), GEN_BS):
            part = sel[g0:g0 + GEN_BS]
            maxb = max(len(B[allw[n][0]]["words"][allw[n][1]].encode()) for n in part)
            t0 = time.perf_counter()
            bi = torch.tensor([allw[n][0] for n in part], device=dev)
            if mode == "ufal":
                hf, a = h[bi], att[bi]
            else:
                m = assign[bi, torch.tensor([allw[n][1] for n in part], device=dev)].unsqueeze(-1)
                f = extra["focus.weight"]
                hf, a = h[bi] + m * f[1] + (1 - m) * f[0], att[bi]
            g = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=hf), attention_mask=a,
                               max_new_tokens=min(MAX_TGT, 4 * min(maxb, MAX_WORD) + 24), num_beams=1, do_sample=False)
            sync(); t["decoder"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for k, n in enumerate(part):
                i, w = allw[n]
                res[(B[i]["cid"], B[i]["widx"][w])] = decode_bytes(g[k])
            t["to_text"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    text = {c: join_spans([res[(c, i)] for i in range(len(TOKS[c]))]) for c in test}
    t["to_text"] += time.perf_counter() - t0
    t["total"] = sum(t.values())
    if args.dump:
        json.dump({f"{c}|{i}": res[(c, i)] for c in test for i in range(len(TOKS[c]))}, open(f"{args.dump}_{mode}_{int(gated)}.json", "w"), ensure_ascii=False)
    cer = 100 * sum(Levenshtein.distance(text[c], REF[c]) for c in test) / sum(len(REF[c]) for c in test)
    return t, len(allw) and sum(len(TOKS[c]) for c in test), n_dec, len(plan), cer


SYSTEMS = [("CTT, free decoding", "focus", False), ("CTT, copy gate", "focus", True), ("Re-encoding", "ufal", False)]
run("focus", 2048, False)
rows = []
for budget in args.budgets:
    for name, mode, gated in SYSTEMS:
        for r in range(args.repeats):
            t, nw, nd, nb, cer = run(mode, budget, gated)
            rows.append(dict(system=name, budget=budget, repeat=r, words=nw, decoded=nd, batches=nb, CER=cer, **t))
            print(json.dumps(rows[-1]), flush=True)
R = pd.DataFrame(rows)
s = R.groupby(["budget", "system"], sort=False).agg(total=("total", "mean"), tmin=("total", "min"), tmax=("total", "max"),
                                                     words=("words", "first"), decoded=("decoded", "first"), CER=("CER", "first"))
s["words_per_s"] = s.words / s.total
print(s.round(2).to_string())
json.dump(dict(device=str(dev), torch=torch.__version__, weights="trained checkpoints (fp16 copies, run in fp32)", runs=rows),
          open(args.out, "w"), indent=1)
