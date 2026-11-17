import argparse, json, os, sys, time, unicodedata
import numpy as np, pandas as pd, torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--device", default="mps")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--budgets", type=int, nargs="+", default=[2048, 4096])
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

EOS, OFF, MARK_L, MARK_R = 1, 3, 259, 260
SRC_BUDGET, MAX_WORD, MAX_SRC, WORDS, CHUNKS, GEN_BS = 160, 64, 320, 48, 8, 48
G = os.path.join(args.root, "results", "gpu")
DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
df = pd.read_csv(os.path.join(args.root, "data", "BanglaMix.csv"), keep_default_na=False).set_index("id")
sp = pd.read_csv(os.path.join(args.root, "data", "splits", "split_seed42.csv"))
test = sp[sp.split == "test"].id.tolist()
test = test[:args.limit] if args.limit else test
TOKS = {c: [t.rsplit("/", 1)[0] for t in df.at[c, "token_labels"].split(" ")] for c in test}
SAVED = {"focus": "nb5_ctt/stage2_three_splits/test_spans_ctx_plain_aks_heads_s42.json",
         "ufal": "nb6_ufal/run1_ufal_s42_s13/test_spans_ufal_plain_s42.json",
         "word": "nb8_oracle_tags/test_spans_word_plain_align_s42.json"}
OUTS = {k: json.load(open(os.path.join(G, v), encoding="utf-8")) for k, v in SAVED.items()}
dev = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()


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
        groups = [[i] for i in range(len(words))] if mode == "word" else chunk_words(words)
        for idx in groups:
            ws = [words[i] for i in idx]
            targets = [len(OUTS[mode][c][i].encode()) + 1 for i in idx]
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
                    ex.append(dict(cid=c, widx=[idx[p]], words=[ws[p]], ids=ids, spans=[(span[0], min(span[1], len(ids) - 1))],
                                   tl=[targets[p]]))
                continue
            ids, spans = [], []
            for j, w in enumerate(ws):
                if j:
                    ids.append(ord(" ") + OFF)
                b = [x + OFF for x in w.encode()[:MAX_WORD]]
                spans.append((len(ids), len(ids) + len(b))); ids += b
            ids = ids[:MAX_SRC - 1] + [EOS]
            ex.append(dict(cid=c, widx=idx, words=ws, ids=ids, spans=[(s, min(e, len(ids) - 1)) for s, e in spans], tl=targets))
    return ex


def batches(ex, budget):
    cap = 2 * (WORDS if all(len(e["words"]) == 1 for e in ex[:200]) else CHUNKS)
    max_w, max_enc = 2 * WORDS, 2 * budget
    out = []
    for m in range(0, len(ex), 2000):
        mega = sorted(range(m, min(m + 2000, len(ex))), key=lambda i: len(ex[i]["ids"]))
        cur, words, T = [], 0, 0
        for i in mega:
            k, T2 = len(ex[i]["words"]), max(T, len(ex[i]["ids"]))
            if cur and (words + k > max_w or len(cur) >= cap or (len(cur) + 1) * T2 > max_enc):
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


cfg = T5Config.from_pretrained("google/byt5-small", local_files_only=True)
torch.manual_seed(0)
model = T5ForConditionalGeneration(cfg).to(dev).eval()
enc = model.get_encoder()
H = cfg.d_model
focus = torch.randn(2, H, device=dev) * 0.1
gate = torch.nn.Linear(H, 3).to(dev)


@torch.no_grad()
def run(mode, budget, gated):
    t = {"prepare": 0.0, "batch": 0.0, "collate_transfer": 0.0, "encoder": 0.0, "gate": 0.0, "decoder": 0.0, "to_text": 0.0}
    t0 = time.perf_counter(); ex = prepare(mode); t["prepare"] = time.perf_counter() - t0
    t0 = time.perf_counter(); plan = batches(ex, budget); t["batch"] = time.perf_counter() - t0
    res, n_dec, n_words = {}, 0, 0
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
        t0 = time.perf_counter(); h = enc(input_ids=ids, attention_mask=att).last_hidden_state; sync()
        t["encoder"] += time.perf_counter() - t0
        allw = [(i, w) for i, e in enumerate(B) for w in range(len(e["spans"]))]
        t0 = time.perf_counter()
        sel = []
        if gated:
            u = torch.bmm(assign / assign.sum(-1, keepdim=True).clamp_min(1.0), h)
            cp = torch.softmax(gate(u).float(), -1).cpu()
        for n, (i, w) in enumerate(allw):
            e = B[i]; word = e["words"][w]; src = unicodedata.normalize("NFC", word)
            saved = OUTS[mode][e["cid"]][e["widx"][w]]
            if gated and saved in (src, src.translate(DIGITS)):
                _ = float(cp[i, w, 1:].sum())
                res[(e["cid"], e["widx"][w])] = src
                continue
            sel.append(n)
        sync(); t["gate"] += time.perf_counter() - t0
        n_words += len(allw); n_dec += len(sel)
        for g0 in range(0, len(sel), GEN_BS):
            part = sel[g0:g0 + GEN_BS]
            L = max(B[allw[n][0]]["tl"][allw[n][1]] for n in part)
            t0 = time.perf_counter()
            bi = torch.tensor([allw[n][0] for n in part], device=dev)
            if mode == "ufal":
                hf, a = h[bi], att[bi]
            else:
                m = assign[bi, torch.tensor([allw[n][1] for n in part], device=dev)].unsqueeze(-1)
                hf, a = h[bi] + m * focus[1] + (1 - m) * focus[0], att[bi]
            g = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=hf), attention_mask=a,
                               max_new_tokens=L, min_new_tokens=L, num_beams=1, do_sample=False)
            sync(); t["decoder"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for k, n in enumerate(part):
                i, w = allw[n]
                res[(B[i]["cid"], B[i]["widx"][w])] = decode_bytes(g[k])
            t["to_text"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    text = {c: " ".join(x for x in (res[(c, i)] for i in range(len(TOKS[c]))) if x) for c in test}
    t["to_text"] += time.perf_counter() - t0
    t["total"] = sum(t.values())
    return t, n_words, n_dec, len(plan)


SYSTEMS = [("CTT, free decoding", "focus", False), ("CTT, copy gate", "focus", True),
           ("Re-encoding", "ufal", False), ("Word level", "word", False)]
run("focus", 2048, False)
rows = []
for budget in args.budgets:
    for name, mode, gated in SYSTEMS:
        for r in range(args.repeats):
            t, nw, nd, nb = run(mode, budget, gated)
            rows.append(dict(system=name, budget=budget, repeat=r, words=nw, decoded=nd, batches=nb, **t))
            print(json.dumps(rows[-1]), flush=True)
R = pd.DataFrame(rows)
summary = R.groupby(["budget", "system"], sort=False).agg(total_mean=("total", "mean"), total_min=("total", "min"),
                                                           total_max=("total", "max"), encoder=("encoder", "mean"),
                                                           decoder=("decoder", "mean"), words=("words", "first"),
                                                           decoded=("decoded", "first"), batches=("batches", "first"))
summary["words_per_s"] = summary.words / summary.total_mean
print(summary.round(2).to_string())
json.dump(dict(device=str(dev), torch=torch.__version__, weights="ByT5-small architecture, random initialization (seed 0)",
               decode_lengths="per-word output lengths of the saved split-42 test predictions of each system",
               runs=rows), open(args.out, "w"), indent=1)
