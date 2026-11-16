import argparse, json, os, time
import numpy as np, pandas as pd, torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--device", default="mps")
ap.add_argument("--check_chunks", type=int, default=150)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", required=True)
args = ap.parse_args()

EOS, OFFSET, BUDGET, MAX_WORD, MAX_SRC = 1, 3, 160, 64, 320
torch.manual_seed(0)
rng = np.random.RandomState(0)


def chunk_words(words):
    chunks, cur, size = [], [], 0
    for i, w in enumerate(words):
        b = len(w.encode("utf-8")) + (1 if cur else 0)
        if cur and size + b > BUDGET:
            chunks.append(cur)
            cur, size, b = [], 0, len(w.encode("utf-8"))
        cur.append(i)
        size += b
    if cur:
        chunks.append(cur)
    return chunks


def build(words):
    ids, spans = [], []
    for j, w in enumerate(words):
        if j:
            ids.append(ord(" ") + OFFSET)
        b = [x + OFFSET for x in w.encode("utf-8")[:MAX_WORD]]
        spans.append((len(ids), len(ids) + len(b)))
        ids += b
    ids = ids[:MAX_SRC - 1] + [EOS]
    return ids, [(s, min(e, len(ids) - 1)) for s, e in spans]


df = pd.read_csv(os.path.join(args.root, "data", "BanglaMix.csv"), keep_default_na=False).set_index("id")
sp = pd.read_csv(os.path.join(args.root, "data", "splits", "split_seed42.csv"))
test = sp[sp.split == "test"].id.tolist()
chunks = []
for c in test:
    toks = [t.rsplit("/", 1)[0] for t in df.at[c, "token_labels"].split(" ")]
    for idx in chunk_words(toks):
        ids, spans = build([toks[i] for i in idx])
        chunks.append((ids, spans))
n_words = sum(len(s) for _, s in chunks)
bytes_shared = sum(len(i) for i, _ in chunks)
bytes_reenc = sum(len(i) * len(s) for i, s in chunks)
bytes_markers = sum((len(i) + 2) * len(s) for i, s in chunks)

dev = torch.device(args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
cfg = T5Config(d_ff=3584, d_kv=64, d_model=1472, decoder_start_token_id=0, dropout_rate=0.1, eos_token_id=1,
               feed_forward_proj="gated-gelu", layer_norm_epsilon=1e-6, num_decoder_layers=4, num_heads=6,
               num_layers=12, pad_token_id=0, relative_attention_num_buckets=32, tie_word_embeddings=False,
               vocab_size=384)
torch.manual_seed(0)
model = T5ForConditionalGeneration(cfg).to(dev).eval()
enc = model.get_encoder()
H = model.config.d_model


def encode(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    att = torch.zeros(len(seqs), L, dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, :len(s)] = torch.tensor(s)
        att[k, :len(s)] = 1
    ids, att = ids.to(dev), att.to(dev)
    return enc(input_ids=ids, attention_mask=att).last_hidden_state, att


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()


with torch.no_grad():
    probe, _ = encode([chunks[0][0]])
    f_ctx = torch.zeros(H, device=dev)
    f_word = torch.randn(H, device=dev) * float(probe.float().std())

    multi = [k for k, (_, s) in enumerate(chunks) if len(s) >= 3]
    sample = rng.choice(multi, min(args.check_chunks, len(multi)), replace=False)
    max_state, max_logit, words_checked, same_tokens = 0.0, 0.0, 0, 0
    for k in sample:
        ids, spans = chunks[k]
        h_shared, att = encode([ids])
        for (s, e) in spans:
            h_again, _ = encode([ids])
            max_state = max(max_state, float((h_again - h_shared).abs().max()))
            m = torch.zeros(1, len(ids), 1, device=dev)
            m[0, s:e] = 1
            mem_a = h_shared + m * f_word + (1 - m) * f_ctx
            mem_b = h_again + m * f_word + (1 - m) * f_ctx
            ga = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem_a), attention_mask=att,
                                max_new_tokens=24, num_beams=1, do_sample=False, output_scores=True,
                                return_dict_in_generate=True)
            gb = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem_b), attention_mask=att,
                                max_new_tokens=24, num_beams=1, do_sample=False, output_scores=True,
                                return_dict_in_generate=True)
            same_tokens += int(torch.equal(ga.sequences, gb.sequences))
            for a, b in zip(ga.scores, gb.scores):
                max_logit = max(max_logit, float((a - b).abs().max()))
            words_checked += 1

    order = sorted(range(len(chunks)), key=lambda k: len(chunks[k][0]))
    shared_seqs = [chunks[k][0] for k in order]
    reenc_seqs = [chunks[k][0] for k in order for _ in chunks[k][1]]

    def time_pass(seqs, max_tokens=4096):
        sync(); t0 = time.time()
        i = 0
        while i < len(seqs):
            j = i
            while j < len(seqs) and (j - i + 1) * len(seqs[j]) <= max_tokens:
                j += 1
            j = max(j, i + 1)
            encode(seqs[i:j])
            i = j
        sync()
        return time.time() - t0

    time_pass(shared_seqs[:64])
    t_shared, t_reenc = [], []
    for _ in range(args.repeats):
        t_shared.append(time_pass(shared_seqs))
        t_reenc.append(time_pass(reenc_seqs))

out = {
    "device": str(dev), "torch": torch.__version__, "weights": "ByT5-small architecture, random initialization (seed 0)",
    "config": {k: getattr(cfg, k) for k in ("d_model", "d_ff", "num_layers", "num_decoder_layers", "num_heads", "vocab_size")}, "test_comments": len(test), "chunks": len(chunks), "words": n_words,
    "encoder_bytes": {"shared": bytes_shared, "reencode_output_marking": bytes_reenc, "reencode_input_markers": bytes_markers},
    "exactness": {"chunks": int(len(sample)), "words": words_checked, "max_abs_state_diff": max_state,
                  "max_abs_logit_diff": max_logit, "identical_greedy_outputs": same_tokens},
    "encoder_seconds": {"shared": t_shared, "reencode": t_reenc,
                        "ratio_mean": float(np.mean(t_reenc) / np.mean(t_shared))},
}
json.dump(out, open(args.out, "w"), indent=1)
print(json.dumps(out, indent=1))
