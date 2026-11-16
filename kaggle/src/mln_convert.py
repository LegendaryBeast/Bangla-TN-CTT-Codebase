import argparse, json, os, random

LANGS = {"iden": "ind_eng", "trde": "tur_deu"}


def load(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        t = line.rstrip("\n").split("\t")
        if t == [""]:
            if cur:
                sents.append(cur)
            cur = []
            continue
        assert len(t) <= 2, (path, line)
        cur.append((t[0], t[1] if len(t) == 2 else ""))
    if cur:
        sents.append(cur)
    return sents


def convert(repo, out, code):
    d = os.path.join(repo, "data", LANGS[code])
    parts = {"train": load(os.path.join(d, "train.norm")), "test": load(os.path.join(d, "test.norm"))}
    if os.path.exists(os.path.join(d, "dev.norm")):
        parts["val"] = load(os.path.join(d, "dev.norm"))
    else:
        idx = list(range(len(parts["train"])))
        random.Random(0).shuffle(idx)
        held = set(idx[: len(idx) // 10])
        parts["val"] = [s for i, s in enumerate(parts["train"]) if i in held]
        parts["train"] = [s for i, s in enumerate(parts["train"]) if i not in held]
    rows, split = [], []
    for name, sents in parts.items():
        for i, s in enumerate(sents):
            assert all(raw and " " not in raw for raw, _ in s), "raw tokens must be non-empty and space-free"
            cid = f"{code}_{name}_{i}"
            spans = [norm for _, norm in s]
            rows.append({"id": cid, "token_labels": " ".join(f"{raw}/UNK" for raw, _ in s),
                         "bengali_transliteration": " ".join(x for x in spans if x), "target_spans": json.dumps(spans, ensure_ascii=False)})
            split.append({"id": cid, "split": name})
    import pandas as pd
    od = os.path.join(out, code)
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(od, "mln.csv"), index=False)
    for seed in (13, 42, 2026):
        pd.DataFrame(split).to_csv(os.path.join(od, f"split_seed{seed}.csv"), index=False)
    n = {k: (len(v), sum(len(s) for s in v)) for k, v in parts.items()}
    print(code, "sentences/tokens:", n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    a = ap.parse_args()
    for code in a.langs:
        convert(a.repo, a.out, code)
