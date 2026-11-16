# %%
!pip -q install rapidfuzz 2>/dev/null

# %% writefile banglish_align.py

# %% writefile ctt.py

# %%
import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd, torch

OUT = "/kaggle/working/ctt"
SCRIPT = "/kaggle/working/ctt.py"
EPOCHS = 5
MODEL = "google/byt5-small"
BASE_ARGS = ["--batch_words", "48", "--accum", "2", "--max_chunks", "8", "--gen_bs", "48"]
POLL_SECONDS = 120

VARIANTS = {
    "ufal_plain":      dict(cfg=dict(mode="ufal", align=1, save_val=1), extra=["--max_enc_tokens", "4096"]),
    "ctx_plain_rerun": dict(cfg=dict(mode="focus", align=1, save_val=1), extra=[]),
}
RUNS = {"ufal_plain": [42, 13]}

JOBS = [dict(variant=v, seed=s, **VARIANTS[v]) for v, ss in RUNS.items() for s in ss]
os.makedirs(OUT, exist_ok=True)
JOBS = [j for j in JOBS if not os.path.exists(f"{OUT}/result_{j['variant']}_s{j['seed']}.json")]
print(len(JOBS), "jobs:", [f"{j['variant']}/s{j['seed']}" for j in JOBS])

# %%
n_gpu = max(1, torch.cuda.device_count())
queue, running = list(JOBS), {}

def launch(gpu, j):
    logf = open(f"{OUT}/log_{j['variant']}_s{j['seed']}.txt", "w")
    cmd = [sys.executable, SCRIPT, "--variant", j["variant"], "--seed", str(j["seed"]), "--epochs", str(EPOCHS),
           "--model", MODEL, "--out", OUT]
    for k, v in j["cfg"].items():
        cmd += [f"--{k}", str(v)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return subprocess.Popen(cmd + BASE_ARGS + j["extra"], stdout=logf, stderr=subprocess.STDOUT, env=env), j, logf

def tail(j):
    try:
        lines = [l for l in open(f"{OUT}/log_{j['variant']}_s{j['seed']}.txt") if l.startswith("[")]
        return lines[-1].strip() if lines else ""
    except FileNotFoundError:
        return ""

t0 = time.time()
while queue or running:
    for g in range(n_gpu):
        if g not in running and queue:
            running[g] = launch(g, queue.pop(0))
    time.sleep(POLL_SECONDS)
    for g, (proc, j, logf) in list(running.items()):
        if proc.poll() is not None:
            logf.close()
            print(f"[{(time.time() - t0) / 60:6.1f} min] finished {j['variant']}/s{j['seed']} exit={proc.returncode}")
            if proc.returncode != 0:
                text = open(logf.name).read()
                print("\n".join(l for l in text.splitlines() if l.startswith("[")))
                print(text[-1500:])
            del running[g]
        else:
            print(f"[{(time.time() - t0) / 60:6.1f} min] gpu{g} {tail(j)}")

# %%
rows, eff, val = [], [], []
for f in sorted(glob.glob(f"{OUT}/result_*.json")):
    r = json.load(open(f))
    v, s = r["args"]["variant"], r["args"]["seed"]
    for mode, m in r["norm"].items():
        rows.append(dict(variant=v, mode=mode, seed=s, CER=m["CER"], WER=m["WER"], EM=m["Exact"],
                         ENG=m["TokAcc_ENG"], ROM=m["TokAcc_ROM_BAN"], UNK=m["TokAcc_UNK"], OOV=m["TokAcc_OOV"]))
    e = r["efficiency"]["free"]
    eff.append(dict(variant=v, seed=s, gen_words_per_sec=e["words_per_sec"], end_to_end_words_per_sec=e.get("total_words_per_sec"),
                    enc_seconds=e.get("enc_seconds"), gen_seconds=e["gen_seconds"], encoder_bytes=e.get("encoder_bytes")))
    if "val" in r:
        val.append(dict(variant=v, seed=s, val_CER_raw=r["val"]["free"]["CER"], val_CER_guard=r["val"]["free+guard"]["CER"],
                        guard_delta=r["val"]["free+guard"]["CER"] - r["val"]["free"]["CER"],
                        words_changed=r["val"]["guard_changed_words"]))
R, E, V = pd.DataFrame(rows), pd.DataFrame(eff), pd.DataFrame(val)
R.to_csv(f"{OUT}/nb6_norm.csv", index=False); E.to_csv(f"{OUT}/nb6_efficiency.csv", index=False); V.to_csv(f"{OUT}/nb6_val_guard.csv", index=False)
if len(R):
    display(R.round(2))
if len(E):
    display(E.round(1))
if len(V):
    display(V.round(3))
print("Reference (paper, guarded, 3-split mean): ctx_plain CER 3.65 | CTT 3.56 | generation-only words/s: ctx_plain 136.9")
