# %%
!pip -q install rapidfuzz 2>/dev/null

# %% writefile banglish_align.py

# %% writefile ctt.py

# %%
import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd, torch

OUT = "/kaggle/working/ctt"
SCRIPT = "/kaggle/working/ctt.py"
EPOCHS, SPLIT = 5, 42
MODEL = "google/byt5-small"
BASE_ARGS = ["--batch_words", "48", "--accum", "2", "--max_chunks", "8", "--gen_bs", "48"]
POLL_SECONDS = 120
SYSTEMS = {
    "ctt":              dict(mode="focus", align=1, pretrain_aksharantar=200000, lam=0.4, copy_mu=0.5, detach_heads=1),
    "ctx_plain":        dict(mode="focus", align=1),
    "word_plain_align": dict(mode="word", align=1),
}
TRAIN_SEEDS = [1, 2]
GPU_QUEUES = [[(f"{s}_ts{t}", t, cfg) for s, cfg in SYSTEMS.items()] for t in TRAIN_SEEDS]

os.makedirs(OUT, exist_ok=True)
done = lambda v: os.path.exists(f"{OUT}/result_{v}_s{SPLIT}.json")
n_gpu = max(1, torch.cuda.device_count())
if n_gpu < len(GPU_QUEUES):
    GPU_QUEUES = [[j for q in GPU_QUEUES for j in q]]
GPU_QUEUES = [[j for j in q if not done(j[0])] for q in GPU_QUEUES]
print(n_gpu, "GPU(s); queues:", [[j[0] for j in q] for q in GPU_QUEUES])

# %%
def launch(gpu, job):
    v, ts, cfg = job
    logf = open(f"{OUT}/log_{v}_s{SPLIT}.txt", "w")
    cmd = [sys.executable, SCRIPT, "--variant", v, "--seed", str(SPLIT), "--train_seed", str(ts), "--epochs", str(EPOCHS),
           "--model", MODEL, "--out", OUT]
    for k, x in cfg.items():
        cmd += [f"--{k}", str(x)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return subprocess.Popen(cmd + BASE_ARGS, stdout=logf, stderr=subprocess.STDOUT, env=env), v, logf

def tail(v):
    try:
        lines = [l for l in open(f"{OUT}/log_{v}_s{SPLIT}.txt") if l.startswith("[")]
        return lines[-1].strip() if lines else ""
    except FileNotFoundError:
        return ""

t0, running = time.time(), {}
while any(GPU_QUEUES) or running:
    for g, q in enumerate(GPU_QUEUES):
        if g not in running and q:
            running[g] = launch(g, q.pop(0))
    time.sleep(POLL_SECONDS)
    for g, (proc, v, logf) in list(running.items()):
        if proc.poll() is not None:
            logf.close()
            print(f"[{(time.time() - t0) / 60:6.1f} min] finished {v} exit={proc.returncode}")
            if proc.returncode != 0:
                text = open(logf.name).read()
                print("\n".join(l for l in text.splitlines() if l.startswith("[")))
                print(text[-1500:])
            del running[g]
        else:
            print(f"[{(time.time() - t0) / 60:6.1f} min] gpu{g} {tail(v)}")

# %%
rows = []
for f in sorted(glob.glob(f"{OUT}/result_*_s{SPLIT}.json")):
    r = json.load(open(f))
    rows.append(dict(variant=r["args"]["variant"], train_seed=r["args"]["train_seed"],
                     CER_raw=r["norm"]["free"]["CER"], CER_guard=r["norm"]["free+guard"]["CER"],
                     EM=r["norm"]["free"]["Exact"]))
display(pd.DataFrame(rows).round(2))
