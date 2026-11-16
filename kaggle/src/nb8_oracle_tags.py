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
    "ctx_crf_rerun":        dict(mode="focus", ext_lid="crf", align=1, eval_ext_lid="gold"),
    "word_crf_align_rerun": dict(mode="word", ext_lid="crf", align=1, eval_ext_lid="gold"),
    "word_plain_align":     dict(mode="word", align=1),
}
GPU_QUEUES = [["ctx_crf_rerun"], ["word_crf_align_rerun", "word_plain_align"]]
SEED = 42

os.makedirs(OUT, exist_ok=True)
done = lambda v: os.path.exists(f"{OUT}/result_{v}_s{SEED}.json")
n_gpu = max(1, torch.cuda.device_count())
if n_gpu < len(GPU_QUEUES):
    GPU_QUEUES = [[v for q in GPU_QUEUES for v in q]]
GPU_QUEUES = [[v for v in q if not done(v)] for q in GPU_QUEUES]
print(n_gpu, "GPU(s); queues:", GPU_QUEUES)

# %%
def launch(gpu, v):
    logf = open(f"{OUT}/log_{v}_s{SEED}.txt", "w")
    cmd = [sys.executable, SCRIPT, "--variant", v, "--seed", str(SEED), "--epochs", str(EPOCHS), "--model", MODEL,
           "--out", OUT]
    for k, x in VARIANTS[v].items():
        cmd += [f"--{k}", str(x)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return subprocess.Popen(cmd + BASE_ARGS, stdout=logf, stderr=subprocess.STDOUT, env=env), v, logf

def tail(v):
    try:
        lines = [l for l in open(f"{OUT}/log_{v}_s{SEED}.txt") if l.startswith("[")]
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
for f in sorted(glob.glob(f"{OUT}/result_*.json")):
    r = json.load(open(f))
    v = r["args"]["variant"]
    rows.append(dict(run=v, test_tags=r["args"]["ext_lid"], CER=r["norm"]["free"]["CER"],
                     CER_guard=r["norm"]["free+guard"]["CER"], EM=r["norm"]["free+guard"]["Exact"]))
    for src, m in r.get("norm_ext", {}).items():
        rows.append(dict(run=v, test_tags=src, CER=m["free"]["CER"], CER_guard=m["free+guard"]["CER"],
                         EM=m["free+guard"]["Exact"]))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/nb8_cells.csv", index=False)
if len(R):
    display(R.round(2))
