# %% [markdown]
# # nb15: balanced re-encoding seeds and chunk budget
#
# Additional runs in the same Kaggle image as nb12 and nb13. (1) ÚFAL-style re-encoding on split 42 with
# training seed 3, so that CTT and re-encoding each have three runs (seeds 1, 2, 3) in one software image. (2) Full CTT
# with a 320-byte chunk budget instead of 160, training seeds 1 and 2, to compare with the 160-byte CTT runs of the same
# seeds (nb12). All other settings are unchanged.
#
# Accelerator: GPU T4 x2 (about 9 hours: re-encoding on one GPU, the two 320-byte runs on the other). With one GPU, run
# the notebook again with this notebook's earlier output attached as input; finished jobs are copied and skipped.

# %%
!pip -q install rapidfuzz 2>/dev/null

# %% writefile banglish_align.py

# %% writefile ctt.py

# %%
import glob, json, os, subprocess, sys, time
import pandas as pd, torch

OUT = "/kaggle/working/ctt"
SCRIPT = "/kaggle/working/ctt.py"
EPOCHS = 5
MODEL, REVISION = "google/byt5-small", "68377bdc18a2ffec8a0533fef03b1c513a4dd49d"
BASE_ARGS = ["--batch_words", "48", "--accum", "2", "--max_chunks", "8", "--gen_bs", "48"]
POLL_SECONDS = 300
CTT = dict(mode="focus", align=1, pretrain_aksharantar=200000, lam=0.4, copy_mu=0.5, detach_heads=1)
UFAL = dict(mode="ufal", align=1, max_enc_tokens=4096)
JOBS = [("ufal_plain_ts3", 42, 3, UFAL),
        ("ctt_b320_ts1", 42, 1, dict(CTT, src_budget=320)),
        ("ctt_b320_ts2", 42, 2, dict(CTT, src_budget=320))]

os.makedirs(OUT, exist_ok=True)
import shutil
for f in glob.glob("/kaggle/input/**/ctt/*", recursive=True):
    if not os.path.exists(os.path.join(OUT, os.path.basename(f))):
        shutil.copy(f, OUT)
done = lambda j: os.path.exists(f"{OUT}/result_{j[0]}_s{j[1]}.json")
n_gpu = max(1, torch.cuda.device_count())
todo = [j for j in JOBS if not done(j)]
QUEUES = [todo[g::n_gpu] for g in range(n_gpu)]
print(n_gpu, "GPU(s); queues:", [[f"{j[0]}_s{j[1]}" for j in q] for q in QUEUES])

# %%
def launch(gpu, job):
    v, split, ts, cfg = job
    logf = open(f"{OUT}/log_{v}_s{split}.txt", "w")
    cmd = [sys.executable, SCRIPT, "--variant", v, "--seed", str(split), "--epochs", str(EPOCHS),
           "--model", MODEL, "--revision", REVISION, "--out", OUT]
    if ts is not None:
        cmd += ["--train_seed", str(ts)]
    for k, x in cfg.items():
        cmd += [f"--{k}", str(x)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    return subprocess.Popen(cmd + BASE_ARGS, stdout=logf, stderr=subprocess.STDOUT, env=env), f"{v}_s{split}", logf

def tail(name):
    try:
        lines = [l for l in open(f"{OUT}/log_{name}.txt") if l.startswith("[")]
        return lines[-1].strip() if lines else ""
    except FileNotFoundError:
        return ""

t0, running = time.time(), {}
while any(QUEUES) or running:
    for g, q in enumerate(QUEUES):
        if g not in running and q:
            running[g] = launch(g, q.pop(0))
    time.sleep(POLL_SECONDS)
    for g, (proc, name, logf) in list(running.items()):
        if proc.poll() is not None:
            logf.close()
            print(f"[{(time.time() - t0) / 60:6.1f} min] finished {name} exit={proc.returncode}")
            if proc.returncode != 0:
                print(open(logf.name).read()[-2000:])
            del running[g]
        else:
            print(f"[{(time.time() - t0) / 60:6.1f} min] gpu{g} {tail(name)}")

# %%
rows = []
for f in sorted(glob.glob(f"{OUT}/result_*.json")):
    r = json.load(open(f))
    a = r["args"]
    rows.append(dict(variant=a["variant"], split=a["seed"], train_seed=a["train_seed"],
                     CER=r["norm"]["free"]["CER"], WER=r["norm"]["free"]["WER"], EM=r["norm"]["free"]["Exact"],
                     words_per_s=r["efficiency"]["free"]["total_words_per_sec"],
                     train_min=sum(h["train_min"] for h in r["history"] if h["stage"] == "banglamix") / EPOCHS))
display(pd.DataFrame(rows).round(2))
print(json.dumps(r.get("env"), indent=1))
