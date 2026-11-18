# One Encoding for Every Word

Code, data splits, annotations, and saved outputs for "One Encoding for Every Word: Efficient Lexical Normalization of
Code-Mixed Bengali–English Text" by Tanzim Hasan Prappo, Nahid Gazi, Nazrul Islam, and M. Shahidur Rahman
(Shahjalal University of Science and Technology).

## Contents

```
data/splits/                  train/val/test comment IDs for split seeds 13, 42, 2026
experiments/                  corpus audit, classical LID, lexicon normalization, confident-learning audit
kaggle/src/, kaggle/notebooks/  GPU training and evaluation (ctt.py trains CTT and all compared systems)
analysis/                     statistics, tests, and tables computed from the saved outputs
annotation/adjudicated_test/  adjudicated 600-comment test subset: guidelines, sheets, adjudication, scoring
results/cpu/, results/gpu/    saved predictions, per-run configurations, logs, and timings
```

## Data

BanglaMix is not included. Download it from its data article (Afridi, Partha, and Khan, Data in Brief 69 (2026) 113317,
https://doi.org/10.1016/j.dib.2026.113317; CC BY 4.0) and place the CSV at `data/BanglaMix.csv`. The file used in the
article has SHA-256 `8108215f738e7daa06b2082c9945f03ae5ab5d08e76c869e2368233dee9fcd89`.

## Environments and hardware

| Work | Hardware | Environment file | Versions |
|---|---|---|---|
| CPU analyses and tests (`experiments/`, `analysis/`, `annotation/`) | any CPU (run on an Apple M2 laptop) | `requirements.txt` = `environment/cpu-analysis.txt`, `environment/environment-cpu.yml` | Python 3.13.9, pinned |
| Exactness check and M2 timings | Apple M2 GPU (PyTorch MPS, fp32); `--device cpu` also works | `environment/m2-timing.txt` | Python 3.13.9, PyTorch 2.14.0, Transformers 4.57.6 |
| GPU runs nb12, nb13, nb15 | Kaggle, NVIDIA T4 x2 (16 GB each), fp32 | `environment/kaggle-gpu-2026.txt` | Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 5.20.2, CUDA 13.0, driver 580.178.04 (stored in each `result_*.json`) |
| Original GPU runs nb1--nb11 | Kaggle, one NVIDIA T4 (16 GB), fp32 | Kaggle's preinstalled GPU image at the time | not recorded |

```
pip install -r requirements.txt                    # CPU analyses
pip install -r environment/m2-timing.txt           # exactness check and M2 timings
```

All GPU runs load `google/byt5-small`; its Hugging Face repository has had one revision,
`68377bdc18a2ffec8a0533fef03b1c513a4dd49d`, since 2023-01-24, so the original runs used the same checkpoint. The
original runs used Kaggle's default image of their time, whose library versions were not recorded. Training is not
bit-identical across software images (the same seed gives slightly different CER), so comparisons between systems
use runs from one image, and `environment/kaggle-gpu-2026.txt` pins the image of the additional runs. To rerun a
notebook outside Kaggle, install that file (torch from the CUDA 13.0 index) on a 16 GB GPU.

Approximate T4 training times per run (5 epochs, BanglaMix): word level 1.5 h, context-only CTT 1.8--1.9 h, CTT
2.3--2.6 h including the warm start, ÚFAL-style re-encoding 8.5 h. Timings in the article use GPU synchronization
(`torch.cuda.synchronize` on the T4, `torch.mps.synchronize` on the M2) and report means of three repeats.

MultiLexNorm is not included; to rerun the MultiLexNorm scripts, clone
https://bitbucket.org/robvanderg/multilexnorm.git at commit e58e946375299a04355fa2d7fa9e1c769234e68a and run

```
python kaggle/src/mln_convert.py --repo <multilexnorm> --out /tmp/mln --langs iden trde
cp /tmp/mln/iden/mln.csv results/gpu/nb9_multilexnorm/mln_iden.csv
cp /tmp/mln/trde/mln.csv results/gpu/nb9_multilexnorm/mln_trde.csv
```

## Reproducing the results (CPU, from the saved outputs)

```
python experiments/01_data_audit.py
python experiments/02_lid_baselines.py
python experiments/03_translit_baselines.py
python experiments/04_noise_audit.py
python experiments/05_extra_analyses.py
python analysis/review_analyses.py
python analysis/analyze_seeds.py
python analysis/randomization_tests.py
python analysis/codemixed_eval.py
python analysis/analyze_mln.py
python analysis/mln_copy_err.py
python analysis/clustered_tests.py
python analysis/chunk_alignment_sensitivity.py
python analysis/replicate_tests.py
python annotation/adjudicated_test/reference_sensitivity.py
```

The confidence intervals and p-values in the article come from `analysis/clustered_tests.py`. It swaps and resamples
all appearances of a near-duplicate group together, across splits and training runs, and on MultiLexNorm all seeds of
a test sentence together; its output is `analysis/clustered_tests_output.txt`. `randomization_tests.py`,
`analyze_seeds.py`, and `codemixed_eval.py` give the earlier comment-level versions of the same tests.
`analysis/chunk_alignment_sensitivity.py` compares single- and multi-chunk comments and chunk-boundary tokens, and
tests training with and without the aligner-recovered comments.
`analysis/replicate_tests.py` treats splits, training runs, and MultiLexNorm seeds as replicates, so its tests include
training variation. `annotation/adjudicated_test/reference_sensitivity.py` scores the systems against each annotator's
references and reweights the adjudicated subset to the composition of the split-42 test set.

The exactness of one shared encoder pass and the encoder cost of per-word re-encoding (Proposition 6.1 and the
second-hardware timing in the article) are checked with

```
python analysis/exactness_and_speed.py --root . --device mps --out exactness_and_speed.json
```

The output of the run reported in the article (Apple M2 GPU) is `analysis/exactness_and_speed_m2.json`.
It builds the ByT5-small architecture with random weights (neither property depends on the weights), so it needs no
download; use `--device cpu` without an Apple GPU. If `transformers` fails because of an installed TensorFlow, set
`USE_TF=0`.

The full inference pipeline (input preparation to output text) is timed for every system at two encoder budgets with

```
python analysis/pipeline_timing.py --root . --device mps --out pipeline_timing.json
```

It also uses random weights and fixes each word's number of decoding steps to the length of the saved output of the
trained system. The run reported in the article took about three hours on an Apple M2; its output is
`analysis/pipeline_timing_m2.json`.

### Three-run comparison with re-encoding (one software image)

The original GPU runs did not record their library versions. The additional split-42 runs were trained later in one
Kaggle image (Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 5.20.2, CUDA 13.0, NVIDIA T4, driver 580.178.04),
with `google/byt5-small` pinned to revision `68377bdc18a2ffec8a0533fef03b1c513a4dd49d` (unchanged on the Hugging Face
Hub since 2023-01-24); each `result_*.json` of these runs stores the versions under `env`. Because the same seed gives
different outputs in different images, this comparison uses only runs from that image:

| Notebook | Runs | Output folder |
|---|---|---|
| nb12 | CTT, training seeds 1, 2, 3 | `results/gpu/nb12_ctt_seeds/` |
| nb13 | ÚFAL-style re-encoding, training seeds 1, 2 | `results/gpu/nb13_reencoding_seeds/` |
| nb15 | re-encoding, seed 3; CTT with a 320-byte chunk budget, seeds 1, 2 | `results/gpu/nb15_reencoding_seed3_chunk320/` |

```
python analysis/reencoding_noninferiority.py
```

tests non-inferiority of CTT to re-encoding with the margin of 0.20 CER points fixed before the runs, on seed-matched
pairs with near-duplicate groups kept together; its output is `analysis/reencoding_noninferiority_output.txt`. It
includes the nb15 seed-3 pair once that folder exists.

The full inference pipeline with the trained weights (CTT seed 3 from nb12, re-encoding seed 1 from nb13, both saved
with `--save_weights 1`; the weight files are not included because of their size) is timed with

```
python analysis/pipeline_timing_trained.py --root . --ctt_weights weights_ctt_ts3_s42.pt --ufal_weights weights_ufal_plain_ts1_s42.pt --device mps --out pipeline_timing_trained.json
```

The adjudicated subset is scored with

```
python annotation/adjudicated_test/agreement.py --a annotation/adjudicated_test/sheets/full_annotator_A_done.xlsx --b annotation/adjudicated_test/sheets/full_annotator_B_rechecked.xlsx --out annotation/adjudicated_test/results/full
python annotation/adjudicated_test/evaluate.py --agreed annotation/adjudicated_test/results/full_agreed.csv --adjudication annotation/adjudicated_test/results/full_adjudication_done.xlsx --out annotation/adjudicated_test/results/final
```

## Where each table and figure comes from

All commands run from the repository root on the saved outputs, without a GPU, unless stated otherwise.

| Article | Content | Command(s) | Saved outputs used |
|---|---|---|---|
| Table 4, Figure 2 | corpus statistics, near-duplicates, OOV rate | `python experiments/01_data_audit.py`, `python experiments/05_extra_analyses.py` | `data/BanglaMix.csv`, `data/splits/` |
| Table 5 | confident-learning flags, two-annotator LID audit, alignment recovery | `python experiments/04_noise_audit.py`, `python analysis/review_analyses.py` | `results/cpu/04_*` |
| Table 6 | training settings | `kaggle/src/ctt.py` (defaults) and the `args` of each `result_*.json` | -- |
| Table 7, Figure 4 | encoder bytes, timings, throughput, memory | `python analysis/analyze_ufal.py`, `python analysis/analyze_seeds.py` | `efficiency` and `history` in `results/gpu/nb5_ctt`, `nb6_ufal`, `nb11_train_seeds`, `nb12_ctt_seeds`, `nb13_reencoding_seeds` |
| Section 8, M2 checks | exactness, encoder-only and full-pipeline timing | `python analysis/exactness_and_speed.py ...`, `python analysis/pipeline_timing.py ...`, `python analysis/pipeline_timing_trained.py ...` (Apple M2) | `analysis/*_m2.json` |
| Table 8, Table 17 | normalization results per split and mean ± std | `python analysis/review_analyses.py`, `python experiments/03_translit_baselines.py`, `python analysis/analyze_stages.py` | `results/gpu/nb2_byt5`, `nb3_llm_qwen`, `nb5_ctt`, `nb6_ufal` |
| Table 9 | paired comparisons with clustered intervals and p-values | `python analysis/clustered_tests.py` | as Table 8 |
| Table 10 | matched context test over three training runs | `python analysis/analyze_seeds.py`, `python analysis/clustered_tests.py`, `python analysis/replicate_tests.py` | `results/gpu/nb11_train_seeds`, `nb8_oracle_tags` |
| Section 9.1, three-run comparison | non-inferiority of CTT to re-encoding | `python analysis/reencoding_noninferiority.py` | `results/gpu/nb12_ctt_seeds`, `nb13_reencoding_seeds` |
| Table 11 | code-mixed, monolingual, and switch-point evaluation | `python analysis/codemixed_eval.py` | as Table 8 |
| Table 12 | adjudicated references | `annotation/adjudicated_test/agreement.py`, `evaluate.py`, `reference_sensitivity.py` (commands below) | `annotation/adjudicated_test/` |
| Table 13 | ablations, oracle tags, copy gate | `python analysis/ablations_and_copy_gate.py`, `python analysis/analyze_oracle.py` | `results/gpu/nb5_ctt`, `nb8_oracle_tags` |
| Table 14 | MultiLexNorm | `python analysis/analyze_mln.py`, `python analysis/score_mln_submissions.py`, `python analysis/mln_copy_err.py` | `results/gpu/nb9_multilexnorm` |
| Table 15, Figure 5 | token LID | `python experiments/02_lid_baselines.py` | `results/cpu/02_*`, `results/gpu/nb1_lid_transformers`, `nb3_llm_qwen` |
| Section 13, Figure 6 | error taxonomy and error annotation | `python analysis/review_analyses.py`, `python analysis/chunk_alignment_sensitivity.py` | `analysis/error_annotation/` |
| Table 16 | summary of Tables 7, 9, 10, 13, and 14 | -- | -- |
| Repetition guard | loop counts and guarded CER | `python analysis/repetition_guard.py` | as Table 8 |

## Retraining (GPU)

`python kaggle/build_notebooks.py` writes the notebooks and `kaggle/kaggle_dataset.zip`. Attach the zip as a Kaggle
dataset, run the notebooks, and copy each output folder into `results/gpu/`.

## System names

| Article | Variant in code | Notebook | Options |
|---|---|---|---|
| CTT | `ctx_plain_aks_heads` (`ctt` in nb11, `ctt_env2`/`ctt` in nb12) | nb5, nb11, nb12 | `--mode focus --align 1 --pretrain_aksharantar 200000 --lam 0.4 --copy_mu 0.5 --detach_heads 1` |
| Context-only CTT | `ctx_plain` | nb5, nb6, nb11 | `--mode focus --align 1` |
| ÚFAL-style re-encoding | `ufal_plain` | nb6, nb13, nb15 | `--mode ufal --align 1 --max_enc_tokens 4096` |
| Word level (matched) | `word_plain_align` | nb8, nb11 | `--mode word --align 1` |
| `word_crf` | `word_crf` | nb5 | `--mode word --ext_lid crf --align 0` |
| Word level, aligner, CRF tag | `word_crf_align` | nb5, nb8 | `--mode word --ext_lid crf --align 1` |
| Context with CRF tag | `ctx_crf` | nb5, nb8 | `--mode focus --ext_lid crf --align 1` |
| Context with joint LID | `ctx_joint` | nb5 | `--mode focus --align 1 --lam 0.4 --inject 1 --copy_mu 0.5` |
| MultiLexNorm systems | `word`, `ctx`, `ufal`, `ctx_copy` | nb9 | `ctx_copy` = `ctx` + `--copy_mu 0.5 --detach_heads 1` |

All runs use `google/byt5-small`, AdamW (learning rate 3e-4), 5 epochs on BanglaMix and 10 on MultiLexNorm, chunks of
at most 160 bytes, copy threshold 0.9, and greedy decoding. Each `result_*.json` stores the run's arguments, training
history, and timings. Seeds: splits 13, 42, 2026; extra split-42 training seeds 1 and 2 (nb11) and 1–3 (nb12, nb13, nb15); MultiLexNorm
seeds 13, 42, 2026.

## Licence

Code: MIT (see LICENSE). BanglaMix is not included and remains under CC BY 4.0.
