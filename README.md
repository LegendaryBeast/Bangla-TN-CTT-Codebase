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

## Setup

```
pip install -r requirements.txt
```

The GPU notebooks were run on Kaggle (one NVIDIA T4, 16 GB). MultiLexNorm is not included; to rerun the
MultiLexNorm scripts, clone https://bitbucket.org/robvanderg/multilexnorm.git at commit
e58e946375299a04355fa2d7fa9e1c769234e68a and run

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
```

The confidence intervals and p-values in the article come from `analysis/clustered_tests.py`. It swaps and resamples
all appearances of a near-duplicate group together, across splits and training runs, and on MultiLexNorm all seeds of
a test sentence together; its output is `analysis/clustered_tests_output.txt`. `randomization_tests.py`,
`analyze_seeds.py`, and `codemixed_eval.py` give the earlier comment-level versions of the same tests.
`analysis/chunk_alignment_sensitivity.py` compares single- and multi-chunk comments and chunk-boundary tokens, and
tests training with and without the aligner-recovered comments.

The exactness of one shared encoder pass and the encoder cost of per-word re-encoding (Proposition 1 and the
second-hardware timing in the article) are checked with

```
python analysis/exactness_and_speed.py --root . --device mps --out exactness_and_speed.json
```

The output of the run reported in the article (Apple M2 GPU) is `analysis/exactness_and_speed_m2.json`.
It builds the ByT5-small architecture with random weights (neither property depends on the weights), so it needs no
download; use `--device cpu` without an Apple GPU. If `transformers` fails because of an installed TensorFlow, set
`USE_TF=0`.

The adjudicated subset is scored with

```
python annotation/adjudicated_test/agreement.py --a annotation/adjudicated_test/sheets/full_annotator_A_done.xlsx --b annotation/adjudicated_test/sheets/full_annotator_B_rechecked.xlsx --out annotation/adjudicated_test/results/full
python annotation/adjudicated_test/evaluate.py --agreed annotation/adjudicated_test/results/full_agreed.csv --adjudication annotation/adjudicated_test/results/full_adjudication_done.xlsx --out annotation/adjudicated_test/results/final
```

## Retraining (GPU)

`python kaggle/build_notebooks.py` writes the notebooks and `kaggle/kaggle_dataset.zip`. Attach the zip as a Kaggle
dataset, run the notebooks, and copy each output folder into `results/gpu/`.

## System names

| Article | Variant in code | Notebook | Options |
|---|---|---|---|
| CTT | `ctx_plain_aks_heads` (`ctt` in nb11) | nb5, nb11 | `--mode focus --align 1 --pretrain_aksharantar 200000 --lam 0.4 --copy_mu 0.5 --detach_heads 1` |
| Context-only CTT | `ctx_plain` | nb5, nb6, nb11 | `--mode focus --align 1` |
| ÚFAL-style re-encoding | `ufal_plain` | nb6 | `--mode ufal --align 1 --max_enc_tokens 4096` |
| Word level (matched) | `word_plain_align` | nb8, nb11 | `--mode word --align 1` |
| `word_crf` | `word_crf` | nb5 | `--mode word --ext_lid crf --align 0` |
| Word level, aligner, CRF tag | `word_crf_align` | nb5, nb8 | `--mode word --ext_lid crf --align 1` |
| Context with CRF tag | `ctx_crf` | nb5, nb8 | `--mode focus --ext_lid crf --align 1` |
| Context with joint LID | `ctx_joint` | nb5 | `--mode focus --align 1 --lam 0.4 --inject 1 --copy_mu 0.5` |
| MultiLexNorm systems | `word`, `ctx`, `ufal`, `ctx_copy` | nb9 | `ctx_copy` = `ctx` + `--copy_mu 0.5 --detach_heads 1` |

All runs use `google/byt5-small`, AdamW (learning rate 3e-4), 5 epochs on BanglaMix and 10 on MultiLexNorm, chunks of
at most 160 bytes, copy threshold 0.9, and greedy decoding. Each `result_*.json` stores the run's arguments, training
history, and timings. Seeds: splits 13, 42, 2026; extra split-42 training seeds 1 and 2; MultiLexNorm seeds 13, 42, 2026.

## Licence

Code: MIT (see LICENSE). BanglaMix is not included and remains under CC BY 4.0.
