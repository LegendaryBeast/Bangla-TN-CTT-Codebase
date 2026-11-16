# Adjudicated normalization test subset (BanglaMix split 42)

600 test comments, 7,275 tokens: all 464 code-mixed comments of the split-42 test set and 136 monolingual comments
(75 Bengali, 38 English, 23 Romanized Bengali). Two annotators check the silver reference token by token, a third
person adjudicates their disagreements, and every controlled system is rescored.

## Files

| File | Purpose |
|---|---|
| `GUIDELINES.md` | annotation guidelines (read first; update the change log after the pilot) |
| `sheets/pilot_annotator_{A,B}.xlsx` | pilot: the first 50 comments (567 tokens), identical for A and B |
| `sheets/full_annotator_{A,B}.xlsx` | all 600 comments (pilot comments first), identical for A and B (blank) |
| `sheets/full_annotator_A_done.xlsx`, `sheets/full_annotator_B_rechecked.xlsx` | the two completed, independent annotations |
| `results/full_agreement.txt` | agreement between the two annotators |
| `results/full_agreed.csv`, `results/full_adjudication.xlsx` | tokens final without adjudication; the sheet given to the adjudicator |
| `results/full_adjudication_done.xlsx` | the completed adjudication (606 tokens, with the adjudicator's notes) |
| `results/final_report.txt`, `results/final_scores.csv`, `results/final_references.csv` | scores of all controlled systems and the final references with accepted variants |
| `sample_key.csv` | one row per comment: order, comment ID, mix type, alignment, pilot flag, silver reference |
| `tokens_key.csv` | one row per token with the silver target (the scoring reference; never edited) |
| `make_sample.py` | draws the sample (seed 2026) and writes the sheets; rerunning reproduces them exactly |
| `agreement.py` | agreement statistics, the list of agreed tokens, and the adjudication sheet |
| `evaluate.py` | scores all controlled systems against silver, adjudicated, and variant-aware references |
| `common.py` | shared helpers |

## Workflow

Run all commands from the repository root.

**1. Pilot.** Annotators A and B each fill their own pilot sheet independently and save it under a new name.

```bash
python annotation/adjudicated_test/agreement.py --a annotation/adjudicated_test/sheets/pilot_annotator_A_done.xlsx --b annotation/adjudicated_test/sheets/pilot_annotator_B_done.xlsx --out annotation/adjudicated_test/results/pilot
```

Discuss only the items in `results/pilot_adjudication.xlsx`, revise `GUIDELINES.md`, and log the changes.

**2. Full annotation.** A and B fill the full sheets independently, re-checking the pilot comments under the revised
guidelines.

```bash
python annotation/adjudicated_test/agreement.py --a annotation/adjudicated_test/sheets/full_annotator_A_done.xlsx --b annotation/adjudicated_test/sheets/full_annotator_B_done.xlsx --out annotation/adjudicated_test/results/full
```

**3. Adjudication.** The adjudicator fills `final_target` (and `final_variants`) for every row of
`results/full_adjudication.xlsx` and saves it as `results/full_adjudication_done.xlsx`. Write `<empty>` for a
deliberately empty target.

**4. Scoring.**

```bash
python annotation/adjudicated_test/evaluate.py --agreed annotation/adjudicated_test/results/full_agreed.csv --adjudication annotation/adjudicated_test/results/full_adjudication_done.xlsx --out annotation/adjudicated_test/results/final
```

`results/final_report.txt` gives the change rates, CER/WER/EM of every system against the three references (all,
code-mixed, monolingual), and two-sided paired randomization tests; `final_scores.csv` and `final_references.csv`
hold the numbers and the adjudicated references.

## Results

Two annotators labelled the full sheets independently, and a third annotator adjudicated the 606 tokens on which they
differed (81 in the corrected form). Agreement on whether a token needs a change is kappa = 0.77; the annotators chose
the same form for 98.9% of tokens. Adjudication changed 2.5% of the silver targets and 13.5% of the comments. Against
the adjudicated references, CTT has the lowest CER of the controlled systems, and the gain of context over the matched
word-level model is preserved (-0.47 CER, p < 0.001). The full numbers are in `results/full_agreement.txt` and
`results/final_report.txt`; they are reported in Sections 5.4 and 9.4 and Table 12 of the article. To reproduce:

```bash
python annotation/adjudicated_test/agreement.py --a annotation/adjudicated_test/sheets/full_annotator_A_done.xlsx --b annotation/adjudicated_test/sheets/full_annotator_B_rechecked.xlsx --out annotation/adjudicated_test/results/full
python annotation/adjudicated_test/evaluate.py --agreed annotation/adjudicated_test/results/full_agreed.csv --adjudication annotation/adjudicated_test/results/full_adjudication_done.xlsx --out annotation/adjudicated_test/results/final
```

Rerunning `agreement.py` rewrites the blank adjudication sheet `results/full_adjudication.xlsx`; the completed one is
`results/full_adjudication_done.xlsx`.

## Checks already run

- With both sheets left unchanged, agreement is 100%, the adjudication sheet is empty, and every adjudicated score
  equals the silver score; the silver code-mixed CERs equal those of `analysis/codemixed_eval.py`.
- With simulated edits, variants, flags, and empty targets, all three scripts run and the outputs are consistent.
- The copy-gate outputs are split back into per-token outputs from the saved texts; 4 of 1,800 comments contain a
  repeat-loop word decoded to a different length and are scored from the saved text directly.
