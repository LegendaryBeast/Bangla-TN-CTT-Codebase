# Adjudicated test subset

600 comments of the split-42 test set (all 464 code-mixed comments and 136 monolingual comments), 7,275 tokens.
Two annotators corrected the silver references independently, and a third annotator adjudicated their disagreements.

| File | Contents |
|---|---|
| `GUIDELINES.md` | annotation guidelines |
| `make_sample.py`, `sample_key.csv`, `tokens_key.csv` | sample (seed 2026) and silver targets |
| `sheets/full_annotator_A_done.xlsx`, `sheets/full_annotator_B_rechecked.xlsx` | the two completed annotations |
| `sheets/*_annotator_{A,B}.xlsx` | blank pilot and full sheets |
| `results/full_agreement.txt` | inter-annotator agreement |
| `results/full_adjudication_done.xlsx` | adjudication of the 606 disputed tokens |
| `results/final_report.txt`, `final_scores.csv`, `final_references.csv` | scores of all systems and the final references |
| `agreement.py`, `evaluate.py`, `common.py` | agreement, adjudication sheet, and scoring |
