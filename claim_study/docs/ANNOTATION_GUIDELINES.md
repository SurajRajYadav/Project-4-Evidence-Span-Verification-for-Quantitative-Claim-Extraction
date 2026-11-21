# Annotation guidelines (v1)

Domain: materials science abstracts (arXiv `cond-mat.mtrl-sci`). Goal: record every **quantitative property claim**.

## Record schema
| field | meaning | example |
|---|---|---|
| material | material/system the value belongs to, as specific as the text allows | `MoS2 monolayer` |
| property | the measured or computed property | `optical band gap` |
| value | the number exactly as written, no unit; keep ranges and ± | `1.85`, `300-400`, `5.2 ± 0.3` |
| unit | unit exactly as written; empty if dimensionless | `eV`, `W/mK`, `%`, `` |
| evidence | verbatim sentence/clause with the value (auto-filled if left blank) | |

## Rules
1. **Include** values reported as results: measured, computed, predicted, or achieved properties (band gap, strength, efficiency, conductivity, capacity, lattice constant, grain size, diameter...).
2. **Exclude** experimental conditions and bookkeeping: measurement temperature/pressure, annealing temperature, number of cycles, doping level, current rate (0.1 C), years, citation numbers, counts of screened candidates.
3. One record per (material, property, value). "from 450 MPa to 1.2 GPa" gives two records.
4. Values compared to a reference ("twice the bulk value") are not records unless the number itself is stated.
5. Do not convert units, compute new numbers, or infer values not stated.
6. Material names: use the name from the text; add a short qualifier when needed (`Na-doped SnSe`). Matching is fuzzy, so small naming differences are tolerated by the relaxed metric.
7. Docs with no claims: leave one row with empty fields.
8. Ambiguous cases: add a note in the `notes` column; discuss them with the second annotator *after* the blind pass and update this file (v2).

## Workflow
1. Annotator A edits `annotation_sheet_A.csv` (LLM draft pre-filled, must be fully checked; remove or fix wrong rows, add missed ones).
2. Annotator B labels a random 20-30% subset **blind** (no pre-filled drafts) from `annotation_sheet_B.csv`.
3. `python cli.py agreement` reports Cohen's kappa and pairwise F1. Report both in the paper, together with these guidelines.
