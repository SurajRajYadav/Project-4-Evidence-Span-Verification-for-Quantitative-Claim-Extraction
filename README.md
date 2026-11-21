# Verifiable claim extraction from scientific literature (Project 4)

**Research question.** How reliably can LLMs extract quantitative property claims (material, property, value, unit) from materials-science abstracts, and does forcing the model to quote a supporting span, then string-matching that span against the source, reduce hallucinated numbers?

Compared methods (all cross-validated on your own annotations):

| method | description |
|---|---|
| `regex_baseline` | rule-based: every number+unit mention is a claim (no model) |
| `zero_shot_text` | zero-shot, pipe-separated text output |
| `zero_shot_json` | zero-shot, structured JSON with evidence span and confidence |
| `few_shot_json` | JSON + 3 fixed demonstrations |
| `rag_json` | JSON + 3 demonstrations retrieved from the training fold by embedding similarity |
| `lora_json` | QLoRA-fine-tuned model (retrained in each fold) |
| `*+verify` | any JSON method after dropping records whose evidence is not a verbatim span of the source containing the value |

Metrics: precision/recall/F1 at three match levels (`strict`, `relaxed`, `value_unit`) with document-level bootstrap 95% CIs; paired bootstrap tests; numeric hallucination rate; calibration (ECE, Brier, reliability diagram) of verbalised confidence; automatic error taxonomy; inter-annotator agreement (Cohen's kappa).

## Quick start (Google Colab)

1. Open Colab and set **Runtime > Change runtime type > T4 GPU**.
2. Upload `colab_runner.ipynb` (it is also inside the zip). Put `claim_study.zip` in the root of your Google Drive (`MyDrive/claim_study.zip`) or choose it from your computer when the first cell asks.
3. Run the cells top to bottom. The runner mounts Drive and links `data/` and `results/` to `MyDrive/claim_data` and `MyDrive/claim_results`, so annotations and finished folds survive a disconnect.
4. The runner does a smoke test first (`python smoke_test.py`, CPU only, ~10 s), then the stages below.

## Stages (`run_experiments.py`)

```bash
python run_experiments.py --stages fetch sheet          # abstracts + annotation CSVs (A: LLM draft, B: blind 25%)
#   >>> correct both CSVs by hand (docs/ANNOTATION_GUIDELINES.md), put them back in data/ <<<
python run_experiments.py --stages gold                 # CSV -> gold, validation issues, kappa / agreement
python run_experiments.py --stages run analyze          # all methods, 5-fold CV, tables + figures
python run_experiments.py --stages run analyze --demo --mock --folds 3   # CPU plumbing test on synthetic data
```

The lower-level `cli.py` commands below do the same things one at a time.

## Full workflow (cli.py)

```bash
pip install -r requirements.txt

# 1. collect ~150 abstracts with numeric content (arXiv metadata is CC0)
python cli.py fetch --n 150 --out data/snippets.jsonl

# 2. annotator A: LLM draft that YOU must correct (CSV opens in Google Sheets / Excel)
python cli.py make-sheet --preannotate --out data/annotation_sheet_A.csv
#    ... correct the CSV following docs/ANNOTATION_GUIDELINES.md, save it ...
python cli.py build-gold --sheet data/annotation_sheet_A.csv --annotator A --out data/gold_A.jsonl
#    prints validation issues (values/units not found in the text): fix them in the CSV

# 3. annotator B: blind 25% subset, no draft -> agreement
python cli.py make-sheet --subset 0.25 --out data/annotation_sheet_B.csv
python cli.py build-gold --sheet data/annotation_sheet_B.csv --annotator B --out data/gold_B.jsonl
python cli.py agreement --a data/gold_A.jsonl --b data/gold_B.jsonl

# 4. run all methods with 5-fold cross-validation (resumable: finished folds are skipped)
python cli.py run --gold data/gold_A.jsonl --out results --folds 5

# 5. tables, figures, summary
python cli.py analyze --gold data/gold_A.jsonl --out results
```

Outputs in `results/`: `preds/` (raw model output for every doc/method/fold), `tables/*.csv` (main results, paired tests, calibration, error taxonomy, `failures_to_review.csv`), `figures/*.pdf|png` (poster-ready vector figures), `summary.md`.

## Useful options

| option | default | note |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-3B-Instruct` | try `Qwen/Qwen2.5-7B-Instruct` on a Colab Pro/A100 (4-bit) for stronger results |
| `--folds` | 5 | LoRA is retrained per fold; use `--methods` to skip it while debugging |
| `--methods` | all | e.g. `--methods zero_shot_json rag_json regex_baseline` |
| `--epochs` | 3 | LoRA epochs |
| `--batch_size` | 4 | lower if you run out of GPU memory |
| `--mock` | off | CPU plumbing test, no real LLM, scores are meaningless |

Rough time on a free T4 (3B model, 150 docs, 5 folds): prompted methods ~30-40 min, LoRA ~60-90 min. Results are saved per fold, so if Colab disconnects, re-run the same command and it resumes. Mount Google Drive and use `--out /content/drive/MyDrive/...` to keep results across sessions.

## What to do by hand (this is what makes it research, not a demo)

1. **Annotate carefully.** The LLM draft is only a speed-up. Read every snippet; note in the paper that drafts were used and that annotator B worked blind.
2. **Review failures.** Open `results/tables/failures_to_review.csv`, fill the `human_category` column for 50-100 rows, and refine the automatic taxonomy.
3. **Check at least 2 prompts.** Edit `claim/prompts.py` (one variant at a time) and re-run on a copy of `results/`; report prompt sensitivity.
4. **Report limitations honestly:** one domain, abstracts only, ~150 documents, one model family, verbalised (not logit-based) confidence, LLM-draft annotation bias, hallucination = value/unit absent from the source (it does not catch a *real* number attached to the *wrong* property).
5. **Stretch goals:** a second model (e.g. Llama-3.1-8B), a second domain, token-probability confidence, 8-bit comparison.

## Repository layout

```
run_experiments.py      stage runner (fetch / sheet / gold / run / analyze)
smoke_test.py           CPU-only wiring check
colab_runner.ipynb      Colab notebook
cli.py                  lower-level commands
claim/utils.py            normalisation of values/units/text, IO
claim/data.py             arXiv fetch, annotation sheets, gold building, demo data
claim/prompts.py          prompts + output parsers
claim/models.py           HF model wrapper (4-bit), retriever, mock model
claim/lora.py             QLoRA training loop
claim/experiment.py       cross-validated runner (resumable)
claim/metrics.py          matching, F1, bootstrap, hallucination, verification, calibration, kappa, taxonomy
claim/analysis.py         tables, figures, summary.md
docs/ANNOTATION_GUIDELINES.md
tests/test_metrics.py   unit tests (pytest)
```

## Reproducibility

Seeds are fixed (`--seed 42`), prompts are in code, decoding is greedy, raw outputs are saved, and `results/config.json` records the configuration. Pin versions with `pip freeze > requirements.lock` after your final run.

## License / data

arXiv abstract metadata is released under CC0. Keep only open-access text in your own dataset, and release your annotations with the guidelines.
