#!/usr/bin/env python
"""Stage runner. Example:  python run_experiments.py --stages fetch sheet

Stages (in order):
  fetch      download arXiv abstracts                       -> data/snippets.jsonl
  sheet      create annotation CSVs (A: LLM draft, B: blind) -> data/annotation_sheet_{A,B}.csv
  gold       CSVs -> gold JSONL, validation, inter-annotator agreement
  run        run all extraction methods, cross-validated    -> results/<run>/preds
  analyze    tables, figures, summary.md                    -> results/<run>/

Between `sheet` and `gold` a human must correct the CSVs (see docs/ANNOTATION_GUIDELINES.md).
`--demo` runs run+analyze on 12 synthetic snippets (add --mock for a CPU-only plumbing test).
"""
import argparse
import types

import cli

STAGES = ["fetch", "sheet", "gold", "run", "analyze"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--n", type=int, default=150, help="abstracts to collect")
    p.add_argument("--category", default="cond-mat.mtrl-sci")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--run-name", default="run1", help="results/<run-name>")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--b-fraction", type=float, default=0.25, help="share of docs for blind annotator B")
    p.add_argument("--no-preannotate", action="store_true", help="annotator A sheet starts blank (no LLM draft)")
    p.add_argument("--demo", action="store_true", help="use built-in synthetic gold data for run/analyze")
    p.add_argument("--mock", action="store_true", help="no real LLM (CPU plumbing test)")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--max-docs", type=int, default=None)
    a = p.parse_args()

    d = a.data_dir
    gold_a = "demo" if a.demo else f"{d}/gold_A.jsonl"
    out = f"results/{a.run_name}"
    ns = lambda **kw: types.SimpleNamespace(seed=a.seed, **kw)

    for st in [s for s in STAGES if s in a.stages]:
        print(f"\n===== STAGE: {st} =====")
        if st == "fetch":
            cli.cmd_fetch(ns(category=a.category, n=a.n, out=f"{d}/snippets.jsonl"))
        elif st == "sheet":
            cli.cmd_make_sheet(ns(snippets=f"{d}/snippets.jsonl", out=f"{d}/annotation_sheet_A.csv",
                                  preannotate=not a.no_preannotate, subset=1.0, model=a.model))
            cli.cmd_make_sheet(ns(snippets=f"{d}/snippets.jsonl", out=f"{d}/annotation_sheet_B.csv",
                                  preannotate=False, subset=a.b_fraction, model=a.model))
            print("\nNext: correct both CSVs by hand (B = a second person, blind), put them back in",
                  d, "and run --stages gold")
        elif st == "gold":
            for who in ["A", "B"]:
                cli.cmd_build_gold(ns(sheet=f"{d}/annotation_sheet_{who}.csv", snippets=f"{d}/snippets.jsonl",
                                      annotator=who, out=f"{d}/gold_{who}.jsonl"))
            cli.cmd_agreement(ns(a=f"{d}/gold_A.jsonl", b=f"{d}/gold_B.jsonl"))
        elif st == "run":
            cli.cmd_run(ns(gold=gold_a, out=out, model=a.model, methods=a.methods, folds=a.folds, k_demos=3,
                           epochs=a.epochs, batch_size=a.batch_size, mock=a.mock, no_4bit=a.no_4bit,
                           max_docs=a.max_docs))
        elif st == "analyze":
            cli.cmd_analyze(ns(gold=gold_a, out=out, n_boot=2000))


if __name__ == "__main__":
    main()
