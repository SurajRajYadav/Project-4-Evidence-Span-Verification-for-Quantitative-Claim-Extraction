#!/usr/bin/env python
"""Command line entry point.  python cli.py <command> --help"""
import argparse
import json
import random

from claim import data as D
from claim import metrics as M
from claim.utils import read_jsonl, set_seed, write_jsonl


def cmd_fetch(a):
    docs = D.fetch_arxiv(a.category, a.n, a.seed)
    write_jsonl(a.out, docs)
    print(f"saved {len(docs)} abstracts -> {a.out}")


def cmd_make_sheet(a):
    docs = read_jsonl(a.snippets)
    drafts = None
    if a.subset < 1.0:
        random.Random(a.seed).shuffle(docs)
        docs = docs[: max(1, int(len(docs) * a.subset))]
    if a.preannotate:
        from claim.models import LLM
        from claim.prompts import build_messages, parse_json_records
        llm = LLM(a.model)
        outs = llm.generate([build_messages("zero_shot_json", d["text"]) for d in docs], batch_size=4)
        drafts = {d["id"]: parse_json_records(o)[0] for d, o in zip(docs, outs)}
        llm.close()
    D.write_sheet(a.out, docs, drafts)
    print(f"wrote {len(docs)} docs -> {a.out}")


def cmd_build_gold(a):
    gold = D.sheet_to_gold(a.sheet, a.snippets, a.annotator)
    write_jsonl(a.out, gold)
    issues = D.validate_gold(gold)
    n = sum(len(d["records"]) for d in gold)
    print(f"{len(gold)} docs, {n} records -> {a.out}; {len(issues)} validation issues")
    for s in issues[:50]:
        print("  -", s)


def cmd_agreement(a):
    res = M.agreement(read_jsonl(a.a), read_jsonl(a.b))
    print(json.dumps(res, indent=2))


def cmd_run(a):
    from claim.experiment import run_experiment
    run_experiment(a.gold, a.out, a.model, a.methods, a.folds, a.k_demos, a.epochs, a.batch_size, a.seed,
                   a.mock, not a.no_4bit, a.max_docs)


def cmd_analyze(a):
    from claim.analysis import analyze
    analyze(a.gold, a.out, a.n_boot)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("fetch", help="download arXiv abstracts")
    s.add_argument("--category", default="cond-mat.mtrl-sci")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="data/snippets.jsonl")
    s.set_defaults(f=cmd_fetch)

    s = sp.add_parser("make-sheet", help="create the annotation CSV")
    s.add_argument("--snippets", default="data/snippets.jsonl")
    s.add_argument("--out", default="data/annotation_sheet_A.csv")
    s.add_argument("--preannotate", action="store_true", help="LLM draft that the annotator must correct")
    s.add_argument("--subset", type=float, default=1.0, help="fraction of docs (use 0.25 for annotator B)")
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(f=cmd_make_sheet)

    s = sp.add_parser("build-gold", help="annotation CSV -> gold JSONL (+validation)")
    s.add_argument("--sheet", required=True)
    s.add_argument("--snippets", default="data/snippets.jsonl")
    s.add_argument("--annotator", default="A")
    s.add_argument("--out", default="data/gold_A.jsonl")
    s.set_defaults(f=cmd_build_gold)

    s = sp.add_parser("agreement", help="inter-annotator agreement (Cohen's kappa + F1)")
    s.add_argument("--a", default="data/gold_A.jsonl")
    s.add_argument("--b", default="data/gold_B.jsonl")
    s.set_defaults(f=cmd_agreement)

    s = sp.add_parser("run", help="run all extraction methods (cross-validated)")
    s.add_argument("--gold", default="data/gold_A.jsonl", help="gold JSONL, or the word 'demo'")
    s.add_argument("--out", default="results")
    s.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    s.add_argument("--methods", nargs="+", default=None)
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--k_demos", type=int, default=3)
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch_size", type=int, default=4)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--max_docs", type=int, default=None)
    s.add_argument("--mock", action="store_true", help="CPU plumbing test (no real LLM)")
    s.add_argument("--no_4bit", action="store_true")
    s.set_defaults(f=cmd_run)

    s = sp.add_parser("analyze", help="tables, figures, summary.md")
    s.add_argument("--gold", default="data/gold_A.jsonl")
    s.add_argument("--out", default="results")
    s.add_argument("--n_boot", type=int, default=2000)
    s.set_defaults(f=cmd_analyze)

    a = p.parse_args()
    set_seed(getattr(a, "seed", 42))
    a.f(a)


if __name__ == "__main__":
    main()
