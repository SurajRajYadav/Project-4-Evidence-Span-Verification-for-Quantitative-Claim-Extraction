"""Cross-validated experiment runner. Saves raw model outputs for every method and fold (resumable)."""
import json
import os
import random

from .data import load_gold
from .models import LLM, MockLLM, Retriever, regex_records
from .lora import train_lora
from .prompts import (METHODS, PROMPTED, build_messages, parse_json_records, parse_text_records)
from .utils import read_jsonl, set_seed, write_jsonl

ALL_METHODS = METHODS + ["regex_baseline"]


def make_folds(ids, k, seed):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    k = max(2, min(k, len(ids)))
    return [ids[i::k] for i in range(k)]


def pick_fixed_demos(train_docs, k, seed):
    pool = sorted([d for d in train_docs if d["records"]], key=lambda d: d["id"])
    return random.Random(seed).sample(pool, min(k, len(pool)))


def _parse(method, raw):
    return parse_text_records(raw) if method == "zero_shot_text" else parse_json_records(raw)


def _predict(llm, method, docs, demos_per_doc, batch_size):
    msgs = [build_messages(method, d["text"], demos) for d, demos in zip(docs, demos_per_doc)]
    outs = llm.generate(msgs, batch_size=batch_size)
    preds = []
    for d, raw in zip(docs, outs):
        recs, ok = _parse(method, raw)
        preds.append({"id": d["id"], "raw": raw, "records": recs, "parse_ok": ok})
    return preds


def run_experiment(gold_path, out_dir, model="Qwen/Qwen2.5-3B-Instruct", methods=None, folds=5, k_demos=3,
                   epochs=3, batch_size=4, seed=42, mock=False, load_4bit=True, max_docs=None, dense_retriever=True):
    set_seed(seed)
    methods = methods or ALL_METHODS
    gold = load_gold(gold_path)
    if max_docs:
        gold = gold[:max_docs]
    by_id = {d["id"]: d for d in gold}
    fold_ids = make_folds(list(by_id), folds, seed)
    pdir = os.path.join(out_dir, "preds")
    os.makedirs(pdir, exist_ok=True)
    json.dump({"gold": gold_path, "model": "mock" if mock else model, "methods": methods, "folds": len(fold_ids),
               "k_demos": k_demos, "epochs": epochs, "seed": seed, "n_docs": len(gold)},
              open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    path = lambda m, f: os.path.join(pdir, f"{m}__fold{f}.jsonl")

    def split(f):
        test = [by_id[i] for i in fold_ids[f]]
        train = [by_id[i] for j, ids in enumerate(fold_ids) if j != f for i in ids]
        return train, test

    if "regex_baseline" in methods:
        for f in range(len(fold_ids)):
            _, test = split(f)
            write_jsonl(path("regex_baseline", f), [{"id": d["id"], "raw": "", "parse_ok": True,
                                                    "records": regex_records(d["text"])} for d in test])

    prompted = [m for m in methods if m in PROMPTED]
    if prompted:
        llm = MockLLM() if mock else LLM(model, load_4bit=load_4bit)
        retr = Retriever(use_dense=dense_retriever and not mock)
        for f in range(len(fold_ids)):
            train, test = split(f)
            for m in prompted:
                if os.path.exists(path(m, f)):
                    print(f"skip {m} fold {f} (exists)")
                    continue
                print(f"run {m} fold {f + 1}/{len(fold_ids)} on {len(test)} docs")
                if m == "few_shot_json":
                    fixed = pick_fixed_demos(train, k_demos, seed)
                    demos = [fixed] * len(test)
                elif m == "rag_json":
                    retr.fit([d["text"] for d in train])
                    demos = [[train[i] for i in reversed(retr.topk(d["text"], k_demos))] for d in test]
                else:
                    demos = [[] for _ in test]
                write_jsonl(path(m, f), _predict(llm, m, test, demos, batch_size))
        llm.close()

    if "lora_json" in methods and not mock:
        for f in range(len(fold_ids)):
            if os.path.exists(path("lora_json", f)):
                print(f"skip lora_json fold {f} (exists)")
                continue
            train, test = split(f)
            print(f"lora_json fold {f + 1}/{len(fold_ids)}: train on {len(train)}, test on {len(test)}")
            llm = LLM(model, load_4bit=load_4bit)
            train_lora(llm, train, epochs=epochs, seed=seed)
            write_jsonl(path("lora_json", f), _predict(llm, "lora_json", test, [[] for _ in test], batch_size))
            llm.close()
    print("done ->", pdir)


def load_preds(out_dir, method):
    pdir = os.path.join(out_dir, "preds")
    rows = []
    for fn in sorted(os.listdir(pdir)):
        if fn.startswith(method + "__fold"):
            rows += read_jsonl(os.path.join(pdir, fn))
    return {r["id"]: r for r in rows}
