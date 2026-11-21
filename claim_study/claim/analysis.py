"""Aggregate predictions -> tables (CSV/markdown), figures, and a failure-review sheet."""
import csv
import itertools
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import metrics as M
from .data import load_gold
from .experiment import ALL_METHODS, load_preds
from .prompts import JSON_METHODS

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#9D755D"]


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def analyze(gold_path, out_dir, n_boot=2000, review_n=100, seed=42):
    gold = load_gold(gold_path)
    gold_by = {d["id"]: d["records"] for d in gold}
    texts = {d["id"]: d["text"] for d in gold}
    res_dir, fig_dir = os.path.join(out_dir, "tables"), os.path.join(out_dir, "figures")
    os.makedirs(res_dir, exist_ok=True)

    preds, parse_rate = {}, {}
    for m in ALL_METHODS:
        try:
            raw = load_preds(out_dir, m)
        except FileNotFoundError:
            continue
        if not raw:
            continue
        preds[m] = {i: raw[i]["records"] for i in gold_by if i in raw}
        parse_rate[m] = float(np.mean([raw[i]["parse_ok"] for i in raw]))
    if not preds:
        raise SystemExit("No predictions found. Run the experiment first.")
    ids_all = set(gold_by)
    for m in preds:
        missing = ids_all - set(preds[m])
        if missing:
            print(f"WARNING: {m} lacks predictions for {len(missing)} docs; they count as empty predictions.")

    # verification variants (JSON methods that report evidence)
    for m in [x for x in JSON_METHODS if x in preds]:
        preds[m + "+verify"] = {i: M.verify_records(r, texts[i]) for i, r in preds[m].items()}
    methods = list(preds)

    # ---- main table
    rows, counts = [], {}
    for m in methods:
        for mode in M.MODES:
            counts[(m, mode)] = M.doc_counts(gold_by, preds[m], mode)
        hall, n_pred = M.hallucination_rate(preds[m], texts)
        row = [m, n_pred, f"{parse_rate.get(m.replace('+verify', ''), 1):.2f}", f"{hall:.3f}"]
        for mode in M.MODES:
            p, r, _ = M.prf(counts[(m, mode)])
            f, lo, hi = M.bootstrap_f1(counts[(m, mode)], n_boot)
            row += [f"{p:.3f}", f"{r:.3f}", f"{f:.3f} [{lo:.3f},{hi:.3f}]"]
        rows.append(row)
    header = ["method", "n_pred", "parse_ok", "halluc_rate"]
    for mode in M.MODES:
        header += [f"P_{mode}", f"R_{mode}", f"F1_{mode} [95% CI]"]
    _write_csv(os.path.join(res_dir, "main_results.csv"), header, rows)

    # ---- paired significance (relaxed F1)
    sig = []
    for a, b in itertools.permutations(methods, 2):
        if a < b:
            d, lo, hi, p = M.paired_bootstrap(counts[(a, "relaxed")], counts[(b, "relaxed")])
            sig.append([a, b, f"{d:+.3f}", f"[{lo:+.3f},{hi:+.3f}]", f"{p:.3f}"])
    _write_csv(os.path.join(res_dir, "paired_bootstrap_relaxed_f1.csv"), ["A", "B", "F1(A)-F1(B)", "95% CI", "p"], sig)

    # ---- calibration (verbalised confidence)
    cal_rows, cal_curves = [], {}
    for m in [x for x in ["zero_shot_json", "few_shot_json", "rag_json"] if x in preds]:
        c, ok = M.pred_correctness(gold_by, preds[m])
        if len(c) >= 5:
            cal_curves[m] = M.reliability(c, ok)
            cal_rows.append([m, len(c), f"{M.ece(c, ok):.3f}", f"{M.brier(c, ok):.3f}", f"{np.mean(c):.2f}", f"{np.mean(ok):.2f}"])
    _write_csv(os.path.join(res_dir, "calibration.csv"), ["method", "n", "ECE", "Brier", "mean_conf", "accuracy"], cal_rows)

    # ---- error taxonomy (relaxed)
    tax = {}
    for m in methods:
        cnt = {c: 0 for c in M.FP_CATS + M.FN_CATS}
        for i in gold_by:
            fps, fns = M.categorize_errors(gold_by[i], preds[m].get(i, []), texts[i])
            for c, _ in fps + fns:
                cnt[c] += 1
        tax[m] = cnt
    cats = M.FP_CATS + M.FN_CATS
    _write_csv(os.path.join(res_dir, "error_taxonomy.csv"), ["method"] + cats,
               [[m] + [tax[m][c] for c in cats] for m in methods])

    # ---- figures
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(methods)), 5))
    w = 0.27
    for k, mode in enumerate(M.MODES):
        vals = [M.bootstrap_f1(counts[(m, mode)], n_boot) for m in methods]
        y = np.array([v[0] for v in vals])
        err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
        ax.bar(np.arange(len(methods)) + (k - 1) * w, y, w, yerr=err, capsize=3, label=mode, color=PALETTE[k])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("micro F1 (95% bootstrap CI)")
    ax.set_ylim(0, 1)
    ax.legend(title="match level")
    _save(fig, fig_dir, "fig1_f1_by_method")

    base = [m for m in JSON_METHODS if m in preds]
    if base:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(base))
        before = [M.hallucination_rate(preds[m], texts)[0] for m in base]
        after = [M.hallucination_rate(preds[m + "+verify"], texts)[0] for m in base]
        ax.bar(x - 0.2, before, 0.4, label="no verification", color=PALETTE[3])
        ax.bar(x + 0.2, after, 0.4, label="with span verification", color=PALETTE[2])
        ax.set_xticks(x)
        ax.set_xticklabels(base, rotation=20, ha="right")
        ax.set_ylabel("hallucinated value/unit rate")
        ax.legend()
        _save(fig, fig_dir, "fig2_hallucination_verification")

        fig, ax = plt.subplots(figsize=(6, 5))
        for k, m in enumerate(base):
            for suffix, mk in [("", "o"), ("+verify", "s")]:
                p, r, _ = M.prf(counts[(m + suffix, "relaxed")])
                ax.scatter(r, p, marker=mk, s=90, color=PALETTE[k], label=m + suffix)
        ax.set_xlabel("recall (relaxed)")
        ax.set_ylabel("precision (relaxed)")
        ax.legend(fontsize=8)
        _save(fig, fig_dir, "fig3_precision_recall_verification")

    if cal_curves:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for k, (m, curve) in enumerate(cal_curves.items()):
            ax.plot([c[0] for c in curve], [c[1] for c in curve], marker="o", color=PALETTE[k], label=m)
        ax.set_xlabel("stated confidence")
        ax.set_ylabel("empirical precision")
        ax.legend(fontsize=9)
        _save(fig, fig_dir, "fig4_reliability")

    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(methods)), 5))
    bottom = np.zeros(len(methods))
    for k, c in enumerate(cats):
        v = np.array([tax[m][c] for m in methods], float)
        ax.bar(range(len(methods)), v, bottom=bottom, label=c, color=(PALETTE + ["#999999"])[k % 8])
        bottom += v
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("error count")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, fig_dir, "fig5_error_taxonomy")

    # ---- failure review sheet for manual taxonomy (best non-verify method)
    cand = [m for m in methods if "+verify" not in m and m != "regex_baseline"] or methods
    best = max(cand, key=lambda m: M.prf(counts[(m, "relaxed")])[2])
    fails = []
    for i in gold_by:
        fps, fns = M.categorize_errors(gold_by[i], preds[best].get(i, []), texts[i])
        for kind, lst in [("FP", fps), ("FN", fns)]:
            for cat, r in lst:
                fails.append([best, i, kind, cat, r.get("material"), r.get("property"), r.get("value"),
                              r.get("unit"), texts[i], ""])
    random.Random(seed).shuffle(fails)
    _write_csv(os.path.join(res_dir, "failures_to_review.csv"),
               ["method", "doc_id", "type", "auto_category", "material", "property", "value", "unit", "text",
                "human_category"], fails[:review_n])

    # ---- markdown summary
    md = [f"# Results summary\n\nDocs: {len(gold)}. Best method (relaxed F1, no verification): **{best}**.\n",
          "## Main results\n", _md(header, rows), "\n## Calibration (verbalised confidence)\n",
          _md(["method", "n", "ECE", "Brier", "mean_conf", "accuracy"], cal_rows) if cal_rows else "n/a",
          "\n## Error taxonomy (relaxed match)\n", _md(["method"] + cats, [[m] + [tax[m][c] for c in cats] for m in methods]),
          "\n## Paired bootstrap on relaxed F1\n", _md(["A", "B", "F1(A)-F1(B)", "95% CI", "p"], sig)]
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:4]))
    print(f"\nTables -> {res_dir}\nFigures -> {fig_dir}\nSummary -> {os.path.join(out_dir, 'summary.md')}")
