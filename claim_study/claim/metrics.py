"""Matching, P/R/F1, bootstrap CIs, hallucination, calibration, agreement, error taxonomy."""
import re

import numpy as np

from .utils import (base_norm, norm_text, norm_unit, norm_value, value_equal,
                    value_in_text, unit_in_text)

MODES = ["strict", "relaxed", "value_unit"]


# matching
def _soft(s):
    return re.sub(r"[^a-z0-9]+", " ", base_norm(s)).strip()


def soft_match(a, b) -> bool:
    a, b = _soft(a), _soft(b)
    if a == b:
        return True
    if not a or not b:
        return False
    ca, cb = a.replace(" ", ""), b.replace(" ", "")
    if min(len(ca), len(cb)) >= 3 and (ca in cb or cb in ca):
        return True
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(1, len(A | B)) >= 0.5


def records_match(g, p, mode) -> bool:
    if not (value_equal(g.get("value", ""), p.get("value", ""))
            and norm_unit(g.get("unit", "")) == norm_unit(p.get("unit", ""))):
        return False
    if mode == "value_unit":
        return True
    if mode == "relaxed":
        return (soft_match(g.get("material", ""), p.get("material", ""))
                and soft_match(g.get("property", ""), p.get("property", "")))
    return (norm_text(g.get("material", "")) == norm_text(p.get("material", ""))
            and norm_text(g.get("property", "")) == norm_text(p.get("property", "")))


def dedupe(recs):
    seen, out = set(), []
    for r in recs:
        k = (norm_text(r.get("material", "")), norm_text(r.get("property", "")),
             norm_value(r.get("value", "")), norm_unit(r.get("unit", "")))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def match_doc(gold, pred, mode):
    """Greedy one-to-one matching. Returns ({pred_idx: gold_idx}, tp, fp, fn)."""
    used, pair = set(), {}
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j not in used and records_match(g, p, mode):
                used.add(j)
                pair[i] = j
                break
    tp = len(pair)
    return pair, tp, len(pred) - tp, len(gold) - tp


def doc_counts(gold_by_id, pred_by_id, mode):
    ids = sorted(gold_by_id)
    rows = []
    for i in ids:
        _, tp, fp, fn = match_doc(gold_by_id[i], dedupe(pred_by_id.get(i, [])), mode)
        rows.append((tp, fp, fn))
    return np.array(rows, dtype=float).reshape(-1, 3)


def prf(counts):
    tp, fp, fn = counts.sum(axis=0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _f1_from_sums(s):
    tp, fp, fn = s[..., 0], s[..., 1], s[..., 2]
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def bootstrap_f1(counts, n_boot=2000, seed=0):
    """Document-level bootstrap. Returns (F1, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = len(counts)
    idx = rng.integers(0, n, size=(n_boot, n))
    f = _f1_from_sums(counts[idx].sum(axis=1))
    return prf(counts)[2], float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))


def paired_bootstrap(ca, cb, n_boot=5000, seed=0):
    """F1(a) - F1(b) with 95% CI and two-sided bootstrap p-value (same documents)."""
    rng = np.random.default_rng(seed)
    n = len(ca)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = _f1_from_sums(ca[idx].sum(axis=1)) - _f1_from_sums(cb[idx].sum(axis=1))
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return prf(ca)[2] - prf(cb)[2], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), min(1.0, float(p))


#  hallucination
def grounding(rec, text):
    return {"value": value_in_text(rec.get("value", ""), text),
            "unit": unit_in_text(rec.get("unit", ""), text),
            "material": (not rec.get("material")) or any(
                t in _soft(text) for t in _soft(rec["material"]).split() if len(t) >= 3)}


def hallucination_rate(pred_by_id, texts):
    """Share of predicted records whose value or unit does not occur in the source."""
    n = bad = 0
    for i, recs in pred_by_id.items():
        for r in dedupe(recs):
            g = grounding(r, texts[i])
            n += 1
            bad += (not g["value"]) or (not g["unit"])
    return bad / n if n else 0.0, n


def verify_records(recs, text):
    """Keep a record only if its evidence is a verbatim span of the source AND contains the value."""
    t = norm_text(text)
    out = []
    for r in recs:
        ev = norm_text(r.get("evidence", ""))
        if ev and ev in t and value_in_text(r.get("value", ""), ev):
            out.append(r)
    return out


#  calibration
def reliability(confs, correct, bins=5):
    confs, correct = np.asarray(confs, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for b in range(bins):
        m = (confs >= edges[b]) & ((confs < edges[b + 1]) if b < bins - 1 else (confs <= 1.0))
        if m.any():
            rows.append((confs[m].mean(), correct[m].mean(), int(m.sum())))
    return rows


def ece(confs, correct, bins=5):
    rows = reliability(confs, correct, bins)
    n = sum(r[2] for r in rows)
    return sum(abs(c - a) * k for c, a, k in rows) / n if n else float("nan")


def brier(confs, correct):
    confs, correct = np.asarray(confs, float), np.asarray(correct, float)
    return float(np.mean((confs - correct) ** 2)) if len(confs) else float("nan")


def pred_correctness(gold_by_id, pred_by_id, mode="relaxed"):
    """Per-prediction (confidence, is_correct) pairs for records that report a confidence."""
    confs, ok = [], []
    for i, g in gold_by_id.items():
        preds = dedupe(pred_by_id.get(i, []))
        pair, *_ = match_doc(g, preds, mode)
        for k, p in enumerate(preds):
            if p.get("confidence") is not None:
                confs.append(p["confidence"])
                ok.append(1.0 if k in pair else 0.0)
    return confs, ok


#  agreement
def cohen_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def candidate_numbers(text):
    return sorted(set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", base_norm(text).replace(",", ""))))


def agreement(docs_a, docs_b):
    """Inter-annotator agreement on the docs both annotators labelled.

    kappa: for every number in the text, did the annotator treat it as part of a claim (1/0)?
    This includes true negatives (years, citation numbers...) so kappa is well defined.
    """
    A, B = {d["id"]: d for d in docs_a}, {d["id"]: d for d in docs_b}
    shared = sorted(set(A) & set(B))
    la, lb = [], []
    for i in shared:
        va = " ".join(r["value"] for r in A[i]["records"])
        vb = " ".join(r["value"] for r in B[i]["records"])
        for num in candidate_numbers(A[i]["text"]):
            la.append(int(value_in_text(num, va)))
            lb.append(int(value_in_text(num, vb)))
    ga = {i: A[i]["records"] for i in shared}
    gb = {i: B[i]["records"] for i in shared}
    res = {"n_docs": len(shared), "n_numbers": len(la), "kappa_number_level": cohen_kappa(la, lb)}
    for m in MODES:
        res[f"f1_A_vs_B_{m}"] = prf(doc_counts(ga, gb, m))[2] if shared else float("nan")
    return res


#  error taxonomy
FP_CATS = ["hallucinated_value", "hallucinated_unit", "wrong_unit", "wrong_material",
           "wrong_property", "spurious_claim"]
FN_CATS = ["missed_claim", "field_error"]


def categorize_errors(gold, pred, text):
    """Relaxed-mode error categories for one document: returns (fp_list, fn_list)."""
    pred = dedupe(pred)
    pair, *_ = match_doc(gold, pred, "relaxed")
    matched_gold = set(pair.values())
    fps, fns = [], []
    for i, p in enumerate(pred):
        if i in pair:
            continue
        if not value_in_text(p.get("value", ""), text):
            cat = "hallucinated_value"
        elif not unit_in_text(p.get("unit", ""), text):
            cat = "hallucinated_unit"
        else:
            same = [g for g in gold if value_equal(g.get("value", ""), p.get("value", ""))]
            if not same:
                cat = "spurious_claim"
            elif not any(norm_unit(g.get("unit", "")) == norm_unit(p.get("unit", "")) for g in same):
                cat = "wrong_unit"
            elif not any(soft_match(g.get("material", ""), p.get("material", "")) for g in same):
                cat = "wrong_material"
            else:
                cat = "wrong_property"
        fps.append((cat, p))
    for j, g in enumerate(gold):
        if j in matched_gold:
            continue
        hit = any(value_equal(g.get("value", ""), p.get("value", "")) for p in pred)
        fns.append(("field_error" if hit else "missed_claim", g))
    return fps, fns
