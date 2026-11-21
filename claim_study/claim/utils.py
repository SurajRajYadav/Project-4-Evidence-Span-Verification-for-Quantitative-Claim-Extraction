"""Shared helpers: IO, seeding, and text/value/unit normalisation."""
import json
import os
import random
import re
import unicodedata

import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_DASHES = "−–—‒"


def base_norm(s) -> str:
    """NFKC, lowercase, unify dashes and the micro sign."""
    s = unicodedata.normalize("NFKC", "" if s is None else str(s)).lower()
    for d in _DASHES:
        s = s.replace(d, "-")
    return s.replace("μ", "u")


def norm_text(s) -> str:
    return re.sub(r"\s+", " ", base_norm(s)).strip()


_SEP = re.compile(r"\s*(±|-|×|\^)\s*")


def norm_value(v) -> str:
    s = base_norm(v).replace(",", "").replace("~", "").replace("≈", "")
    s = _SEP.sub(r"\1", s)
    return re.sub(r"\s+", "", s).lstrip("+")


def _unit_compact(s) -> str:
    s = base_norm(s)
    for ch in ["°", "·", "⋅", "*", "^"]:
        s = s.replace(ch, "")
    return re.sub(r"\s+", "", s)


def norm_unit(u) -> str:
    return _unit_compact(u)


def try_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def value_equal(a, b) -> bool:
    na, nb = norm_value(a), norm_value(b)
    if na == nb:
        return True
    fa, fb = try_float(na), try_float(nb)
    return fa is not None and fb is not None and abs(fa - fb) <= 1e-9 * max(1.0, abs(fa))


def value_in_text(value, text) -> bool:
    """Is the value present in the text as a whole number token?"""
    v = norm_value(value)
    if not v:
        return False
    t = _SEP.sub(r"\1", base_norm(text).replace(",", ""))
    return re.search(r"(?<![\d.])" + re.escape(v) + r"(?!\d)", t) is not None


def unit_in_text(unit, text) -> bool:
    u = _unit_compact(unit)
    if not u:
        return True
    t = base_norm(text)
    for ch in ["°", "·", "⋅", "*", "^"]:
        t = t.replace(ch, "")
    pat = r"\s?".join(re.escape(c) for c in u)  # tolerate spaces inside compound units
    return re.search(r"(?<![a-z])" + pat + r"(?![a-z])", t) is not None


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text.strip()) if s]


def autofill_evidence(text, rec) -> str:
    """Pick the sentence in `text` that contains the record's value."""
    if rec.get("evidence"):
        return rec["evidence"]
    for s in split_sentences(text):
        if value_in_text(rec.get("value", ""), s):
            return s
    return ""
