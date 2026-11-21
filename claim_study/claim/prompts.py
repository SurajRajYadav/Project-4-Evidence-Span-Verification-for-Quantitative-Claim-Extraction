"""Prompt templates and output parsers for the five LLM methods."""
import json
import re

from .utils import autofill_evidence

DOMAIN = "materials science"

SCHEMA = f"""Extract every quantitative property claim from a {DOMAIN} text.
A record has four fields:
- material: the material or system the value belongs to (e.g. "MoS2 monolayer")
- property: the measured or computed property (e.g. "band gap")
- value: the number exactly as written in the text (keep ranges and +/- as written; no unit)
- unit: the unit exactly as written ("" if dimensionless)
Rules:
- Only explicit values stated in the text. Never compute, convert or guess.
- Experimental conditions (temperature, cycle count, doping level, year, citation numbers) are NOT records.
- One record per (material, property, value)."""

JSON_FORMAT = """Return ONLY a JSON list. Each item must be:
{"material": str, "property": str, "value": str, "unit": str, "evidence": str, "confidence": float}
"evidence" must be copied verbatim from the text (the clause or sentence containing the value).
"confidence" is your probability (0 to 1) that the record is correct. If there are no claims return []."""

TEXT_FORMAT = """Return one record per line in the form:
material | property | value | unit
If there are no claims return the single word NONE."""

LORA_FORMAT = """Return ONLY a JSON list of objects with keys material, property, value, unit, evidence. Return [] if there are none."""

METHODS = ["zero_shot_text", "zero_shot_json", "few_shot_json", "rag_json", "lora_json"]
PROMPTED = ["zero_shot_text", "zero_shot_json", "few_shot_json", "rag_json"]
JSON_METHODS = ["zero_shot_json", "few_shot_json", "rag_json", "lora_json"]


def user_msg(text):
    return {"role": "user", "content": f"Text:\n{text}"}


def demo_answer(doc, with_conf=True):
    out = []
    for r in doc["records"]:
        item = {k: r.get(k, "") for k in ["material", "property", "value", "unit"]}
        item["evidence"] = autofill_evidence(doc["text"], r)
        if with_conf:
            item["confidence"] = 0.9
        out.append(item)
    return json.dumps(out, ensure_ascii=False)


def build_messages(method, text, demos=None):
    demos = demos or []
    if method == "zero_shot_text":
        system = SCHEMA + "\n" + TEXT_FORMAT
    elif method == "lora_json":
        system = SCHEMA + "\n" + LORA_FORMAT
    else:
        system = SCHEMA + "\n" + JSON_FORMAT
    msgs = [{"role": "system", "content": system}]
    for d in demos:
        msgs.append(user_msg(d["text"]))
        msgs.append({"role": "assistant", "content": demo_answer(d)})
    msgs.append(user_msg(text))
    return msgs


#  parsers
def _clean(item):
    if not isinstance(item, dict):
        return None
    rec = {k: str(item.get(k, "") if item.get(k) is not None else "").strip()
           for k in ["material", "property", "value", "unit", "evidence"]}
    conf = item.get("confidence")
    try:
        rec["confidence"] = min(1.0, max(0.0, float(conf)))
    except (TypeError, ValueError):
        rec["confidence"] = None
    return rec if rec["value"] else None


def parse_json_records(output):
    """Returns (records, parse_ok)."""
    s = re.sub(r"```(?:json)?", "", output or "").strip()
    a, b = s.find("["), s.rfind("]")
    if a != -1 and b > a:
        try:
            data = json.loads(s[a:b + 1])
            recs = [_clean(x) for x in data]
            return [r for r in recs if r], True
        except json.JSONDecodeError:
            pass
    # salvage individual objects from truncated / malformed output
    recs = []
    for m in re.finditer(r"\{[^{}]*\}", s):
        try:
            r = _clean(json.loads(m.group(0)))
            if r:
                recs.append(r)
        except json.JSONDecodeError:
            continue
    ok = bool(recs) or s.strip() in ("[]", "")
    return recs, ok


def parse_text_records(output):
    out = (output or "").strip()
    if out.upper().startswith("NONE"):
        return [], True
    recs = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.strip().lstrip("-*0123456789. ").split("|")]
        if len(parts) >= 3 and parts[2]:
            recs.append({"material": parts[0], "property": parts[1], "value": parts[2],
                         "unit": parts[3] if len(parts) > 3 else "", "evidence": "",
                         "confidence": None})
    return recs, bool(recs)
