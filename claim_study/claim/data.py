"""Data: arXiv snippet collection, annotation sheets, gold-file building, demo data."""
import csv
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .utils import autofill_evidence, read_jsonl, value_in_text, unit_in_text, write_jsonl

ARXIV_API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom"}

UNITS = ["eV", "meV", "GPa", "MPa", "kPa", "Pa", "K", "°C", "nm", "μm", "µm", "um", "Å",
         "mm", "cm", "%", "W/mK", "W/m K", "W m-1 K-1", "S/cm", "S/m", "mS/cm", "mAh/g", "V", "mV", "A", "mA",
         "Hz", "GHz", "THz", "kHz", "MHz", "J", "kJ", "T", "Oe", "emu/g", "ps", "ns", "fs", "g/cm3", "g/cm³",
         "cm2/Vs", "cm²/Vs", "cm2/V s", "F/g", "μC/cm2", "µC/cm2", "W/cm2", "mW/cm2", "Wh/kg",
         "mol%", "wt%", "at%", "ppm", "GW", "MW", "kW", "W"]
_U = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
NUM_UNIT_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:\s?±\s?\d+(?:\.\d+)?)?[\s  ]?(?:" + _U + r")(?![A-Za-z])")


def fetch_arxiv(category="cond-mat.mtrl-sci", n=150, seed=42, pool_factor=4, pause=3.0, min_numeric=2):
    """Collect abstracts (arXiv metadata is CC0) that contain >= min_numeric number+unit mentions."""
    rng = random.Random(seed)
    got, seen, tries = [], set(), 0
    while len(got) < n * pool_factor and tries < 30:
        tries += 1
        q = urllib.parse.urlencode({"search_query": f"cat:{category}", "start": rng.randint(0, 20000),
                                    "max_results": 100, "sortBy": "submittedDate", "sortOrder": "descending"})
        req = urllib.request.Request(f"{ARXIV_API}?{q}", headers={"User-Agent": "sci-claim-extraction/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                root = ET.fromstring(r.read())
        except Exception as e:  # network error, arXiv rate limit, bad XML
            if got:
                print(f"arXiv request failed ({e}); using the {len(got)} abstracts collected so far")
                break
            raise RuntimeError(f"Could not reach the arXiv API ({e}). Retry in a minute, or put your own "
                               "abstracts in data/snippets.jsonl (one JSON object per line with keys "
                               "id, title, text).") from e
        for e in root.findall("a:entry", _NS):
            aid = e.find("a:id", _NS).text.strip().split("/abs/")[-1]
            text = re.sub(r"\s+", " ", e.find("a:summary", _NS).text).strip()
            if aid in seen or len(NUM_UNIT_RE.findall(text)) < min_numeric or not (300 <= len(text) <= 1800):
                continue
            seen.add(aid)
            title = re.sub(r"\s+", " ", e.find("a:title", _NS).text).strip()
            got.append({"id": aid, "title": title, "text": text, "source": f"arXiv:{aid}"})
        time.sleep(pause)
    rng.shuffle(got)
    return got[:n]


# ---------------------------------------------------------- annotation I/O
SHEET_COLS = ["doc_id", "text", "material", "property", "value", "unit", "evidence", "notes"]


def write_sheet(path, docs, drafts=None):
    """One row per record; docs without records get one blank row.
    `drafts` (optional) = {doc_id: [records]} used to pre-fill (annotator must correct them)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLS)
        w.writeheader()
        for d in docs:
            recs = (drafts or {}).get(d["id"]) or [{}]
            for k, r in enumerate(recs):
                w.writerow({"doc_id": d["id"], "text": d["text"] if k == 0 else "",
                            **{c: r.get(c, "") for c in ["material", "property", "value", "unit", "evidence"]}})


def sheet_to_gold(sheet_path, snippets_path, annotator="A"):
    snippets = {d["id"]: d for d in read_jsonl(snippets_path)}
    by_doc = {}
    with open(sheet_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            did = (row.get("doc_id") or "").strip()
            if not did:
                continue
            by_doc.setdefault(did, [])
            if (row.get("value") or "").strip():
                by_doc[did].append({c: (row.get(c) or "").strip()
                                    for c in ["material", "property", "value", "unit", "evidence"]})
    gold = []
    for did, recs in by_doc.items():
        if did not in snippets:
            print(f"WARNING: doc_id {did} not in snippets, skipped")
            continue
        d = snippets[did]
        for r in recs:
            r["evidence"] = autofill_evidence(d["text"], r)
        gold.append({"id": did, "title": d.get("title", ""), "text": d["text"], "records": recs,
                     "annotator": annotator, "source": d.get("source", "")})
    return gold


def validate_gold(gold):
    issues = []
    for d in gold:
        for r in d["records"]:
            tag = f"{d['id']} [{r.get('material')} | {r.get('property')} | {r.get('value')}]"
            if not r.get("material") or not r.get("property"):
                issues.append(f"{tag}: missing material/property")
            if not value_in_text(r["value"], d["text"]):
                issues.append(f"{tag}: value not found in text")
            if not unit_in_text(r.get("unit", ""), d["text"]):
                issues.append(f"{tag}: unit not found in text")
    return issues


# ------------------------------------------------------------- demo data
_DEMO = [
    ("We synthesized monolayer MoS2 by chemical vapor deposition and measured a direct optical band gap of 1.85 eV at room temperature. The carrier mobility reached 45 cm2/Vs in back-gated transistors.",
     [("MoS2 monolayer", "optical band gap", "1.85", "eV"), ("MoS2 monolayer", "carrier mobility", "45", "cm2/Vs")]),
    ("Doping SnSe with 2% Na lifts the figure of merit ZT to 1.8 at 773 K, while the lattice thermal conductivity drops to 0.45 W/mK.",
     [("Na-doped SnSe", "figure of merit ZT", "1.8", ""), ("Na-doped SnSe", "lattice thermal conductivity", "0.45", "W/mK")]),
    ("Li-rich layered oxide cathodes deliver a discharge capacity of 280 mAh/g at 0.1 C and retain 92% of capacity after 200 cycles.",
     [("Li-rich layered oxide cathode", "discharge capacity", "280", "mAh/g"), ("Li-rich layered oxide cathode", "capacity retention", "92", "%")]),
    ("The yield strength of the CoCrNi medium-entropy alloy increases from 450 MPa to 1.2 GPa after severe plastic deformation, with elongation maintained at 18%.",
     [("CoCrNi alloy", "yield strength", "450", "MPa"), ("CoCrNi alloy", "yield strength", "1.2", "GPa"), ("CoCrNi alloy", "elongation", "18", "%")]),
    ("Perovskite solar cells based on FA0.9Cs0.1PbI3 achieved a power conversion efficiency of 24.6% with an open-circuit voltage of 1.18 V.",
     [("FA0.9Cs0.1PbI3 solar cell", "power conversion efficiency", "24.6", "%"), ("FA0.9Cs0.1PbI3 solar cell", "open-circuit voltage", "1.18", "V")]),
    ("Graphene was first isolated in 2004. In this review we discuss transport phenomena in two-dimensional materials and open challenges for scalable production.",
     []),
    ("DFT calculations predict that the CsPbBr3 perovskite has a lattice constant of 5.87 Å and a bulk modulus of 18.5 GPa.",
     [("CsPbBr3", "lattice constant", "5.87", "Å"), ("CsPbBr3", "bulk modulus", "18.5", "GPa")]),
    ("The BiFeO3 thin film exhibits a remanent polarization of 55 μC/cm2 and a Curie temperature of 1103 K.",
     [("BiFeO3 thin film", "remanent polarization", "55", "μC/cm2"), ("BiFeO3 thin film", "Curie temperature", "1103", "K")]),
    ("A MXene Ti3C2Tx electrode showed a specific capacitance of 450 F/g and an electrical conductivity of 6500 S/cm.",
     [("Ti3C2Tx MXene electrode", "specific capacitance", "450", "F/g"), ("Ti3C2Tx MXene electrode", "electrical conductivity", "6500", "S/cm")]),
    ("ZnO nanowires with diameters of 80 nm display a thermal conductivity of 35 W/mK at 300 K, about half the bulk value.",
     [("ZnO nanowires", "diameter", "80", "nm"), ("ZnO nanowires", "thermal conductivity", "35", "W/mK")]),
    ("Annealing at 800 °C increases the grain size of TiO2 films from 20 nm to 65 nm and narrows the band gap from 3.2 eV to 3.0 eV.",
     [("TiO2 film", "grain size", "20", "nm"), ("TiO2 film", "grain size", "65", "nm"), ("TiO2 film", "band gap", "3.2", "eV"), ("TiO2 film", "band gap", "3.0", "eV")]),
    ("Using machine-learned potentials we screened 12,000 candidate electrolytes and identified Li7P3S11 with an ionic conductivity of 17 mS/cm.",
     [("Li7P3S11", "ionic conductivity", "17", "mS/cm")]),
]


def demo_gold():
    """12 SYNTHETIC snippets with invented numbers. Only for smoke-testing the pipeline."""
    docs = []
    for k, (text, recs) in enumerate(_DEMO):
        rr = [dict(zip(["material", "property", "value", "unit"], r)) for r in recs]
        for r in rr:
            r["evidence"] = autofill_evidence(text, r)
        docs.append({"id": f"demo{k:02d}", "title": "synthetic demo", "text": text, "records": rr,
                     "annotator": "demo", "source": "synthetic"})
    return docs


def load_gold(path):
    return demo_gold() if path == "demo" else read_jsonl(path)
