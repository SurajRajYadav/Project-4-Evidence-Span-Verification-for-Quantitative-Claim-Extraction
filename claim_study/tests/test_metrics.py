import numpy as np

from claim import metrics as M
from claim.data import demo_gold, validate_gold
from claim.prompts import parse_json_records, parse_text_records
from claim.utils import value_in_text, unit_in_text


def test_demo_gold_is_valid():
    assert validate_gold(demo_gold()) == []


def test_value_grounding():
    assert value_in_text("1.5", "gap of 1.5 eV")
    assert not value_in_text("1.5", "gap of 11.5 eV")
    assert not value_in_text("5", "gap of 1.5 eV")
    assert value_in_text("300-400", "from 300 – 400 K")
    assert value_in_text("1000", "about 1,000 K")
    assert unit_in_text("eV", "1.5 eV") and not unit_in_text("K", "bulk modulus")
    assert unit_in_text("°C", "at 800 °C")


def test_matching_levels():
    g = [{"material": "MoS2 monolayer", "property": "band gap", "value": "1.85", "unit": "eV"}]
    p = [{"material": "MoS2", "property": "optical band gap", "value": "1.850", "unit": "ev"}]
    assert M.match_doc(g, p, "strict")[1] == 0
    assert M.match_doc(g, p, "relaxed")[1] == 1
    assert M.match_doc(g, [{"material": "", "property": "", "value": "1.85", "unit": "eV"}], "value_unit")[1] == 1


def test_prf_and_bootstrap():
    c = np.array([[2, 0, 0], [1, 1, 1], [0, 0, 2]], float)
    p, r, f = M.prf(c)
    assert abs(p - 3 / 4) < 1e-9 and abs(r - 3 / 6) < 1e-9
    f1, lo, hi = M.bootstrap_f1(c, 500)
    assert lo <= f1 <= hi
    d, lo, hi, pv = M.paired_bootstrap(c, c)
    assert d == 0 and 0 <= pv <= 1


def test_verification_and_hallucination():
    text = "The band gap is 1.85 eV at room temperature."
    good = {"value": "1.85", "unit": "eV", "evidence": "The band gap is 1.85 eV", "material": "x", "property": "y"}
    bad = {"value": "2.1", "unit": "eV", "evidence": "The band gap is 2.1 eV", "material": "x", "property": "y"}
    assert M.verify_records([good, bad], text) == [good]
    rate, n = M.hallucination_rate({"d": [good, bad]}, {"d": text})
    assert n == 2 and rate == 0.5


def test_kappa():
    assert M.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert M.cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0


def test_agreement_identical():
    g = demo_gold()
    res = M.agreement(g, g)
    assert res["kappa_number_level"] == 1.0 and res["f1_A_vs_B_strict"] == 1.0


def test_parsers():
    r, ok = parse_json_records('```json\n[{"material":"A","property":"p","value":"1","unit":"K","evidence":"e","confidence":0.7}]\n```')
    assert ok and r[0]["confidence"] == 0.7
    r, ok = parse_json_records('[{"material":"A","property":"p","value":"1","unit":"K"}, {"material":"B"')
    assert len(r) == 1
    r, ok = parse_text_records("MoS2 | band gap | 1.85 | eV\nNONE-ish line")
    assert r[0]["value"] == "1.85"
    assert parse_text_records("NONE") == ([], True)
