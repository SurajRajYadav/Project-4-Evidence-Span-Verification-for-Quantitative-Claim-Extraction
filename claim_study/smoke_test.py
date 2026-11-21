"""CPU-only check that the code is wired up (no model download, ~10 s). Run before any real run."""
import os
import shutil
import tempfile

from claim import metrics as M
from claim.analysis import analyze
from claim.data import demo_gold, validate_gold
from claim.experiment import run_experiment
from claim.prompts import parse_json_records, parse_text_records
from claim.utils import value_in_text

gold = demo_gold()
assert validate_gold(gold) == [], "demo gold failed validation"
print(f"dataset OK: {len(gold)} synthetic docs, {sum(len(d['records']) for d in gold)} records")

assert value_in_text("1.5", "gap of 1.5 eV") and not value_in_text("1.5", "gap of 11.5 eV")
recs, ok = parse_json_records('[{"material":"A","property":"p","value":"1","unit":"K","evidence":"e"}]')
assert ok and len(recs) == 1 and parse_text_records("NONE") == ([], True)
print("parsers OK")

c = M.doc_counts({d["id"]: d["records"] for d in gold}, {d["id"]: d["records"] for d in gold}, "strict")
assert M.prf(c)[2] == 1.0
assert M.agreement(gold, gold)["kappa_number_level"] == 1.0
print("metrics OK (perfect predictions give F1 = 1.0, kappa = 1.0)")

tmp = tempfile.mkdtemp()
try:
    run_experiment("demo", tmp, mock=True, folds=3)
    analyze("demo", tmp, n_boot=200)
    for f in ["summary.md", "tables/main_results.csv", "figures/fig1_f1_by_method.png"]:
        assert os.path.exists(os.path.join(tmp, f)), f"missing output {f}"
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("\nmock end-to-end run + analysis OK")
print("\nSMOKE TEST PASSED (GPU model code is exercised by the first real run; see README)")
