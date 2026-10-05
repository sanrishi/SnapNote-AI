"""Phase 3: deterministic full-corpus audit. Reports findings; fixes nothing
silently — prints FIXABLE items for explicit follow-up. Exit 1 on failure."""
import json
import os
import re
import sys

CORPUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(CORPUS, "data", "jee-main-physics-rotational-pyq.json")
REG = os.path.join(CORPUS, "sources", "registry.json")
TAX = os.path.join(CORPUS, "..", "jee-concept-taxonomy", "data",
                   "jee-physics-rotational-motion-taxonomy.json")

records = json.load(open(DATA, encoding="utf-8"))
registry = {s["source_id"]: s for s in json.load(open(REG, encoding="utf-8"))["sources"]}
tax_ids = {c["concept_id"] for c in json.load(open(TAX, encoding="utf-8"))}

# Numerical records whose official answer was never recovered keep null, but
# only with an explicit documented reason in the verification method.
NUMERICAL_NULL_EXCEPTIONS = {
    "pyq-jee-main-2026-apr02-s2-phy-050":
        "official numeric answer not recovered from the NTA paper; left null",
}

fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        fails.append(msg)


ids = [r["question_id"] for r in records]
check(len(ids) == len(set(ids)), f"duplicate question_ids: {[i for i in ids if ids.count(i) > 1]}")
fps = [r.get("content_fingerprint") for r in records]
check(len(fps) == len(set(fps)), "duplicate content_fingerprints")
check(all(f for f in fps), "empty content_fingerprint present")

REQUIRED = ["question_id", "exam", "year", "subject", "chapter", "concept_ids",
            "question_type", "paper_question_number", "nta_question_id",
            "source_url", "source_document", "verification_status", "verification_method"]
for r in records:
    for f in REQUIRED:
        check(f in r, f"{r.get('question_id')}: missing field {f}")
    for c in r.get("concept_ids", []):
        check(c in tax_ids, f"{r['question_id']}: unknown concept {c}")
    check(r.get("verification_status") == "verified",
          f"{r['question_id']}: status {r.get('verification_status')}")
    check(r.get("difficulty") is None, f"{r['question_id']}: difficulty must stay null")
    # Answer-format discipline.
    a = r.get("answer_key")
    if r.get("question_type") == "Numerical":
        if isinstance(a, str) and re.fullmatch(r"-?\d+(\.\d+)?", a) is not None:
            pass
        elif (a is None and r["question_id"] in NUMERICAL_NULL_EXCEPTIONS
                and NUMERICAL_NULL_EXCEPTIONS[r["question_id"]].split(";")[0]
                in (r.get("verification_method") or "")):
            pass
        else:
            fails.append(f"{r['question_id']}: numerical answer_key malformed: {a!r}")
    else:
        check(a is None or (isinstance(a, str) and "MR" in a.replace(" ", "")) or
              (isinstance(a, str) and len(a) <= 16),
              f"{r['question_id']}: MCQ answer_key suspicious: {a!r}")
    # Provenance honesty: third-party records must never claim NTA-PDF origin.
    doc = (r.get("source_document") or "") + " " + (r.get("verification_method") or "")
    if "jeeprep" in (r.get("source_url") or ""):
        check("third-party transcription" in doc,
              f"{r['question_id']}: jeeprep record missing transcription label")
        check("official NTA paper PDF (page rendering inspected)" not in doc,
              f"{r['question_id']}: jeeprep record claims NTA-PDF origin")
    else:
        check("cdnbbsr.s3waas.gov.in" in (r.get("source_url") or ""),
              f"{r['question_id']}: non-jeeprep record without NTA URL")
    # Shift/session coherence with registry paper identity.
    sid = None
    m = re.search(r"/pyq/([\w-]+)/physics", r.get("source_url") or "")
    if m:
        y = r["year"]
        sid = f"jeeprep-main-{y}-{m.group(1)}"
        src = registry.get(sid)
        check(src is not None, f"{r['question_id']}: registry source {sid} missing")
        if src:
            check(src["shift"] == r["shift"], f"{r['question_id']}: shift mismatch vs registry")
            check(src["year"] == r["year"], f"{r['question_id']}: year mismatch vs registry")

print(f"audited {len(records)} records")
if fails:
    print(f"FAILURES ({len(fails)}):")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("audit clean")
