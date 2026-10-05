"""Phase 1: deterministic pre-screen of the 50-entry rotational review queue.

Reads the LOCAL transcribed index (gitignored raw stems) + public queue.
Writes:
  - <corpus>/.cache/prescreen.local.json  (full findings incl. short evidence)
  - index/prescreen_report.json            (public: counts, classes, reasons;
                                            no question stems)

Never marks anything VERIFIED.
"""
import collections
import json
import os
import re

CORPUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_INDEX = os.path.join(CORPUS, ".cache", "jeeprep")
QUEUE_PATH = os.path.join(CORPUS, "index", "verification_queue.json")
DATA_PATH = os.path.join(CORPUS, "data", "jee-main-physics-rotational-pyq.json")

STRONG = [
    "torque", "moment of inertia", "angular momentum", "angular velocity",
    "angular acceleration", "rolling", "centre of mass", "center of mass",
    "rigid body", "gyration", "rotational", "rotating disc", "rotating ring",
]
WEAK = ["rotate", "rotation", "angular", "spin", "disc", "cylinder", "sphere",
        "balance", "equilibrium", "pivot", "hinge", "lever", "moment of"]
# Weak tokens that are usually NOT rotational mechanics when dominant:
FALSE_POSITIVE_CONTEXT = [
    "angular frequency", "shm", "simple harmonic", "oscillation",
    "wave", "electron spin", "spin quantum", "optics", "polariz",
    "electromagnetic", "alternating current",
]
CORRUPTION_MARKERS = [
    "[Option lost", "[Stem truncated", "\ufffd", "�",
]
TRUNCATION_HINTS = [
    "the remainder of this question is lost",
    "surviving text ends mid-sentence",
]


def _load_local_index() -> dict:
    path = os.path.join(CORPUS, "index", "transcribed_index.local.json")
    if os.path.exists(path):
        rows = json.load(open(path, encoding="utf-8"))
    else:
        rows = []
        for root, _, files in os.walk(LOCAL_INDEX):
            for f in files:
                if f.endswith(".json"):
                    rows.extend(json.load(open(os.path.join(root, f), encoding="utf-8")))
    return {r["normalized_text_hash"]: r for r in rows}


def integrity_flags(stem: str) -> list[str]:
    flags = []
    low = stem.lower()
    # Split off the mirror's figure-description tail: glyph losses there are
    # figure-template noise, not stem corruption.
    body, _, fig_tail = low.partition("figure — described")
    if not fig_tail:
        body, _, fig_tail = low.partition("figure - described")
    for m in CORRUPTION_MARKERS:
        if m == "�":
            if m in body:
                flags.append("has_corruption_marker")
            elif m in fig_tail:
                flags.append("figure_glyph_loss")
        elif m.lower() in low:
            flags.append("has_corruption_marker")
            break
    for h in TRUNCATION_HINTS:
        if h in low:
            flags.append("truncated_stem")
            break
    if len(stem) < 80:
        flags.append("stem_too_short")
    if stem.count("$") % 2 == 1:
        flags.append("unbalanced_math_delimiter")
    if len(re.findall(r"[a-zA-Z]{3,}", stem)) < 8:
        flags.append("too_few_words")
    # Runs of blanks/ellipsis (numerical blanks "____", "...") are legitimate.
    if re.search(r"(.)\1{5,}", re.sub(r"[_\-.=*\s]+", "", stem)):
        flags.append("repeated_char_run")
    # Ends mid-formula/sentence without terminal punctuation or option block.
    # Option-stripped stems legitimately end on noun phrases ("its length",
    # figure descriptions); only dangling connectives signal real truncation.
    DANGLING = {"with", "of", "and", "or", "the", "a", "an", "to", "in", "on",
                "by", "for", "is", "are", "be", "than", "as", "at", "from",
                "that", "which", "whose", "where", "when", "if", "then", "so",
                "but", "plus", "minus", "times", "equals", "about", "along",
                "per", "via", "between", "among", "into", "onto", "upon",
                "following", "given"}
    tail = stem.strip()[-60:].lower()
    tail200 = stem.strip()[-200:].lower()
    last_word = re.sub(r"[^a-z ]", " ", stem.strip().lower()).split()
    last_word = last_word[-1] if last_word else ""
    if (not re.search(r"[.?:…_]$", stem.strip())
            and "options" not in low
            and "figure" not in tail
            and "figure" not in tail200
            and last_word in DANGLING):
        flags.append("abrupt_ending")
    return sorted(set(flags))


def concept_flags(stem: str) -> list[str]:
    low = stem.lower()
    strong = [t for t in STRONG if t in low]
    weak = [t for t in WEAK if t in low]
    fp_ctx = [t for t in FALSE_POSITIVE_CONTEXT if t in low]
    flags = []
    if strong:
        flags.append("strong_rotational_signal:" + ",".join(sorted(set(strong))[:3]))
    if weak and not strong:
        flags.append("weak_only:" + ",".join(sorted(set(weak))[:3]))
    if not strong and not weak:
        flags.append("no_rotational_signal")
    if fp_ctx and not strong:
        flags.append("possible_false_positive_context")
    return flags


def main() -> int:
    by_hash = _load_local_index()
    queue = json.load(open(QUEUE_PATH, encoding="utf-8"))["entries"]
    verified = json.load(open(DATA_PATH, encoding="utf-8"))
    verified_identities = {
        (r.get("year"), str(r.get("session")), str(r.get("shift")), r.get("question_number"))
        for r in verified
    }

    seen_hashes: dict[str, list[str]] = collections.defaultdict(list)
    findings = []
    for entry in queue:
        qid = entry["queue_id"]
        rec = by_hash.get(entry["normalized_text_hash"])
        item = {"queue_id": qid, "source_id": entry["source_id"],
                "question_number": entry["question_number"]}
        if rec is None:
            item["flags"] = ["index_record_missing"]
            item["stem_chars"] = 0
            findings.append(item)
            continue
        stem = rec.get("stem_text", "")
        item["stem_chars"] = len(stem)
        flags = integrity_flags(stem)
        flags.extend(concept_flags(stem))
        # Metadata completeness.
        for field in ("exam", "year", "session", "shift", "subject", "source_url"):
            if not rec.get(field):
                flags.append(f"missing_{field}")
        # Duplicate within queue.
        seen_hashes[rec["normalized_text_hash"]].append(qid)
        # Duplicate against verified corpus by paper identity.
        ident = (rec.get("year"), str(rec.get("session")), str(rec.get("shift")),
                 rec.get("question_number"))
        if ident in verified_identities:
            flags.append("identity_already_verified")
        # Provenance quality.
        if not rec.get("source_url"):
            flags.append("missing_source_url")
        if rec.get("answer_key_status") != "claimed-official-final-answer-key":
            flags.append("answer_key_unclaimed")
        item["flags"] = sorted(set(flags))
        item["topic_tag"] = rec.get("topic")
        item["difficulty_tag"] = rec.get("difficulty")
        findings.append(item)

    for h, qids in seen_hashes.items():
        if len(qids) > 1:
            for f in findings:
                if f["queue_id"] in qids:
                    f["flags"] = sorted(set(f["flags"]) | {"duplicate_within_queue"})

    # Phase-2 classification (deterministic rules; human confirms A before verify).
    classes = collections.Counter()
    for f in findings:
        fl = set(f["flags"])
        if "duplicate_within_queue" in fl or "identity_already_verified" in fl:
            f["class"] = "D_DUPLICATE"
        elif {"has_corruption_marker", "truncated_stem", "stem_too_short",
              "too_few_words", "index_record_missing"} & fl:
            f["class"] = "C_CORRUPTED_REJECT"
        elif ({"weak_only", "possible_false_positive_context", "no_rotational_signal",
               "abrupt_ending", "unbalanced_math_delimiter", "figure_glyph_loss",
               "repeated_char_run"} & {x.split(":")[0] for x in fl}) or \
                (not any(x.startswith("strong_rotational_signal") for x in fl)):
            f["class"] = "B_NEEDS_SOURCE_REVIEW"
        else:
            f["class"] = "A_HIGH_CONFIDENCE"
        classes[f["class"]] += 1

    os.makedirs(os.path.join(CORPUS, ".cache"), exist_ok=True)
    with open(os.path.join(CORPUS, ".cache", "prescreen.local.json"), "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=1)

    public = {
        "total": len(findings),
        "classes": dict(classes),
        "entries": [
            {"queue_id": f["queue_id"], "source_id": f["source_id"],
             "question_number": f.get("question_number"), "class": f["class"],
             "flags": [x for x in f["flags"] if not x.startswith("strong_rotational_signal:")],
             "topic_tag": f.get("topic_tag")}
            for f in findings
        ],
        "note": "Phase 1 pre-screen only. Nothing is VERIFIED. Flags are machine "
                "signals; class A still needs human visual read before verification.",
    }
    with open(os.path.join(CORPUS, "index", "prescreen_report.json"), "w", encoding="utf-8") as fh:
        json.dump(public, fh, indent=1)
    print(json.dumps({"total": len(findings), "classes": dict(classes)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
