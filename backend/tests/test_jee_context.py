"""End-to-end tests for the JEE Intelligence vertical slice (#32).

Covers: exact match, alias match, ambiguous tie, unknown concept, concept
with no PYQs, concept with verified PYQs, multi-record retrieval, and the
full screenshot -> StudyNotes -> concept -> syllabus -> PYQ -> response path.

Every factual assertion traces to stored corpus data. No credits are touched
(JEE lookup is free metadata).
"""
import asyncio
import io
import json

from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.services.jee_service import map_concept_to_jee

transport = ASGITransport(app=app)

TORQUE_05 = "jee-physics-rotational-motion-concept-005"
ANGMOM_06 = "jee-physics-rotational-motion-concept-006"
GYR_09 = "jee-physics-rotational-motion-concept-009"


def _post_map(client, topic_title, key_terms):
    return client.post(
        "/api/jee/map",
        json={"topic_title": topic_title, "key_terms": key_terms},
    )


async def _map(topic_title, key_terms):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await _post_map(client, topic_title, key_terms)
    assert resp.status_code == 200
    return resp.json()


# 1. Exact concept-name match
def test_exact_concept_match():
    body = asyncio.run(_map("Torque", []))
    assert body["match_status"] == "matched"
    assert body["concept_id"] == TORQUE_05
    assert body["canonical_name"] == "Torque"
    assert body["subject"] == "Physics" and body["chapter"] == "Rotational Motion"


# 2. Alias match (no exact title hit)
def test_alias_match():
    body = asyncio.run(_map("Turning moment basics", []))
    assert body["match_status"] == "matched"
    assert body["concept_id"] == TORQUE_05
    assert any("alias" in e for e in body["evidence"])


# 3. Ambiguous tie refuses to guess
def test_ambiguous_tie_returns_candidates():
    body = asyncio.run(_map("Moment of inertia", ["torque"]))
    assert body["match_status"] == "ambiguous"
    assert body["concept_id"] is None
    assert len(body["candidates"]) >= 2
    assert body["pyqs"] == []  # never attach evidence to a guess


# 4. Unknown concept is explicit
def test_unknown_concept_unresolved():
    body = asyncio.run(_map("Quantum tunneling", ["barrier penetration"]))
    assert body["match_status"] == "unresolved"
    assert body["concept_id"] is None
    assert body["pyqs"] == [] and body["pyq_count"] == 0


# 5. Matched concept with no PYQs yet (honest empty state).
# Data-driven: finds a taxonomy concept with zero verified records so the test
# stays valid as the corpus grows (until every concept has evidence).
def test_concept_with_no_pyqs():
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(repo_root, "docs", "jee-pyq-corpus", "data",
                            "jee-main-physics-rotational-pyq.json"), encoding="utf-8") as fh:
        records = json.load(fh)
    used = {c for r in records for c in r["concept_ids"]}
    with open(os.path.join(repo_root, "docs", "jee-concept-taxonomy", "data",
                            "jee-physics-rotational-motion-taxonomy.json"), encoding="utf-8") as fh:
        concepts = json.load(fh)
    empty = [c for c in concepts if c["concept_id"] not in used]
    assert empty, "every concept has PYQs; retire this test"
    target = empty[0]
    body = asyncio.run(_map(target["display_name"], []))
    assert body["match_status"] == "matched"
    assert body["concept_id"] == target["concept_id"]
    assert body["pyqs"] == [] and body["pyq_count"] == 0


# 6. Matched concept carries verified PYQs with provenance
def test_concept_with_verified_pyqs():
    body = asyncio.run(_map("Torque", ["τ = r × F"]))
    assert body["match_status"] == "matched"
    assert body["concept_id"] == TORQUE_05
    assert body["pyq_count"] >= 1
    ids = {pyq["question_id"] for pyq in body["pyqs"]}
    assert "pyq-jee-main-2026-apr02-s1-phy-029" in ids  # founding verified record
    pyq = next(p for p in body["pyqs"] if p["question_id"] == "pyq-jee-main-2026-apr02-s1-phy-029")
    assert pyq["year"] == 2026 and pyq["shift"] == "Shift 1 (2 Apr 2026)"
    assert pyq["paper_question_number"] == 29 and pyq["marks"] == 4
    assert "cdnbbsr.s3waas.gov.in" in pyq["source_url"]
    assert "frequency" not in pyq and "weightage" not in pyq


# 7. Multi-record retrieval is exact stored facts (chapter slice)
def test_multi_record_chapter_retrieval():
    import sys
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    queries_dir = os.path.join(repo_root, "docs", "jee-pyq-corpus", "queries")
    sys.path.insert(0, queries_dir)
    try:
        from pyq_queries import get_by_chapter
    finally:
        sys.path.remove(queries_dir)
    records = get_by_chapter("Rotational Motion")
    ids = {r["question_id"] for r in records}
    # Founding 2026 records must always be present; corpus only grows.
    assert {
        "pyq-jee-main-2026-apr02-s1-phy-029",
        "pyq-jee-main-2026-apr02-s1-phy-031",
        "pyq-jee-main-2026-apr04-s1-phy-030",
        "pyq-jee-main-2026-apr02-s2-phy-028",
        "pyq-jee-main-2026-apr02-s2-phy-050",
        "pyq-jee-main-2026-apr04-s2-phy-035",
        "pyq-jee-main-2026-apr05-s2-phy-030",
        "pyq-jee-main-2026-apr05-s2-phy-031",
        "pyq-jee-main-2026-apr08-s2-phy-032",
    } <= ids
    assert len(records) == len(ids)  # no duplicate ingestions


# 8. Full path: screenshot-style StudyNotes -> concept -> syllabus -> PYQs
def test_e2e_torque_fixture():
    topic_title = "Torque"
    key_terms = ["τ = r × F", "L = Iω", "position vector", "applied force"]
    ctx = map_concept_to_jee(topic_title, key_terms)
    assert ctx.match_status == "matched"
    assert ctx.concept_id == TORQUE_05
    # syllabus node linked by official subtopic ref
    assert ctx.syllabus is not None
    assert ctx.syllabus.subtopic == "Torque"
    assert "cdnbbsr.s3waas.gov.in" in ctx.syllabus.source_url
    # taxonomy relationships intact
    assert any(p.concept_id == "jee-physics-rotational-motion-concept-004" for p in ctx.prerequisites)
    assert any(r.concept_id == ANGMOM_06 for r in ctx.related)
    # verified exam evidence attached (corpus grows; founding record stays)
    assert ctx.pyq_count >= 1
    assert "pyq-jee-main-2026-apr02-s1-phy-029" in {p.question_id for p in ctx.pyqs}


def test_concept_endpoint_serves_stored_context():
    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(f"/api/jee/concept/{TORQUE_05}")

    body = asyncio.run(run()).json()
    assert body["match_status"] == "matched"
    assert body["canonical_name"] == "Torque"
    assert body["pyq_count"] >= 1


def test_concept_endpoint_unknown_id():
    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/jee/concept/jee-physics-nope-999")

    body = asyncio.run(run()).json()
    assert body["match_status"] == "unresolved"
    assert body["concept_id"] is None


def test_angular_momentum_resolves_with_pyq():
    body = asyncio.run(_map("Angular momentum", []))
    assert body["match_status"] == "matched"
    assert body["concept_id"] == ANGMOM_06
    assert body["pyq_count"] >= 1  # founding Q29 record plus later additions
    assert "pyq-jee-main-2026-apr02-s1-phy-029" in {p["question_id"] for p in body["pyqs"]}


def test_mapping_touches_no_credits():
    from app.utils import credits_store

    def _boom(*_a, **_k):
        raise AssertionError("JEE lookup must not touch credits")

    orig_use, orig_add = credits_store.use_credits, credits_store.add_credits
    credits_store.use_credits, credits_store.add_credits = _boom, _boom
    try:
        asyncio.run(_map("Torque", ["τ"]))
    finally:
        credits_store.use_credits, credits_store.add_credits = orig_use, orig_add
