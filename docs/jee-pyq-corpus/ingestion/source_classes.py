"""Source provenance classes for the corpus factory.

Every ingested record resolves to exactly one class. The audit enforces the
labeling; nothing may be promoted without evidence.
"""
from enum import Enum


class SourceClass(str, Enum):
    OFFICIAL = "official"  # NTA/official body document, directly retrieved.
    THIRD_PARTY_TRANSCRIPTION = "third-party-transcription"  # transcribed from
    # official material by a third party; never equivalent to OFFICIAL.
    DISCOVERY_ONLY = "discovery-only"  # external datasets/mocks; quarantined,
    # never a source of exam facts.


def classify_record(record: dict) -> SourceClass:
    url = (record.get("source_url") or "")
    doc = ((record.get("source_document") or "") + " " +
           (record.get("verification_method") or "")).lower()
    if "cdnbbsr.s3waas.gov.in" in url or "nta.ac.in" in url or "jeemain.nta" in url:
        return SourceClass.OFFICIAL
    if "third-party transcription" in doc:
        return SourceClass.THIRD_PARTY_TRANSCRIPTION
    return SourceClass.DISCOVERY_ONLY


def label_is_honest(record: dict) -> bool:
    """A record is honest iff its class matches its labeling."""
    cls = classify_record(record)
    doc = ((record.get("source_document") or "") + " " +
           (record.get("verification_method") or ""))
    if cls == SourceClass.OFFICIAL:
        return "third-party transcription" not in doc.lower()
    if cls == SourceClass.THIRD_PARTY_TRANSCRIPTION:
        return "third-party transcription" in doc.lower()
    return True
