"""Contract tests for the explicit storage backend switch (upload_image).

imgbb remains the default production backend. data_uri is staging/testing
only: no network, exact bytes preserved, true MIME type from magic bytes.
"""
import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.services import storage_service
from app.services.storage_service import upload_image

transport = ASGITransport(app=app)


def _make_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="JPEG")
    return buf.getvalue()


def _reset_credits(device_id: str, amount: int = 50) -> None:
    from app.utils.credits_store import _get_conn, init_device
    init_device(device_id)
    conn = _get_conn()
    conn.execute(
        "UPDATE device_credits SET credits_remaining = ?, credits_used = 0 WHERE device_id = ?",
        (amount, device_id),
    )
    conn.commit()


# ── data_uri backend: format, MIME, exactness, no network ──


def test_data_uri_png_format_and_mime(monkeypatch):
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    png = _make_png()
    url = upload_image(png, {"title": "explain-visually"})
    assert url is not None and url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == png  # byte-exact round-trip


def test_data_uri_detects_jpeg_not_png(monkeypatch):
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    jpeg = _make_jpeg()
    url = upload_image(jpeg)
    assert url is not None and url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == jpeg


def test_data_uri_never_touches_network(monkeypatch):
    import httpx

    def _boom(*_a, **_k):
        raise AssertionError("data_uri backend must not call the network")

    monkeypatch.setattr(httpx, "post", _boom)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    url = upload_image(_make_png())
    assert url is not None and url.startswith("data:image/png;base64,")


def test_data_uri_rejects_unknown_bytes(monkeypatch):
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    assert upload_image(b"not-an-image") is None  # explicit failure, never mislabeled


def test_data_uri_output_is_deterministic(monkeypatch):
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    png = _make_png()
    assert upload_image(png) == upload_image(png)


def test_unknown_backend_returns_none(monkeypatch):
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "bogus", raising=False)
    assert upload_image(_make_png()) is None


# ── imgbb backend unchanged ──


def test_r2_is_default_backend():
    from app.config import settings
    assert settings.IMAGE_STORAGE_BACKEND == "r2"


def test_imgbb_backend_still_uploads(monkeypatch):
    import httpx

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"success": True, "data": {"url": "https://imgbb.test/kept.png"}}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "imgbb", raising=False)
    monkeypatch.setattr("app.config.settings.IMGBB_API_KEY", "test-key", raising=False)
    assert upload_image(_make_png()) == "https://imgbb.test/kept.png"


# ── routes: diagram + visual over data_uri, credits, dedup ──

_DIAGRAM_PAYLOAD = {
    "topic": {"title": "Torque", "is_probable": False},
    "what_you_should_remember": "Takeaway.",
    "key_formulas": [{"formula": "t = r x F", "explanation": "torque link between vectors", "uncertain_symbols": [], "confidence": "clear"}],
    "understand_it": ["Concept explanation."],
    "common_mistakes": [],
    "thirty_second_revision": [],
    "visual_context": {"present": False, "summary": ""},
    "verify_before_studying": [],
    "uncertainties": [],
    "analogy": "",
}


def _post_diagram(client, device: str, image: bytes):
    return client.post(
        "/api/extract/diagram",
        files={"image": ("d.png", image, "image/png")},
        data={"context": json.dumps({}), "deviceId": device},
    )


def test_diagram_route_returns_data_uri_and_charges_5(sample_diagram_image, monkeypatch):
    from app.utils.credits_store import get_credits

    device = "datauri-diagram-device-00000000-0000-000000000001"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    resp = asyncio.run(run())
    assert resp.status_code == 200
    body = resp.json()
    assert body["creditsUsed"] == 5
    assert body["imageUrl"] is not None and body["imageUrl"].startswith("data:image/jpeg;base64,")
    remaining, _ = get_credits(device)
    assert remaining == 45  # exactly 5 charged


def test_visual_route_data_uri_generated_then_reused(sample_diagram_image, monkeypatch):
    from app.utils.credits_store import get_credits, get_visual_entitlement

    device = "datauri-visual-device-00000000-0000-000000000002"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    png = _make_png()
    render_mock = AsyncMock(return_value=("png", png))
    monkeypatch.setattr("app.routes.extract.build_visual_spec", AsyncMock(return_value=_rich_force_spec()))
    monkeypatch.setattr("app.routes.extract.should_use_v3", lambda spec: True)
    monkeypatch.setattr("app.routes.extract.render_v3_visual", render_mock)

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp1 = asyncio.run(run_visual())
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["status"] == "generated"
    assert body1["imageUrl"] is not None and body1["imageUrl"].startswith("data:image/png;base64,")
    assert base64.b64decode(body1["imageUrl"].split(",", 1)[1]) == png
    assert get_visual_entitlement(diagram_id)[1] == body1["imageUrl"]
    remaining, _ = get_credits(device)
    assert remaining == 45  # visual is bundled/free, no extra charge

    resp2 = asyncio.run(run_visual())
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["status"] == "already_generated"
    assert body2["imageUrl"] == body1["imageUrl"]  # immutable, no second store
    assert render_mock.await_count == 1  # no second generation
    remaining, _ = get_credits(device)
    assert remaining == 45


# ── r2 backend (production default) ──

_R2_PUBLIC = "https://pub-abc123.r2.dev"


def _set_r2_settings(monkeypatch, public_url: str = _R2_PUBLIC):
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "r2", raising=False)
    monkeypatch.setattr("app.config.settings.R2_ACCOUNT_ID", "acct123", raising=False)
    monkeypatch.setattr("app.config.settings.R2_ACCESS_KEY_ID", "AKIDEXAMPLE", raising=False)
    monkeypatch.setattr("app.config.settings.R2_SECRET_ACCESS_KEY", "secret-example", raising=False)
    monkeypatch.setattr("app.config.settings.R2_BUCKET_NAME", "snapnote-diagrams", raising=False)
    monkeypatch.setattr("app.config.settings.R2_PUBLIC_URL", public_url, raising=False)


class _R2Resp:
    def __init__(self, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        return None


def test_r2_upload_success_returns_public_url(monkeypatch):
    import httpx

    captured: dict = {}

    def _put(url, content=None, headers=None, timeout=None):
        captured["url"] = url
        captured["content"] = content
        captured["headers"] = headers
        return _R2Resp(200)

    monkeypatch.setattr(httpx, "put", _put)
    _set_r2_settings(monkeypatch)
    png = _make_png()
    url = upload_image(png, {"title": "explain-visually"})
    assert url is not None and url.startswith(_R2_PUBLIC + "/visuals/")
    assert url.endswith(".png")
    assert captured["url"].startswith("https://acct123.r2.cloudflarestorage.com/snapnote-diagrams/")
    assert captured["content"] == png  # byte integrity
    assert captured["headers"]["Content-Type"] == "image/png"
    assert captured["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert "x-amz-date" in captured["headers"]


def test_r2_object_key_is_content_addressed():
    from app.services.storage_service import _r2_object_key

    png = _make_png()
    key1 = _r2_object_key(png, {"title": "explain-visually"})
    key2 = _r2_object_key(png, {"title": "explain-visually"})
    assert key1 == key2  # deterministic: same bytes, same key
    assert key1.startswith("visuals/") and key1.endswith(".png")
    other = _r2_object_key(_make_jpeg(), {"title": "explain-visually"})
    assert other != key1  # different bytes never collide
    assert other.startswith("visuals/") and other.endswith(".jpg")  # true MIME, not label
    diagram_key = _r2_object_key(_make_jpeg(), None)
    assert diagram_key.startswith("diagrams/")


def test_r2_upload_failure_returns_none(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "put", lambda *a, **k: _R2Resp(403, "forbidden"))
    _set_r2_settings(monkeypatch)
    assert upload_image(_make_png()) is None  # controlled failure, no raise


def test_r2_missing_config_returns_none(monkeypatch):
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "r2", raising=False)
    monkeypatch.setattr("app.config.settings.R2_ACCOUNT_ID", "", raising=False)
    assert upload_image(_make_png()) is None


def test_r2_rejects_placeholder_public_url(monkeypatch):
    from app.services.storage_service import validate_storage_config

    _set_r2_settings(monkeypatch, public_url="https://pub-xxxxx.r2.dev")
    assert upload_image(_make_png()) is None
    with pytest.raises(RuntimeError):
        validate_storage_config()


def test_validate_storage_config_accepts_r2_and_data_uri(monkeypatch):
    from app.services.storage_service import validate_storage_config

    _set_r2_settings(monkeypatch)
    validate_storage_config()  # must not raise
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    validate_storage_config()  # must not raise
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "bogus", raising=False)
    with pytest.raises(RuntimeError):
        validate_storage_config()


def test_validate_allows_empty_account_with_explicit_endpoint(monkeypatch):
    from app.services.storage_service import validate_storage_config

    _set_r2_settings(monkeypatch)
    monkeypatch.setattr("app.config.settings.R2_ACCOUNT_ID", "", raising=False)
    monkeypatch.setattr(
        "app.config.settings.S3_ENDPOINT_URL",
        "https://ref.storage.supabase.co/storage/v1/s3",
        raising=False,
    )
    validate_storage_config()  # Supabase-style config must boot


def test_s3_endpoint_override_uses_path_style(monkeypatch):
    import httpx

    captured: dict = {}

    def _put(url, content=None, headers=None, timeout=None):
        captured["url"] = url
        captured["content"] = content
        captured["headers"] = headers
        captured["auth"] = (headers or {}).get("Authorization", "")
        return _R2Resp(200)

    monkeypatch.setattr(httpx, "put", _put)
    _set_r2_settings(monkeypatch, public_url="https://ref.supabase.co/storage/v1/object/public/snapnote-diagrams")
    monkeypatch.setattr(
        "app.config.settings.S3_ENDPOINT_URL",
        "https://ref.storage.supabase.co/storage/v1/s3",
        raising=False,
    )
    monkeypatch.setattr("app.config.settings.S3_REGION", "ap-south-1", raising=False)
    # No R2 account needed when an explicit endpoint is set.
    monkeypatch.setattr("app.config.settings.R2_ACCOUNT_ID", "", raising=False)
    png = _make_png()
    url = upload_image(png, {"title": "explain-visually"})
    assert url is not None and url.startswith(
        "https://ref.supabase.co/storage/v1/object/public/snapnote-diagrams/visuals/"
    )
    assert captured["url"].startswith(
        "https://ref.storage.supabase.co/storage/v1/s3/snapnote-diagrams/visuals/"
    )
    assert captured["content"] == png
    assert captured["headers"]["Content-Type"] == "image/png"
    assert "ap-south-1" in captured["auth"]
    assert captured["auth"].startswith("AWS4-HMAC-SHA256 ")


def test_visual_route_r2_generated_then_reused(sample_diagram_image, monkeypatch):
    import httpx

    from app.utils.credits_store import get_credits, get_visual_entitlement

    device = "r2-visual-device-00000000-0000-000000000003"
    _reset_credits(device)
    _set_r2_settings(monkeypatch)

    puts = {"n": 0}

    def _put(url, content=None, headers=None, timeout=None):
        puts["n"] += 1
        puts["content"] = content
        return _R2Resp(200)

    monkeypatch.setattr(httpx, "put", _put)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    png = _make_png()
    render_mock = AsyncMock(return_value=("png", png))
    monkeypatch.setattr("app.routes.extract.build_visual_spec", AsyncMock(return_value=_rich_force_spec()))
    monkeypatch.setattr("app.routes.extract.should_use_v3", lambda spec: True)
    monkeypatch.setattr("app.routes.extract.render_v3_visual", render_mock)

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp1 = asyncio.run(run_visual())
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["status"] == "generated"
    assert body1["imageUrl"] is not None and body1["imageUrl"].startswith(_R2_PUBLIC + "/visuals/")
    assert puts["content"] == png  # exact bytes stored
    assert get_visual_entitlement(diagram_id)[1] == body1["imageUrl"]
    remaining, _ = get_credits(device)
    assert remaining == 45  # visual is bundled/free

    resp2 = asyncio.run(run_visual())
    body2 = resp2.json()
    assert body2["status"] == "already_generated"
    assert body2["imageUrl"] == body1["imageUrl"]
    assert render_mock.await_count == 1
    remaining, _ = get_credits(device)
    assert remaining == 45


# ── visual storage failure: honest 502, notes untouched, free retry ──

def test_visual_route_storage_failure_then_retry(sample_diagram_image, monkeypatch):
    """Production hotfix regression: when the upload returns no URL, the
    visual route must answer 502 (never a fake success), charge nothing, and
    let the same entitlement succeed on retry without a second generation."""
    from app.utils.credits_store import get_credits, get_visual_entitlement

    device = "visual-fail-device-00000000-0000-000000000003"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    png = _make_png()
    monkeypatch.setattr("app.routes.extract.build_visual_spec", AsyncMock(return_value=_rich_force_spec()))
    monkeypatch.setattr("app.routes.extract.should_use_v3", lambda spec: False)
    monkeypatch.setattr("app.routes.extract.generate_visual", AsyncMock(return_value=("png", png)))

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    # Failure first: upload returns no URL.
    monkeypatch.setattr("app.routes.extract.upload_image", lambda *a, **k: None)
    resp_fail = asyncio.run(run_visual())
    assert resp_fail.status_code == 502
    assert "Could not store" in resp_fail.json()["error"]
    assert get_visual_entitlement(diagram_id)[1] in (None, "")  # nothing stored, no fake state
    remaining, _ = get_credits(device)
    assert remaining == 45  # diagram charged once; failed visual charged nothing

    # Retry with a working upload: succeeds, still free, stored once.
    monkeypatch.setattr(
        "app.routes.extract.upload_image",
        lambda *a, **k: "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
    )
    resp_ok = asyncio.run(run_visual())
    assert resp_ok.status_code == 200
    assert resp_ok.json()["status"] == "generated"
    assert get_visual_entitlement(diagram_id)[1] is not None
    remaining, _ = get_credits(device)
    assert remaining == 45


# ── thin-spec repair: one bounded second chance before rendering ──

def _thin_generic_spec():
    from app.models.schemas import (
        DeterministicVisual,
        VisualGeneric,
        VisualRenderMode,
        VisualScene,
        VisualSpec,
    )

    return VisualSpec(
        concept="Something",
        render_mode=VisualRenderMode.DETERMINISTIC,
        deterministic=DeterministicVisual(
            title="Something",
            scene=VisualScene(
                scene_kind="generic",
                generic=VisualGeneric(central_label="Something", callouts=["a", "b"]),
            ),
        ),
    )


def _rich_force_spec():
    from app.models.schemas import (
        DeterministicVisual,
        VisualObject,
        VisualRelation,
        VisualRenderMode,
        VisualScene,
        VisualSpec,
        VisualVector,
        ForceDiagram,
    )

    return VisualSpec(
        concept="Torque",
        render_mode=VisualRenderMode.DETERMINISTIC,
        deterministic=DeterministicVisual(
            title="Torque",
            scene=VisualScene(
                scene_kind="force_diagram",
                force=ForceDiagram(
                    object=VisualObject(kind="pivot", label="O"),
                    vectors=[
                        VisualVector(label="r", angle_deg=45, length=1.0),
                        VisualVector(label="F", angle_deg=90, length=1.0, tail="r"),
                    ],
                    relation=VisualRelation(expression="t = r x F"),
                ),
            ),
        ),
    )


def test_visual_route_repairs_thin_spec_once(sample_diagram_image, monkeypatch):
    """A generic-family spec triggers exactly one repair; the repaired rich
    spec is what renders and stores. No extra credits, no second repair."""
    from app.utils.credits_store import get_credits

    device = "visual-repair-device-00000000-0000-000000000004"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=_thin_generic_spec()))
    repair_mock = AsyncMock(return_value=_rich_force_spec())
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)
    monkeypatch.setattr(
        "app.routes.extract.render_v3_visual", AsyncMock(return_value=("svg", "<svg>hero</svg>")))

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 200
    assert resp.json()["status"] == "generated"
    assert repair_mock.await_count == 1  # bounded: exactly one repair call
    remaining, _ = get_credits(device)
    assert remaining == 45  # repair + visual both free


def test_visual_route_skips_repair_for_rich_spec(sample_diagram_image, monkeypatch):
    """A geometry-rich spec renders directly with no repair call."""
    device = "visual-norepair-device-00000000-0000-000000000005"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=_rich_force_spec()))

    async def _boom(*_a, **_k):
        raise AssertionError("repair must not run for rich specs")

    monkeypatch.setattr("app.routes.extract.repair_thin_spec", _boom)
    monkeypatch.setattr(
        "app.routes.extract.render_v3_visual", AsyncMock(return_value=("svg", "<svg>hero</svg>")))

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 200
    assert resp.json()["status"] == "generated"


def test_visual_route_repairs_validation_refusal(sample_diagram_image, monkeypatch):
    """A spec the gates would refuse (paraphrased exact claim) gets one
    bounded reconciliation; the reconciled spec renders. Genuinely broken
    specs still refuse     honestly (covered by the storage-failure test)."""
    from app.models.schemas import VisualTask
    from app.utils.credits_store import get_credits

    device = "visual-gaterepair-device-00000000-0000-000000000006"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    thin_tasked = _rich_force_spec()
    thin_tasked.deterministic.task = VisualTask(
        task_type="vector_arrangement",
        learning_outcome="x",
        required_elements=["torque"],
        required_relationships=[],
        exact_claims=["unicorn levitation field"],
        acceptable_families=["force_diagram"],
        assumptions=[],
        missing=[],
    )
    fixed = _rich_force_spec()
    fixed.deterministic.task = VisualTask(
        task_type="vector_arrangement",
        learning_outcome="x",
        required_elements=["torque"],
        required_relationships=[],
        exact_claims=[],
        acceptable_families=["force_diagram"],
        assumptions=[],
        missing=[],
    )
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=thin_tasked))
    repair_mock = AsyncMock(return_value=fixed)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)
    monkeypatch.setattr(
        "app.routes.extract.render_v3_visual", AsyncMock(return_value=("svg", "<svg>hero</svg>")))

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 200
    assert resp.json()["status"] == "generated"
    assert repair_mock.await_count == 1
    remaining, _ = get_credits(device)
    assert remaining == 45


def test_visual_route_promotes_equations_only_claim(sample_diagram_image, monkeypatch):
    """Trust layer: a claim living only in deterministic.equations renders
    nowhere; the route promotes it into a visible callout deterministically
    instead of paying for a model repair."""
    from app.models.schemas import (
        CompositionCallout,
        LessonComposition,
        VisualEquation,
        VisualTask,
    )
    from app.utils.credits_store import get_credits

    device = "visual-promote-claim-00000000-0000-000000000019"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    spec = _rich_force_spec()
    spec.deterministic.equations = [
        VisualEquation(expression="L = I\u03c9", meaning="angular momentum")
    ]
    spec.deterministic.composition = LessonComposition(
        title="Torque",
        callouts=[CompositionCallout(label="O", value="pivot")],
    )
    spec.deterministic.task = VisualTask(
        task_type="vector_arrangement",
        learning_outcome="x",
        required_elements=[],
        required_relationships=[],
        exact_claims=["L = I\u03c9"],
        acceptable_families=["force_diagram"],
        assumptions=[],
        missing=[],
    )
    monkeypatch.setattr("app.routes.extract.build_visual_spec", AsyncMock(return_value=spec))
    repair_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)
    monkeypatch.setattr(
        "app.routes.extract.render_v3_visual", AsyncMock(return_value=("svg", "<svg>hero</svg>")))

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 200
    assert resp.json()["status"] == "generated"
    assert repair_mock.await_count == 0
    assert spec.deterministic.composition is not None
    assert any(
        c.label == "L = I\u03c9" for c in spec.deterministic.composition.callouts
    )
    remaining, _ = get_credits(device)
    assert remaining == 45


def test_visual_route_derives_missing_task(sample_diagram_image, monkeypatch):
    """A task-less spec gets a derived contract attached before rendering —
    the legacy bypass is closed, deterministically, with no extra LLM call."""
    from app.utils.credits_store import get_credits

    device = "visual-derivetask-device-00000000-0000-000000000007"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    taskless = _rich_force_spec()
    taskless.deterministic.task = None
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=taskless))

    async def _boom(*_a, **_k):
        raise AssertionError("no repair needed: derivation is free")

    monkeypatch.setattr("app.routes.extract.repair_thin_spec", _boom)

    seen: dict = {}

    async def capture_render(spec):
        seen["task"] = spec.deterministic.task
        return ("svg", "<svg>hero</svg>")

    monkeypatch.setattr("app.routes.extract.generate_visual", capture_render)

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 200
    assert resp.json()["status"] == "generated"
    assert seen["task"] is not None
    assert seen["task"].task_type == "vector_arrangement"
    assert any("derived" in a for a in seen["task"].assumptions)
    remaining, _ = get_credits(device)
    assert remaining == 45

def test_visual_derivation_failure_refuses_honestly(sample_diagram_image, monkeypatch):
    """A derivation crash must refuse, never silently render unvalidated."""
    device = "visual-derivfail-00000000-0000-000000000008"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    taskless = _rich_force_spec()
    taskless.deterministic.task = None
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=taskless))
    monkeypatch.setattr(
        "app.routes.extract.derive_task_from_source",
        MagicMock(side_effect=RuntimeError("derivation exploded")))

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image)},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["error"]


def test_visual_validation_exception_refuses(sample_diagram_image, monkeypatch):
    """A validator crash counts as failure, not as an empty error list."""
    device = "visual-valfail-device-00000000-0000-000000000009"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=_rich_force_spec()))

    def _boom(*_a, **_k):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr("app.routes.extract.validate_legacy_visual", _boom)
    monkeypatch.setattr(
        "app.routes.extract.repair_thin_spec", AsyncMock(return_value=None))

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image)},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["error"]


def test_visual_missing_source_requirement_refuses(sample_diagram_image, monkeypatch):
    """Sparse-but-consistent scene missing a source requirement refuses;
    notes stay intact (diagram response already delivered them)."""
    from app.models.schemas import VisualTask

    device = "visual-missingreq-device-00000000-0000-000000000010"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, sample_diagram_image)

    diagram_id = asyncio.run(run_diagram()).json()["diagramId"]

    thin = _thin_generic_spec()
    thin.deterministic.task = VisualTask(
        task_type="definition_overview",
        learning_outcome="x",
        required_elements=["unicorn levitation field"],
        required_relationships=[],
        exact_claims=[],
        acceptable_families=["generic"],
        assumptions=[],
        missing=[],
    )
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=thin))
    monkeypatch.setattr(
        "app.routes.extract.repair_thin_spec", AsyncMock(return_value=None))

    async def run_visual():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", sample_diagram_image)},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    resp = asyncio.run(run_visual())
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["error"]


# ── worked mathematics (number_line): verified or refused, never wrong ──

_WORKED_MATH_PROBLEM = "log_{x+3}(x^2 - x) < 1"
_CORRECT_SOLUTION = [
    ("-3", "-2", False, False),
    ("-1", "0", False, False),
    ("1", "3", False, False),
]


def _number_line_spec(solution_intervals, problem=_WORKED_MATH_PROBLEM):
    from app.models.schemas import (
        DeterministicVisual,
        VisualInterval,
        VisualMathCase,
        VisualNumberLine,
        VisualNumberLineTrack,
        VisualScene,
        VisualSpec,
        VisualTask,
    )

    def track(intervals, points=None, label=""):
        return VisualNumberLineTrack(
            label=label,
            intervals=[
                VisualInterval(lo=lo, hi=hi, lo_closed=lo_c, hi_closed=hi_c)
                for lo, hi, lo_c, hi_c in intervals
            ],
            points=points or [],
        )

    nl = VisualNumberLine(
        problem=problem,
        domain=track(
            [
                ("-3", "-2", False, False),
                ("-2", "0", False, False),
                ("1", "inf", False, False),
            ],
            points=["-2"],
        ),
        cases=[
            VisualMathCase(
                condition="x + 3 > 1",
                transformed="(x^2 - x) < x + 3",
                solution="(-1, 0) \u222a (1, 3)",
            ),
            VisualMathCase(
                condition="0 < x + 3 < 1",
                transformed="(x^2 - x) > x + 3",
                solution="(-3, -2)",
            ),
        ],
        solution=track(solution_intervals, label="Final answer"),
    )
    task = VisualTask(
        task_type="interval_solution",
        learning_outcome="solve a logarithmic inequality with a variable base",
        required_elements=["domain", "final answer"],
        required_relationships=[],
        exact_claims=[],
        acceptable_families=["number_line"],
        assumptions=[],
        missing=[],
    )
    return VisualSpec(
        concept="logarithmic inequality with a variable base",
        render_mode="deterministic",
        text_required=True,
        deterministic=DeterministicVisual(
            title="Log Inequality \u2014 Full Case Analysis",
            task=task,
            scene=VisualScene(
                scene_kind="number_line",
                title="Log Inequality \u2014 Full Case Analysis",
                caption=(
                    "A logarithmic inequality with a variable base must satisfy "
                    "the domain first; the base regime decides the direction."
                ),
                number_line=nl,
            ),
        ),
    )


def _run_worked_math_visual(device, diagram_id, image):
    async def run():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/api/extract/visual",
                files={"image": ("v.png", image, "image/png")},
                data={"deviceId": device, "diagramId": diagram_id},
            )

    return asyncio.run(run())


def _make_diagram(device, image, monkeypatch):
    mock_model = AsyncMock()
    mock_model.generate_content_async = AsyncMock(
        return_value=type("o", (), {"text": json.dumps(_DIAGRAM_PAYLOAD)})()
    )
    monkeypatch.setattr("app.services.vision_service.model", mock_model)

    async def run_diagram():
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await _post_diagram(client, device, image)

    return asyncio.run(run_diagram()).json()["diagramId"]


def test_visual_route_refuses_wrong_math_solution(sample_diagram_image, monkeypatch):
    """The exact CEO regression: a spec claiming (-3, -2) ∪ (-1, 3) is refused
    with the precise correction, never rendered."""
    device = "visual-wrongmath-00000000-0000-000000000011"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)

    wrong = _number_line_spec([("-3", "-2", False, False), ("-1", "3", False, False)])
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=wrong))
    repair_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["error"]
    # Two bounded attempts: the second still gets the reason and may not pass.
    assert repair_mock.await_count == 2
    reason = repair_mock.call_args.args[3]
    assert "mathematically wrong" in reason
    assert "(-3, -2) \u222a (-1, 0) \u222a (1, 3)" in reason
    from app.utils.credits_store import get_visual_entitlement

    assert get_visual_entitlement(diagram_id)[1] in (None, "")  # nothing stored


def test_visual_route_repairs_wrong_math_and_renders_verified(
    sample_diagram_image, monkeypatch
):
    """One bounded repair is allowed to fix the math; the repaired spec is
    verified again and the REAL renderer then produces the verified SVG."""
    device = "visual-fixmath-00000000-0000-000000000012"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)

    wrong = _number_line_spec([("-3", "-2", False, False), ("-1", "3", False, False)])
    corrected = _number_line_spec(_CORRECT_SOLUTION)
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=wrong))
    repair_mock = AsyncMock(return_value=corrected)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "generated"
    assert body["renderMode"] == "deterministic"
    svg = body["imageSvg"]
    assert "FINAL ANSWER \u2014 MATHEMATICALLY VERIFIED" in svg
    assert "x \u2208 (-1, 0) \u222a (1, 3)" in svg
    # The repaired 3-interval final track renders 6 open endpoint dots; the
    # wrong 2-interval track would render 4. Geometry proves which track won.
    assert svg.count('fill="white" stroke="#16a34a"') == 6
    assert repair_mock.await_count == 1
    from app.utils.credits_store import get_credits

    remaining, _ = get_credits(device)
    assert remaining == 45  # diagram charged once; explain visually is free


def test_visual_route_refuses_unverifiable_math(sample_diagram_image, monkeypatch):
    """Mathematics outside the supported class refuses honestly: no visual,
    precise reason handed to repair, study notes stay intact."""
    device = "visual-unverifiable-00000000-0000-000000000013"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)

    unsupported = _number_line_spec(
        [("-inf", "inf", False, False)], problem="x^2 - 2 < 0"
    )
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=unsupported))
    repair_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["error"]
    reason = repair_mock.call_args.args[3]
    assert "cannot verify the mathematics" in reason
    assert "never displayed unverified" in reason


def test_visual_route_refuses_incomplete_math_stores_nothing(
    sample_diagram_image, monkeypatch
):
    """Completeness failures (missing domain track, empty case list) refuse
    end-to-end with a precise repair reason, and NO visual result is stored —
    the entitlement row stays empty, so the gate can never leave a false
    'verified' artifact behind."""
    from app.utils.credits_store import get_visual_entitlement

    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)

    # Scenario 1 — missing domain track on a variable-base log.
    device = "visual-incomplete-00000000-0000-000000000014"
    _reset_credits(device)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)
    no_domain = _number_line_spec(_CORRECT_SOLUTION)
    scene = no_domain.deterministic.scene
    assert scene is not None and scene.number_line is not None
    scene.number_line.domain = None
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=no_domain))
    repair_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["error"]
    reason = repair_mock.call_args.args[3]
    assert "domain track is required" in reason
    assert get_visual_entitlement(diagram_id)[1] in (None, "")
    assert get_visual_entitlement(diagram_id)[3] in (None, "")

    # Scenario 2 — entire case list missing on a variable-base log.
    device2 = "visual-incomplete-00000000-0000-000000000015"
    _reset_credits(device2)
    diagram_id2 = _make_diagram(device2, sample_diagram_image, monkeypatch)
    no_cases = _number_line_spec(_CORRECT_SOLUTION)
    scene2 = no_cases.deterministic.scene
    assert scene2 is not None and scene2.number_line is not None
    scene2.number_line.cases = []
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=no_cases))
    repair_mock2 = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock2)

    resp2 = _run_worked_math_visual(device2, diagram_id2, sample_diagram_image)
    assert resp2.status_code == 502
    reason2 = repair_mock2.call_args.args[3]
    assert "expected 2 case branch(es)" in reason2
    assert get_visual_entitlement(diagram_id2)[1] in (None, "")
    assert get_visual_entitlement(diagram_id2)[3] in (None, "")


def test_visual_route_repaired_spec_is_reverified(sample_diagram_image, monkeypatch):
    """A repair that returns ANOTHER wrong spec must be refused: the retry
    never bypasses the gate, stores nothing, and costs no extra credits."""
    from app.utils.credits_store import get_credits, get_visual_entitlement

    device = "visual-reverify-00000000-0000-000000000016"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)

    wrong = _number_line_spec([("-3", "-2", False, False), ("-1", "3", False, False)])
    still_wrong = _number_line_spec(
        [("-3", "-2", False, False), ("-1", "3", False, False)]
    )
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=wrong)
    )
    repair_mock = AsyncMock(return_value=still_wrong)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["error"]
    # Both bounded attempts returned a still-wrong spec; the gate held.
    assert repair_mock.await_count == 2
    assert get_visual_entitlement(diagram_id)[1] in (None, "")
    assert get_visual_entitlement(diagram_id)[3] in (None, "")
    remaining, _ = get_credits(device)
    assert remaining == 45  # diagram charged once; the refusal costs nothing


def test_visual_route_second_repair_attempt_recovers(
    sample_diagram_image, monkeypatch
):
    """Reliability milestone: when the first repair returns a still-invalid
    spec, the SECOND attempt receives the fresh revalidation errors and may
    recover — the visual renders verified instead of a false refusal."""
    device = "visual-repair2-00000000-0000-000000000018"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)

    wrong = _number_line_spec([("-3", "-2", False, False), ("-1", "3", False, False)])
    still_wrong = _number_line_spec([("-3", "-2", False, False), ("-1", "3", False, False)])
    corrected = _number_line_spec(_CORRECT_SOLUTION)
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=wrong)
    )
    repair_mock = AsyncMock(side_effect=[still_wrong, corrected])
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "generated"
    assert repair_mock.await_count == 2
    # The second attempt was fed the fresh validation errors as its reason.
    second_reason = repair_mock.call_args.args[3]
    assert "mathematically wrong" in second_reason


def test_visual_route_rate_limited_spec_reports_high_demand(
    sample_diagram_image, monkeypatch
):
    """A Gemini 429 during specification is upstream load, not a property of
    the student's material: the error must say high demand, never 'the concept
    may not translate'."""
    from app.exceptions import UpstreamError
    from app.utils.credits_store import get_credits, get_visual_entitlement

    device = "visual-ratelimit-spec-0000000000000019"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)

    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec",
        AsyncMock(
            side_effect=UpstreamError(
                service="Gemini Vision",
                detail="429 RESOURCE_EXHAUSTED You exceeded your current quota",
            )
        ),
    )
    repair_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert "high demand" in error
    assert "may not translate" not in error
    assert repair_mock.await_count == 0  # no pointless repair calls
    assert get_visual_entitlement(diagram_id)[1] in (None, "")
    remaining, _ = get_credits(device)
    assert remaining == 45  # diagram charged once; the refusal costs nothing


def test_visual_route_rate_limited_repairs_report_high_demand(
    sample_diagram_image, monkeypatch
):
    """When BOTH bounded repairs fail on 429 (no spec ever came back), the
    refusal is upstream load; when any repair did return a spec that failed
    validation, the honest material-specific refusal stands."""
    from app.exceptions import UpstreamError
    from app.utils.credits_store import get_visual_entitlement

    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    rate_limited = UpstreamError(
        service="Gemini Vision",
        detail="429 RESOURCE_EXHAUSTED You exceeded your current quota",
    )
    wrong = _number_line_spec([("-3", "-2", False, False), ("-1", "3", False, False)])

    # Scenario A — validation fails, both repair calls are rate limited.
    device = "visual-ratelimit-fix-0000000000000020"
    _reset_credits(device)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=wrong)
    )
    repair_mock = AsyncMock(side_effect=rate_limited)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert "high demand" in error
    assert "may not translate" not in error
    assert repair_mock.await_count == 2
    assert get_visual_entitlement(diagram_id)[1] in (None, "")

    # Scenario B — one repair 429, the other returns a still-invalid spec:
    # a real validation signal exists, so the material refusal is truthful.
    device2 = "visual-ratelimit-mix-0000000000000021"
    _reset_credits(device2)
    diagram_id2 = _make_diagram(device2, sample_diagram_image, monkeypatch)
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=wrong)
    )
    repair_mock2 = AsyncMock(side_effect=[rate_limited, wrong])
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock2)

    resp2 = _run_worked_math_visual(device2, diagram_id2, sample_diagram_image)
    assert resp2.status_code == 502
    error2 = resp2.json()["error"]
    assert "unavailable" in error2
    assert "high demand" not in error2
    assert repair_mock2.await_count == 2


def test_visual_route_security_payload_refused(sample_diagram_image, monkeypatch):
    """Executable-looking text smuggled through the math field is parsed by the
    safe grammar, refused as unverifiable, and never stored."""
    from app.utils.credits_store import get_visual_entitlement

    device = "visual-secpayload-00000000-0000-000000000017"
    _reset_credits(device)
    monkeypatch.setattr("app.config.settings.IMAGE_STORAGE_BACKEND", "data_uri", raising=False)
    diagram_id = _make_diagram(device, sample_diagram_image, monkeypatch)

    malicious = _number_line_spec(
        [("-inf", "inf", False, False)],
        problem="__import__('os').system('echo pwned') < 1",
    )
    monkeypatch.setattr(
        "app.routes.extract.build_visual_spec", AsyncMock(return_value=malicious)
    )
    repair_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routes.extract.repair_thin_spec", repair_mock)

    resp = _run_worked_math_visual(device, diagram_id, sample_diagram_image)
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["error"]
    reason = repair_mock.call_args.args[3]
    assert "cannot verify the mathematics" in reason
    assert "never displayed unverified" in reason
    assert get_visual_entitlement(diagram_id)[1] in (None, "")
    assert get_visual_entitlement(diagram_id)[3] in (None, "")
