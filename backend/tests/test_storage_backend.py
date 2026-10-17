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
