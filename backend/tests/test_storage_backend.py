"""Contract tests for the explicit storage backend switch (upload_image).

imgbb remains the default production backend. data_uri is staging/testing
only: no network, exact bytes preserved, true MIME type from magic bytes.
"""
import asyncio
import base64
import io
import json
from unittest.mock import AsyncMock

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
    "topic": {"title": "Rotation", "is_probable": False},
    "what_you_should_remember": "Takeaway.",
    "key_formulas": [{"formula": "τ = r × F", "explanation": "", "uncertain_symbols": [], "confidence": "clear"}],
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
    monkeypatch.setattr("app.routes.extract.build_visual_spec", AsyncMock(return_value=object()))
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
    monkeypatch.setattr("app.routes.extract.build_visual_spec", AsyncMock(return_value=object()))
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
