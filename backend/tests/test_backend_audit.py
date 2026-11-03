"""Regression tests for backend defects found in the production audit.

Each test names the defect it locks down:
- framework 404/405 and request-validation errors must use {"error", "code"}
- a placeholder JWT secret must never sign production sessions
- mock payments must never mint credits in production, and verify is idempotent
- prose must not be classified as a one-column table
- decompression bombs are refused before decoding
- the Chrome extension exchange must issue a real SnapNote session token
- CORS wildcard must not be combined with credentials
"""
import asyncio
import io
import json
import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import UploadFile

from app.config import DEV_JWT_SECRET, Settings, is_weak_jwt_secret
from app.main import app

transport = ASGITransport(app=app)


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _set_device_credits(device_id: str, amount: int) -> None:
    from app.utils.credits_store import _get_conn, init_device

    init_device(device_id)
    conn = _get_conn()
    conn.execute(
        "UPDATE device_credits SET credits_remaining = ?, credits_used = 0 WHERE device_id = ?",
        (amount, device_id),
    )
    conn.commit()


def _device_credits(device_id: str) -> int:
    from app.utils.credits_store import get_credits

    return get_credits(device_id)[0]


# ── Error envelope ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_route_uses_error_envelope():
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/definitely-not-a-route")
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == 404
    assert body["error"]
    assert "detail" not in body


@pytest.mark.asyncio
async def test_wrong_method_uses_error_envelope():
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/extract/text")
    assert resp.status_code == 405
    body = resp.json()
    assert body["code"] == 405
    assert "detail" not in body


@pytest.mark.asyncio
async def test_request_validation_uses_error_envelope():
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/extract/text", data={})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == 422
    assert "image" in body["error"] or "deviceId" in body["error"]
    assert "detail" not in body


# ── JWT secret hardening ────────────────────────────────────────────────────


def test_weak_secret_detection():
    assert is_weak_jwt_secret(DEV_JWT_SECRET)
    assert is_weak_jwt_secret("change-this-to-a-random-32-char-string-in-production")
    assert is_weak_jwt_secret("short")
    assert not is_weak_jwt_secret("s" * 48)


def test_production_never_signs_with_placeholder_secret():
    generated = Settings(
        _env_file=None,
        JWT_SECRET=DEV_JWT_SECRET,
        DEBUG=False,
        CREDITS_DB_PATH="credits.db",
    )
    assert generated.JWT_SECRET != DEV_JWT_SECRET
    assert generated.JWT_SECRET_GENERATED is True
    # Relative DB path: no secret file is written into the working tree.
    assert generated.JWT_SECRET_PERSISTED is False
    assert generated.JWT_SECRET_STORAGE == "ephemeral"
    assert generated.JWT_SECRET_STORAGE_REASON == "relative_path"

    strong = Settings(_env_file=None, JWT_SECRET="s" * 48, DEBUG=False)
    assert strong.JWT_SECRET == "s" * 48
    assert strong.JWT_SECRET_GENERATED is False
    assert strong.JWT_SECRET_STORAGE == "environment"

    # Local DEBUG keeps the documented default for a familiar dev workflow.
    debug = Settings(_env_file=None, JWT_SECRET=DEV_JWT_SECRET, DEBUG=True)
    assert debug.JWT_SECRET == DEV_JWT_SECRET
    assert debug.JWT_SECRET_STORAGE == "development"


def test_render_relative_default_upgrades_to_mounted_disk(tmp_path, monkeypatch):
    from app import config as config_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config_module, "_RENDER_DATA_DIR", data_dir)
    monkeypatch.setattr(config_module, "_render_disk_available", lambda: True)

    strong = config_module.Settings(
        _env_file=None,
        JWT_SECRET="s" * 48,
        DEBUG=False,
        CREDITS_DB_PATH="credits.db",
    )
    assert strong.CREDITS_DB_PATH == str(data_dir / "credits.db")
    assert strong.CREDITS_DB_SOURCE == "render_default"

    generated = config_module.Settings(
        _env_file=None,
        JWT_SECRET=DEV_JWT_SECRET,
        DEBUG=False,
        CREDITS_DB_PATH="credits.db",
    )
    assert generated.JWT_SECRET_PERSISTED is True
    assert generated.JWT_SECRET_STORAGE == "persisted"
    assert generated.JWT_SECRET_STORAGE_REASON == ""
    assert (data_dir / ".jwt_secret").exists()


def test_generated_secret_persists_beside_mounted_database(tmp_path):
    db_path = str(tmp_path / "credits.db")
    first = Settings(
        _env_file=None,
        JWT_SECRET=DEV_JWT_SECRET,
        DEBUG=False,
        CREDITS_DB_PATH=db_path,
    )
    assert first.JWT_SECRET_PERSISTED is True
    secret_file = tmp_path / ".jwt_secret"
    assert secret_file.exists()

    second = Settings(
        _env_file=None,
        JWT_SECRET=DEV_JWT_SECRET,
        DEBUG=False,
        CREDITS_DB_PATH=db_path,
    )
    assert second.JWT_SECRET == first.JWT_SECRET
    assert second.JWT_SECRET_PERSISTED is True


def test_placeholder_signed_token_is_rejected_by_generated_secret():
    import jwt

    forged = jwt.encode({"sub": "attacker"}, DEV_JWT_SECRET, algorithm="HS256")
    production = Settings(
        _env_file=None,
        JWT_SECRET=DEV_JWT_SECRET,
        DEBUG=False,
        CREDITS_DB_PATH="credits.db",
    )
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(forged, production.JWT_SECRET, algorithms=["HS256"])


# ── Payments ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_order_refuses_unconfigured_gateway_outside_debug(monkeypatch):
    from app.routes import payments

    monkeypatch.setattr(payments, "MOCK_MODE", False)
    monkeypatch.setattr(payments.settings, "RAZORPAY_KEY_ID", "")
    monkeypatch.setattr(payments.settings, "RAZORPAY_KEY_SECRET", "")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/payments/create-order",
            json={"deviceId": _uid("no-gateway"), "plan": "starter"},
        )
    assert resp.status_code == 503
    assert resp.json()["code"] == 503


@pytest.mark.asyncio
async def test_mock_verify_rejects_non_mock_order_ids(monkeypatch):
    from app.routes import payments

    monkeypatch.setattr(payments, "MOCK_MODE", True)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/payments/verify",
            json={
                "razorpay_order_id": "order_real_gateway_id",
                "razorpay_payment_id": "pay_fake",
                "razorpay_signature": "fake",
                "deviceId": _uid("mock-verify"),
            },
        )
    assert resp.status_code == 422
    assert "unknown order" in resp.json()["error"].lower()


@pytest.mark.asyncio
async def test_verify_credits_exactly_once_and_only_owning_device(monkeypatch):
    from app.routes import payments

    monkeypatch.setattr(payments, "MOCK_MODE", True)
    owner = _uid("order-owner")
    stranger = _uid("order-stranger")
    _set_device_credits(owner, 10)
    _set_device_credits(stranger, 10)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/payments/create-order",
            json={"deviceId": owner, "plan": "starter"},
        )
        assert created.status_code == 200
        order = created.json()
        assert order["mock"] is True
        assert order["credits"] == 50

        stolen = await client.post(
            "/api/payments/verify",
            json={
                "razorpay_order_id": order["orderId"],
                "razorpay_payment_id": "mock_pay_stolen",
                "razorpay_signature": "irrelevant",
                "deviceId": stranger,
            },
        )
        assert stolen.status_code == 422
        assert _device_credits(stranger) == 10
        assert _device_credits(owner) == 10

        first = await client.post(
            "/api/payments/verify",
            json={
                "razorpay_order_id": order["orderId"],
                "razorpay_payment_id": "mock_pay_ok",
                "razorpay_signature": "irrelevant",
                "deviceId": owner,
            },
        )
        assert first.status_code == 200
        assert first.json()["creditsAdded"] == 50
        assert _device_credits(owner) == 60

        replay = await client.post(
            "/api/payments/verify",
            json={
                "razorpay_order_id": order["orderId"],
                "razorpay_payment_id": "mock_pay_ok",
                "razorpay_signature": "irrelevant",
                "deviceId": owner,
            },
        )
        assert replay.status_code == 200
        assert replay.json()["creditsAdded"] == 0
        assert _device_credits(owner) == 60


@pytest.mark.asyncio
async def test_reset_endpoint_restores_allowance(monkeypatch):
    from app.routes import payments

    monkeypatch.setattr(payments.settings, "DEBUG", True)
    device = _uid("reset-device")
    _set_device_credits(device, 3)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(f"/api/payments/reset?deviceId={device}")
    assert resp.status_code == 200
    assert resp.json()["creditsRemaining"] == payments.settings.FREE_CREDITS_MONTHLY
    assert _device_credits(device) == payments.settings.FREE_CREDITS_MONTHLY


class _FakeRazorpayResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _FakeRazorpayClient:
    """Stand-in for httpx.AsyncClient used by the order-creation call."""

    last_request: dict = {}
    response = _FakeRazorpayResponse(200, {"id": "order_live_123"})

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json=None, auth=None):
        _FakeRazorpayClient.last_request = {
            "url": url,
            "json": json,
            "auth": auth,
        }
        return _FakeRazorpayClient.response


@pytest.mark.asyncio
async def test_configured_gateway_creates_order_without_sdk(monkeypatch):
    from app.routes import payments

    monkeypatch.setattr(payments, "MOCK_MODE", False)
    monkeypatch.setattr(payments.settings, "RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setattr(payments.settings, "RAZORPAY_KEY_SECRET", "rzp_test_secret")
    monkeypatch.setattr(payments.httpx, "AsyncClient", _FakeRazorpayClient)
    _FakeRazorpayClient.response = _FakeRazorpayResponse(200, {"id": "order_live_123"})

    device = _uid("live-gateway")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/payments/create-order",
            json={"deviceId": device, "plan": "popular"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["orderId"] == "order_live_123"
    assert body["mock"] is False
    assert body["credits"] == 120
    request = _FakeRazorpayClient.last_request
    assert request["url"] == payments.RAZORPAY_ORDERS_URL
    assert request["auth"] == ("rzp_test_key", "rzp_test_secret")
    assert request["json"]["amount"] == payments.settings.CREDIT_PACKS["popular"]["price_paise"]


@pytest.mark.asyncio
async def test_gateway_failure_is_an_upstream_error(monkeypatch):
    from app.routes import payments

    monkeypatch.setattr(payments, "MOCK_MODE", False)
    monkeypatch.setattr(payments.settings, "RAZORPAY_KEY_ID", "rzp_test_key")
    monkeypatch.setattr(payments.settings, "RAZORPAY_KEY_SECRET", "rzp_test_secret")
    monkeypatch.setattr(payments.httpx, "AsyncClient", _FakeRazorpayClient)
    _FakeRazorpayClient.response = _FakeRazorpayResponse(401, {"error": "bad key"})

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/payments/create-order",
            json={"deviceId": _uid("bad-gateway"), "plan": "starter"},
        )
    assert resp.status_code == 502
    assert resp.json()["code"] == 502


# ── Context parsing ─────────────────────────────────────────────────────────


def test_malformed_context_types_are_ignored_not_fatal():
    from app.utils.tags import parse_context

    ctx = parse_context('{"title": {"nested": 1}, "week": 12, "url": 5}')
    assert ctx.title == ""
    assert ctx.week == ""


def test_json_array_response_is_a_decode_error_not_type_error():
    from app.services.vision_service import _extract_json

    with pytest.raises(json.JSONDecodeError):
        _extract_json("[1, 2, 3]")
    with pytest.raises(json.JSONDecodeError):
        _extract_json('"just a string"')


# ── Password boundaries ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_signup_rejects_password_beyond_bcrypt_limit():
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/auth/signup",
            json={
                "email": f"long-{uuid.uuid4().hex}@example.com",
                "password": "a" * 73,
                "name": "Long Password",
            },
        )
    assert resp.status_code == 422
    assert "72" in resp.json()["error"]


@pytest.mark.asyncio
async def test_gateway_signature_is_verified_before_crediting(monkeypatch):
    import hashlib
    import hmac

    from app.routes import payments

    monkeypatch.setattr(payments, "MOCK_MODE", False)
    monkeypatch.setattr(payments.settings, "RAZORPAY_KEY_SECRET", "rzp_test_secret")
    device = _uid("signed-order")
    _set_device_credits(device, 0)
    order_id = f"order_{uuid.uuid4().hex[:16]}"
    payments._db_put_order(order_id, device, "starter", 50, 4900)
    payment_id = "pay_test_123"
    expected = hmac.new(
        b"rzp_test_secret",
        f"{order_id}|{payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        rejected = await client.post(
            "/api/payments/verify",
            json={
                "razorpay_order_id": order_id,
                "razorpay_payment_id": payment_id,
                "razorpay_signature": "0" * 64,
                "deviceId": device,
            },
        )
        assert rejected.status_code == 422
        assert _device_credits(device) == 0

        accepted = await client.post(
            "/api/payments/verify",
            json={
                "razorpay_order_id": order_id,
                "razorpay_payment_id": payment_id,
                "razorpay_signature": expected,
                "deviceId": device,
            },
        )
    assert accepted.status_code == 200
    assert accepted.json()["creditsAdded"] == 50
    assert _device_credits(device) == 50


# ── OCR layout classification ───────────────────────────────────────────────


def _ocr_token(x: float, y: float, text: str, width: float = 150.0) -> tuple:
    box = [[x, y - 10], [x + width, y - 10], [x + width, y + 10], [x, y + 10]]
    return (box, text, 0.95)


def test_prose_lines_are_not_classified_as_a_table():
    from app.services.ocr_service import is_table_layout

    prose = [
        _ocr_token(30, 20, "Introduction to Statistics"),
        _ocr_token(30, 80, "Mean is the average of all values"),
        _ocr_token(30, 140, "Median is the middle value"),
        _ocr_token(30, 200, "Mode is the most frequent value"),
    ]
    assert is_table_layout(prose) is False


def test_regular_grid_is_classified_as_a_table():
    from app.services.ocr_service import format_as_table, is_table_layout

    rows = [
        ["Name", "Age", "Score"],
        ["Alice", "20", "85"],
        ["Bob", "22", "90"],
    ]
    tokens = [
        _ocr_token(40 + col * 200, 30 + row * 60, cell, width=120.0)
        for row, cells in enumerate(rows)
        for col, cell in enumerate(cells)
    ]
    assert is_table_layout(tokens) is True
    markdown = format_as_table(tokens)
    assert markdown.count("\n") == 3
    pipe_counts = {line.count("|") for line in markdown.splitlines()}
    assert pipe_counts == {4}


def test_ragged_table_rows_are_padded_not_dropped():
    from app.services.ocr_service import format_as_table

    tokens = [
        _ocr_token(40, 30, "Name", width=100.0),
        _ocr_token(240, 30, "Score", width=100.0),
        _ocr_token(40, 90, "Alice", width=100.0),
        # second row intentionally misses the score cell
    ]
    markdown = format_as_table(tokens)
    lines = markdown.splitlines()
    assert len(lines) == 3
    assert {line.count("|") for line in lines} == {3}
    assert "Alice" in markdown


# ── Decode-time memory bounds ───────────────────────────────────────────────


def test_decompression_bomb_dimensions_are_refused():
    from app.services.preprocessor import _validate_resolution

    with pytest.raises(ValueError, match="megapixels"):
        _validate_resolution(30000, 30000)
    # A 12 MP phone photo stays well inside the 24 MP budget.
    _validate_resolution(4000, 3000)
    with pytest.raises(ValueError, match="Invalid image data"):
        _validate_resolution(0, 100)


def test_invalid_image_raises_value_error_before_vision():
    from app.services.preprocessor import enhance_for_vision

    with pytest.raises(ValueError, match="Invalid image data"):
        enhance_for_vision(b"this is not an image")


def test_upload_read_refuses_declared_oversize_without_reading():
    from app.exceptions import ImageTooLargeError
    from app.routes.extract import _read_validated_image

    payload = b"x" * 16
    upload = UploadFile(
        file=io.BytesIO(payload),
        filename="bomb.png",
        size=11 * 1024 * 1024,
    )
    with pytest.raises(ImageTooLargeError):
        asyncio.run(_read_validated_image(upload))


def test_upload_read_caps_body_when_declared_size_is_absent(monkeypatch):
    from app.exceptions import ImageTooLargeError
    from app.routes import extract as extract_routes

    # A client controls the multipart body, not the declared size Starlette
    # records, so the read itself must stay bounded.
    monkeypatch.setattr(extract_routes.settings, "MAX_IMAGE_SIZE_MB", 1)
    upload = UploadFile(
        file=io.BytesIO(b"x" * (1024 * 1024 + 1)),
        filename="bomb.png",
        size=None,
    )
    with pytest.raises(ImageTooLargeError):
        asyncio.run(extract_routes._read_validated_image(upload))


# ── Chrome extension session exchange ───────────────────────────────────────


@pytest.mark.asyncio
async def test_chrome_auth_returns_snapnote_session_not_google_token(monkeypatch):
    from app.routes import auth as auth_routes
    from app.utils.auth import decode_token

    email = f"chrome-{uuid.uuid4().hex}@example.com"
    monkeypatch.setattr(
        auth_routes,
        "_verify_google_access_token",
        AsyncMock(
            return_value={
                "email": email,
                "name": "Chrome Student",
                "picture": "",
                "email_verified": True,
            }
        ),
    )
    # The test environment may carry a newer bcrypt than production's pinned
    # 4.0.1, which passlib 1.7.4 cannot initialize. Hashing is not what this
    # test covers, so stub it and keep the assertion on the token contract.
    monkeypatch.setattr(auth_routes, "hash_password", lambda _password: "stub-hash")

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/auth/chrome",
            json={"googleAccessToken": "google-access-token-not-a-jwt"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["accessToken"] != "google-access-token-not-a-jwt"
    decoded = decode_token(body["accessToken"])
    assert decoded["sub"] == body["uid"]
    assert decoded["email"] == email

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        me = await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {body['accessToken']}"},
        )
    assert me.status_code == 200
    assert me.json()["uid"] == body["uid"]


@pytest.mark.asyncio
async def test_existing_visual_is_returned_without_resending_image():
    from app.utils.credits_store import record_diagram_grant, set_visual_result

    device = _uid("visual-owner")
    diagram_id = uuid.uuid4().hex
    stored_svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>verified</text></svg>'
    record_diagram_grant(device, diagram_id, "{}")
    assert set_visual_result(
        diagram_id, device, "deterministic", visual_svg=stored_svg
    )

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/visual",
            data={"deviceId": device, "diagramId": diagram_id},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "already_generated"
    assert body["renderMode"] == "deterministic"
    assert body["imageSvg"] == stored_svg


@pytest.mark.asyncio
async def test_visual_generation_requires_the_screenshot():
    from app.utils.credits_store import record_diagram_grant

    device = _uid("visual-needs-image")
    diagram_id = uuid.uuid4().hex
    record_diagram_grant(device, diagram_id, "{}")

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/extract/visual",
            data={"deviceId": device, "diagramId": diagram_id},
        )
    assert resp.status_code == 422
    assert "screenshot" in resp.json()["error"].lower()


# ── CORS ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wildcard_cors_never_allows_credentials():
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.options(
            "/api/extract/text",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    assert resp.headers.get("access-control-allow-credentials") is None
