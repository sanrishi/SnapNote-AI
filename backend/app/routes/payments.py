import hashlib
import hmac
import logging
import secrets
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.exceptions import InvalidInputError, PaymentsUnavailableError, UpstreamError
from app.utils.credits_store import (
    complete_order_and_credit,
    peek_credits,
    reset_device_credits,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _mock_payments_enabled() -> bool:
    """Mock checkout is a development aid, never a production fallback.

    It used to switch on whenever the Razorpay keys were absent, which meant a
    deployment that forgot the keys silently minted credits for any order id
    and any signature. Now it requires DEBUG or an explicit PAYMENTS_MOCK flag.
    """
    if settings.PAYMENTS_MOCK:
        return True
    configured = bool(settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET)
    return settings.DEBUG and not configured


MOCK_MODE = _mock_payments_enabled()
if MOCK_MODE:
    logger.warning("Razorpay mock payments enabled (development only — no real charges)")


def _orders_conn():
    from app.utils.credits_store import _get_conn

    return _get_conn()


def _db_get_order(order_id: str) -> dict | None:
    row = _orders_conn().execute(
        "SELECT order_id, device_id, plan, credits, amount, status, razorpay_payment_id "
        "FROM razorpay_orders WHERE order_id = ?",
        (order_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "order_id": row["order_id"],
        "deviceId": row["device_id"],
        "plan": row["plan"],
        "credits": row["credits"],
        "amount": row["amount"],
        "status": row["status"],
        "razorpay_payment_id": row["razorpay_payment_id"],
    }


def _db_put_order(order_id: str, device_id: str, plan: str, credits: int, amount: int) -> None:
    _orders_conn().execute(
        "INSERT OR IGNORE INTO razorpay_orders (order_id, device_id, plan, credits, amount, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'created', ?)",
        (order_id, device_id, plan, credits, amount, int(time.time())),
    )
    _orders_conn().commit()


class CreateOrderRequest(BaseModel):
    deviceId: str = Field(min_length=1, max_length=128)
    plan: str = Field(min_length=1, max_length=32)


class CreateOrderResponse(BaseModel):
    orderId: str
    amount: int
    currency: str = "INR"
    keyId: str
    credits: int
    mock: bool = False


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str = Field(min_length=1, max_length=128)
    razorpay_payment_id: str = Field(min_length=1, max_length=128)
    razorpay_signature: str = Field(min_length=1, max_length=256)
    deviceId: str = Field(min_length=1, max_length=128)


class VerifyPaymentResponse(BaseModel):
    success: bool
    creditsAdded: int
    creditsRemaining: int


class CreditPlan(BaseModel):
    slug: str
    credits: int
    price_paise: int
    price_rupees: float


class PlanListResponse(BaseModel):
    plans: list[CreditPlan]


class ResetResponse(BaseModel):
    creditsRemaining: int


RAZORPAY_ORDERS_URL = "https://api.razorpay.com/v1/orders"


async def _create_razorpay_order(pack: dict[str, int], device_id: str, plan: str) -> str:
    """Create a gateway order over the REST API and return its id.

    The razorpay SDK is not a project dependency, so the previous
    ``import razorpay`` made every real checkout a 500. The REST call needs
    only httpx (already installed) and the configured key pair.
    """
    receipt = f"snapnote_{device_id[:8]}_{int(time.time())}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                RAZORPAY_ORDERS_URL,
                json={
                    "amount": pack["price_paise"],
                    "currency": "INR",
                    "receipt": receipt,
                    "notes": {"deviceId": device_id, "plan": plan},
                },
                auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET),
            )
    except httpx.RequestError as exc:
        logger.error("Razorpay order request failed: %s", exc)
        raise UpstreamError(service="Payment gateway", detail="could not reach the gateway")
    if response.status_code >= 400:
        logger.error(
            "Razorpay order creation HTTP %s: %s",
            response.status_code,
            " ".join(response.text.split())[:300],
        )
        raise UpstreamError(
            service="Payment gateway", detail=f"HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError:
        payload = None
    order_id = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(order_id, str) or not order_id:
        raise UpstreamError(service="Payment gateway", detail="no order id in response")
    return order_id


@router.get("/plans")
async def list_plans() -> PlanListResponse:
    plans = [
        CreditPlan(
            slug=slug,
            credits=int(pack["credits"]),
            price_paise=int(pack["price_paise"]),
            price_rupees=pack["price_paise"] / 100,
        )
        for slug, pack in settings.CREDIT_PACKS.items()
    ]
    return PlanListResponse(plans=plans)


@router.post("/create-order", response_model=CreateOrderResponse)
async def create_order(req: CreateOrderRequest) -> CreateOrderResponse:
    pack = settings.CREDIT_PACKS.get(req.plan)
    if not pack:
        raise InvalidInputError(message=f"Unknown plan: {req.plan}")

    if MOCK_MODE:
        order_id = f"mock_{secrets.token_hex(8)}"
        _db_put_order(order_id, req.deviceId, req.plan, pack["credits"], pack["price_paise"])
        return CreateOrderResponse(
            orderId=order_id,
            amount=pack["price_paise"],
            keyId="mock",
            credits=pack["credits"],
            mock=True,
        )

    if not (settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET):
        # Never fall through to a half-configured gateway.
        raise PaymentsUnavailableError()

    order_id = await _create_razorpay_order(pack, req.deviceId, req.plan)
    _db_put_order(order_id, req.deviceId, req.plan, pack["credits"], pack["price_paise"])

    return CreateOrderResponse(
        orderId=order_id,
        amount=pack["price_paise"],
        keyId=settings.RAZORPAY_KEY_ID,
        credits=pack["credits"],
    )


@router.post("/verify", response_model=VerifyPaymentResponse)
async def verify_payment(req: VerifyPaymentRequest) -> VerifyPaymentResponse:
    if not MOCK_MODE:
        if not settings.RAZORPAY_KEY_SECRET:
            raise PaymentsUnavailableError()
        expected = hmac.new(
            settings.RAZORPAY_KEY_SECRET.encode(),
            f"{req.razorpay_order_id}|{req.razorpay_payment_id}".encode(),
            hashlib.sha256,
        ).hexdigest()
        # Constant-time compare prevents a signature oracle.
        if not hmac.compare_digest(expected, req.razorpay_signature):
            raise InvalidInputError(message="Payment signature mismatch")
    elif not req.razorpay_order_id.startswith("mock_"):
        # Mock checkout can only complete orders this server minted in mock
        # mode; a real gateway order id never becomes free credits.
        raise InvalidInputError(message="Unknown order")

    order = _db_get_order(req.razorpay_order_id)
    if order is None:
        logger.warning("Unknown order %s", req.razorpay_order_id[:12])
        remaining, _ = peek_credits(req.deviceId)
        return VerifyPaymentResponse(success=False, creditsAdded=0, creditsRemaining=remaining)

    if order["deviceId"] != req.deviceId:
        # Credits belong to the device that created the order. Before this
        # check a valid order id could be replayed against another device.
        logger.warning(
            "Order %s belongs to a different device (requested %s)",
            req.razorpay_order_id[:12],
            req.deviceId[:8],
        )
        raise InvalidInputError(message="This order does not belong to this device.")

    if order["status"] == "completed":
        remaining, _ = peek_credits(order["deviceId"])
        return VerifyPaymentResponse(
            success=True,
            creditsAdded=0,
            creditsRemaining=remaining,
        )

    credits = int(order["credits"])
    completed_now, new_remaining = complete_order_and_credit(
        req.razorpay_order_id, order["deviceId"], req.razorpay_payment_id
    )
    if not completed_now:
        # A concurrent verify won the claim (or the order vanished). Report the
        # stored state instead of crediting twice.
        refreshed = _db_get_order(req.razorpay_order_id)
        if refreshed is None or refreshed["status"] != "completed":
            raise InvalidInputError(
                message="Order could not be completed. Please contact support."
            )
        logger.info("Order %s already completed; no credits added", req.razorpay_order_id[:12])
        return VerifyPaymentResponse(
            success=True,
            creditsAdded=0,
            creditsRemaining=new_remaining,
        )

    logger.info(
        "Payment verified: device=%s plan=%s credits=%d id=%s",
        order["deviceId"][:8],
        order["plan"],
        credits,
        req.razorpay_payment_id[:12],
    )
    return VerifyPaymentResponse(
        success=True,
        creditsAdded=credits,
        creditsRemaining=new_remaining,
    )


@router.post("/reset", response_model=ResetResponse)
async def reset_device(deviceId: str) -> ResetResponse:
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    reset_device_credits(deviceId)
    logger.info("Credits reset for device %s", deviceId[:8])
    return ResetResponse(creditsRemaining=settings.FREE_CREDITS_MONTHLY)
