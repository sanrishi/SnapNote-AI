import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from app.config import settings
from app.exceptions import InvalidInputError
from app.utils.credits_store import get_credits, add_credits, init_device

logger = logging.getLogger(__name__)
router = APIRouter()

# Kept for backward compat in tests that patch _orders, but real source of
# truth is the razorpay_orders table in credits.db (see credits_store).
_orders: dict[str, dict] = {}

MOCK_MODE = not (settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET)
if MOCK_MODE:
    logger.warning("Razorpay not configured — mock payment enabled (no real charges)")


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
    # Mirror to in-memory dict so legacy test patches still see it.
    _orders[order_id] = {
        "deviceId": device_id,
        "plan": plan,
        "credits": credits,
        "amount": amount,
        "status": "created",
        "razorpay_payment_id": None,
    }


def _db_mark_completed(order_id: str, payment_id: str) -> None:
    _orders_conn().execute(
        "UPDATE razorpay_orders SET status='completed', razorpay_payment_id=? WHERE order_id=?",
        (payment_id, order_id),
    )
    _orders_conn().commit()
    if order_id in _orders:
        _orders[order_id]["status"] = "completed"
        _orders[order_id]["razorpay_payment_id"] = payment_id


class CreateOrderRequest(BaseModel):
    deviceId: str
    plan: str


class CreateOrderResponse(BaseModel):
    orderId: str
    amount: int
    currency: str = "INR"
    keyId: str
    credits: int
    mock: bool = False


class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    deviceId: str


class VerifyPaymentResponse(BaseModel):
    success: bool
    creditsAdded: int
    creditsRemaining: int


class PlanListResponse(BaseModel):
    plans: list[dict]


@router.get("/plans")
async def list_plans() -> PlanListResponse:
    plans = []
    for slug, pack in settings.CREDIT_PACKS.items():
        plans.append({
            "slug": slug,
            "credits": pack["credits"],
            "price_paise": pack["price_paise"],
            "price_rupees": pack["price_paise"] / 100,
        })
    return PlanListResponse(plans=plans)


def _build_order(req: CreateOrderRequest, pack: dict) -> dict:
    order_id = f"mock_{secrets.token_hex(8)}" if MOCK_MODE else ""
    return {
        "deviceId": req.deviceId,
        "plan": req.plan,
        "credits": pack["credits"],
        "amount": pack["price_paise"],
        "status": "created",
        "razorpay_payment_id": None,
    }


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

    import razorpay
    razorpay_client = razorpay.Client(
        auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
    )
    try:
        order = razorpay_client.order.create({
            "amount": pack["price_paise"],
            "currency": "INR",
            "receipt": f"snapnote_{req.deviceId[:8]}_{datetime.now(timezone.utc).timestamp():.0f}",
            "notes": {"deviceId": req.deviceId, "plan": req.plan},
        })
    except Exception as e:
        logger.error("Razorpay order creation failed: %s", str(e))
        raise InvalidInputError(message="Payment gateway error")

    _db_put_order(order["id"], req.deviceId, req.plan, pack["credits"], pack["price_paise"])

    return CreateOrderResponse(
        orderId=order["id"],
        amount=pack["price_paise"],
        keyId=settings.RAZORPAY_KEY_ID,
        credits=pack["credits"],
    )


@router.post("/verify", response_model=VerifyPaymentResponse)
async def verify_payment(req: VerifyPaymentRequest) -> VerifyPaymentResponse:
    # Signature first (even in mock mode we validate when keys exist).
    if not MOCK_MODE:
        if not settings.RAZORPAY_KEY_SECRET:
            raise InvalidInputError(message="Payment gateway not configured")
        expected = hmac.new(
            settings.RAZORPAY_KEY_SECRET.encode(),
            f"{req.razorpay_order_id}|{req.razorpay_payment_id}".encode(),
            hashlib.sha256,
        ).hexdigest()
        if expected != req.razorpay_signature:
            raise InvalidInputError(message="Payment signature mismatch")

    order = _db_get_order(req.razorpay_order_id)
    # Fallback to legacy in-memory dict (tests patch it).
    if order is None:
        order = _orders.get(req.razorpay_order_id)

    if not order:
        logger.warning("Unknown order %s", req.razorpay_order_id[:12])
        remaining, _ = get_credits(req.deviceId)
        return VerifyPaymentResponse(success=False, creditsAdded=0, creditsRemaining=remaining)

    if order["status"] == "completed":
        remaining, _ = get_credits(req.deviceId)
        return VerifyPaymentResponse(
            success=True,
            creditsAdded=0,
            creditsRemaining=remaining,
        )

    credits = order["credits"]
    new_remaining = add_credits(req.deviceId, credits)
    _db_mark_completed(req.razorpay_order_id, req.razorpay_payment_id)

    logger.info(
        "Payment verified: device=%s plan=%s credits=%d",
        req.deviceId[:8], order["plan"], credits,
    )

    return VerifyPaymentResponse(
        success=True,
        creditsAdded=credits,
        creditsRemaining=new_remaining,
    )


@router.post("/reset")
async def reset_device(deviceId: str) -> dict:
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    init_device(deviceId)
    remaining, _ = get_credits(deviceId)
    logger.info("Credits reset for device %s", deviceId[:8])
    return {"creditsRemaining": remaining}
