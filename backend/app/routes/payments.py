import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.exceptions import InvalidInputError
from app.utils.credits_store import get_credits, add_credits, init_device

logger = logging.getLogger(__name__)
router = APIRouter()

_orders: dict[str, dict] = {}

MOCK_MODE = not (settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET)
if MOCK_MODE:
    logger.warning("Razorpay not configured — mock payment enabled (no real charges)")


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
        _orders[order_id] = _build_order(req, pack)
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

    _orders[order["id"]] = {
        "deviceId": req.deviceId,
        "plan": req.plan,
        "credits": pack["credits"],
        "amount": pack["price_paise"],
        "status": "created",
        "razorpay_payment_id": None,
    }

    return CreateOrderResponse(
        orderId=order["id"],
        amount=pack["price_paise"],
        keyId=settings.RAZORPAY_KEY_ID,
        credits=pack["credits"],
    )


@router.post("/verify", response_model=VerifyPaymentResponse)
async def verify_payment(req: VerifyPaymentRequest) -> VerifyPaymentResponse:
    order = _orders.get(req.razorpay_order_id)

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
    order["status"] = "completed"
    order["razorpay_payment_id"] = req.razorpay_payment_id

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
    init_device(deviceId)
    remaining, _ = get_credits(deviceId)
    logger.info("Credits reset for device %s", deviceId[:8])
    return {"creditsRemaining": remaining}
