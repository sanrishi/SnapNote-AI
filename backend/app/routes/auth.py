import logging
from functools import lru_cache

import firebase_admin
import httpx
from fastapi import APIRouter
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials
from pydantic import BaseModel

from app.config import settings
from app.exceptions import AuthError
from app.models.schemas import AuthResponse, DeviceAuthRequest, DeviceAuthResponse, GoogleAuthRequest
from app.utils.credits_store import get_credits, init_device

logger = logging.getLogger(__name__)
router = APIRouter()


@lru_cache(maxsize=1)
def _get_firebase_app():
    if firebase_admin._apps:
        return firebase_admin.get_app()
    try:
        return firebase_admin.initialize_app(
            credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
        )
    except Exception as e:
        logger.warning("Firebase not configured: %s", str(e))
        return None


@router.post("/google", response_model=AuthResponse)
async def google_auth(req: GoogleAuthRequest) -> AuthResponse:
    app = _get_firebase_app()
    if app is None:
        raise AuthError(message="Firebase not configured on server")

    try:
        decoded = firebase_auth.verify_id_token(req.idToken)
    except ValueError as e:
        raise AuthError(message=f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error("Firebase auth failed: %s", str(e))
        raise AuthError()

    return AuthResponse(
        accessToken=req.idToken,
        uid=decoded["uid"],
        email=decoded.get("email", ""),
        name=decoded.get("name", "User"),
        creditsRemaining=settings.FREE_CREDITS_MONTHLY,
    )


class ChromeAuthRequest(BaseModel):
    googleAccessToken: str


@router.post("/chrome", response_model=AuthResponse)
async def chrome_auth(req: ChromeAuthRequest) -> AuthResponse:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/tokeninfo",
                params={"access_token": req.googleAccessToken},
            )
            if resp.status_code != 200:
                raise AuthError(message="Invalid Google token")
            data = resp.json()
    except httpx.RequestError as e:
        logger.error("Google token verification failed: %s", str(e))
        raise AuthError(message="Failed to verify Google token")

    return AuthResponse(
        accessToken=req.googleAccessToken,
        uid=data["sub"],
        email=data.get("email", ""),
        name=data.get("name", data.get("email", "User")),
        creditsRemaining=settings.FREE_CREDITS_MONTHLY,
    )


@router.post("/device", response_model=DeviceAuthResponse)
async def device_auth(req: DeviceAuthRequest) -> DeviceAuthResponse:
    remaining, used = init_device(req.deviceId)
    logger.info("Device auth: %s -> %d credits remaining", req.deviceId[:8], remaining)
    return DeviceAuthResponse(
        deviceId=req.deviceId,
        creditsRemaining=remaining,
        creditsUsed=used,
    )
