import logging
import re
from functools import lru_cache

import httpx
from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, EmailStr

from app.config import settings
from app.exceptions import AuthError, InvalidInputError
from app.models.schemas import AuthResponse, DeviceAuthRequest, DeviceAuthResponse, GoogleAuthRequest
from app.utils.auth import create_access_token, decode_token, hash_password, verify_password
from app.utils.credits_store import create_user, get_credits, get_user_by_email, get_user_by_id, get_user_credits, init_device

logger = logging.getLogger(__name__)
router = APIRouter()


@lru_cache(maxsize=1)
def _get_firebase_app():
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError:
        logger.warning("firebase-admin not installed")
        return None
    if firebase_admin._apps:
        return firebase_admin.get_app()
    try:
        cred_path = (settings.FIREBASE_CREDENTIALS_PATH or "").strip()
        if not cred_path:
            logger.warning("Firebase credentials not configured (FIREBASE_CREDENTIALS_PATH empty)")
            return None
        if cred_path.lstrip().startswith("{"):
            # raw JSON blob pasted into the env var (Render friendly)
            import json

            cred_dict = json.loads(cred_path)
            cred = credentials.Certificate(cred_dict)
        else:
            # file path on disk
            cred = credentials.Certificate(cred_path)
        return firebase_admin.initialize_app(cred)
    except Exception as e:
        logger.warning("Firebase not configured: %s", str(e))
        return None


@router.post("/google", response_model=AuthResponse)
async def google_auth(req: GoogleAuthRequest) -> AuthResponse:
    app = _get_firebase_app()
    if app is None:
        raise AuthError(message="Firebase not configured on server")
    try:
        from firebase_admin import auth as firebase_auth
        decoded = firebase_auth.verify_id_token(req.idToken)
    except ValueError as e:
        raise AuthError(message=f"Invalid token: {str(e)}")
    except Exception as e:
        logger.error("Firebase auth failed: %s", str(e))
        raise AuthError()

    email = (decoded.get("email") or "").lower().strip()
    name = decoded.get("name") or (email.split("@")[0] if email else "User")
    if not email:
        raise AuthError(message="Google account has no email")
    row = get_user_by_email(email)
    if row is None:
        import secrets

        placeholder_hash = hash_password(secrets.token_urlsafe(32))
        try:
            user_id = create_user(email, placeholder_hash, name)
            row = get_user_by_id(user_id)
        except Exception:
            row = get_user_by_email(email)
            if row is None:
                raise
    assert row is not None
    token = create_access_token(row["id"], row["email"])
    logger.info("Google auth: %s -> %s", email, row["id"][:8])
    return AuthResponse(accessToken=token, uid=row["id"], email=row["email"], name=row["name"], creditsRemaining=row["credits_remaining"])


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


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/signup", response_model=AuthResponse)
async def signup(req: SignupRequest) -> AuthResponse:
    if len(req.password) < 6:
        raise InvalidInputError(message="Password must be at least 6 characters")
    if req.name and len(req.name) > 50:
        raise InvalidInputError(message="Name too long")
    email = req.email.lower().strip()
    if get_user_by_email(email) is not None:
        raise InvalidInputError(message="Email already registered. Please log in.")
    pwd_hash = hash_password(req.password)
    user_id = create_user(email, pwd_hash, req.name or email.split("@")[0])
    token = create_access_token(user_id, email)
    remaining, _ = get_user_credits(user_id)
    logger.info("User signed up: %s (%s)", email, user_id[:8])
    return AuthResponse(accessToken=token, uid=user_id, email=email, name=req.name or email.split("@")[0], creditsRemaining=remaining)


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest) -> AuthResponse:
    row = get_user_by_email(req.email.lower().strip())
    if row is None or not verify_password(req.password, row["password_hash"]):
        raise AuthError(message="Invalid email or password")
    token = create_access_token(row["id"], row["email"])
    logger.info("User logged in: %s", row["email"])
    return AuthResponse(
        accessToken=token, uid=row["id"], email=row["email"], name=row["name"], creditsRemaining=row["credits_remaining"]
    )


@router.get("/me", response_model=AuthResponse)
async def me(authorization: str | None = Header(default=None)) -> AuthResponse:
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthError(message="Not authenticated")
    data = decode_token(authorization[7:])
    row = get_user_by_id(data["sub"])
    if row is None:
        raise AuthError(message="User not found")
    return AuthResponse(accessToken=authorization[7:], uid=row["id"], email=row["email"], name=row["name"], creditsRemaining=row["credits_remaining"])


@router.post("/device", response_model=DeviceAuthResponse)
async def device_auth(req: DeviceAuthRequest) -> DeviceAuthResponse:
    remaining, used = init_device(req.deviceId)
    logger.info("Device auth: %s -> %d credits remaining", req.deviceId[:8], remaining)
    return DeviceAuthResponse(
        deviceId=req.deviceId,
        creditsRemaining=remaining,
        creditsUsed=used,
    )
