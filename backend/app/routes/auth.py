import logging
import secrets
from functools import lru_cache

import httpx
from fastapi import APIRouter, Header
from pydantic import BaseModel, EmailStr, Field

from app.config import settings
from app.exceptions import AuthError, InvalidInputError
from app.models.schemas import AuthResponse, DeviceAuthRequest, DeviceAuthResponse, GoogleAuthRequest
from app.utils.auth import create_access_token, decode_token, hash_password, verify_password
from app.utils.credits_store import (
    UserRecord,
    create_user,
    get_user_by_email,
    get_user_by_id,
    get_user_credits,
    init_device,
    set_user_picture,
)

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


def _verify_google_id_token(id_token: str) -> dict:
    """Verify a Google ID token (Google Identity Services). No Firebase needed."""
    client_id = (settings.GOOGLE_CLIENT_ID or "").strip()
    if not client_id:
        raise AuthError(message="Google sign-in not configured on server yet. Please use email.")
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        req = google_requests.Request()
        info = google_id_token.verify_oauth2_token(id_token, req, client_id)
    except Exception as e:
        logger.warning("Google ID token verify failed: %s", e)
        raise AuthError(message="Could not verify Google sign-in. Please use email.")
    if not info.get("email_verified", False):
        raise AuthError(message="Google account email not verified.")
    return info


def _truthy(value: object) -> bool:
    """Google's tokeninfo endpoint returns booleans as strings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def _allowed_client_ids() -> set[str]:
    raw = (settings.GOOGLE_ALLOWED_CLIENT_IDS or settings.GOOGLE_CLIENT_ID or "").strip()
    return {part.strip() for part in raw.split(",") if part.strip()}


async def _verify_google_access_token(access_token: str) -> dict:
    """Verify a Google OAuth 2.0 access token (Chrome extension identity).

    ``chrome.identity.getAuthToken`` yields an access token, not an ID token.
    Google's tokeninfo endpoint validates it; the audience is then checked
    against the configured client allowlist so a token minted for an unrelated
    Google app cannot sign in here.
    """
    token = (access_token or "").strip()
    if not token:
        raise AuthError(message="Missing Google token")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://www.googleapis.com/oauth2/v3/tokeninfo",
                params={"access_token": token},
            )
    except httpx.RequestError as e:
        logger.error("Google token verification failed: %s", str(e))
        raise AuthError(message="Failed to verify Google token")
    if resp.status_code != 200:
        raise AuthError(message="Invalid Google token")
    try:
        data = resp.json()
    except ValueError:
        raise AuthError(message="Invalid Google token")
    if not _truthy(data.get("email_verified")):
        raise AuthError(message="Google account email not verified.")
    allowed = _allowed_client_ids()
    if allowed:
        audience = {str(data.get("aud") or ""), str(data.get("azp") or "")}
        if not (audience & allowed):
            logger.warning(
                "Rejected Google token for unexpected client (aud=%s azp=%s)",
                data.get("aud"),
                data.get("azp"),
            )
            raise AuthError(message="This Google token was issued for a different application.")
    return data


def _user_for_google(email: str, name: str, picture: str) -> UserRecord:
    """Find or create the local account for a verified Google identity."""
    row = get_user_by_email(email)
    if row is None:
        placeholder_hash = hash_password(secrets.token_urlsafe(32))
        try:
            user_id = create_user(email, placeholder_hash, name, picture=picture)
            row = get_user_by_id(user_id)
        except Exception:
            # Concurrent first sign-in: the other request created the account.
            row = get_user_by_email(email)
            if row is None:
                raise
    elif picture and not row["picture"]:
        # existing account: store the profile picture once
        set_user_picture(row["id"], picture)
        row = get_user_by_id(row["id"])
    assert row is not None
    return row


def _auth_response(row: UserRecord) -> AuthResponse:
    return AuthResponse(
        accessToken=create_access_token(row["id"], row["email"]),
        uid=row["id"],
        email=row["email"],
        name=row["name"],
        picture=row["picture"],
        creditsRemaining=row["credits_remaining"],
    )


@router.post("/google", response_model=AuthResponse)
async def google_auth(req: GoogleAuthRequest) -> AuthResponse:
    # Prefer direct Google ID token verification (Google Identity Services).
    if (settings.GOOGLE_CLIENT_ID or "").strip():
        decoded = _verify_google_id_token(req.idToken)
    else:
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
        if decoded.get("email_verified") is False:
            raise AuthError(message="Google account email not verified.")

    email = (decoded.get("email") or "").lower().strip()
    if not email:
        raise AuthError(message="Google account has no email")
    name = decoded.get("name") or email.split("@")[0]
    picture = decoded.get("picture") or ""
    row = _user_for_google(email, name, picture)
    logger.info("Google auth: %s -> %s", email, row["id"][:8])
    return _auth_response(row)


class ChromeAuthRequest(BaseModel):
    googleAccessToken: str = Field(min_length=1, max_length=4096)


@router.post("/chrome", response_model=AuthResponse)
async def chrome_auth(req: ChromeAuthRequest) -> AuthResponse:
    """Exchange a verified Google access token for a SnapNote session.

    This endpoint used to return the Google access token itself as
    ``accessToken``. It is not a SnapNote JWT, so every downstream call failed
    its decode and silently fell back to the anonymous device identity while
    the UI showed a fake 50-credit balance. Now the Google identity is mapped
    to a real account and a real session token is issued.
    """
    info = await _verify_google_access_token(req.googleAccessToken)
    email = (info.get("email") or "").lower().strip()
    if not email:
        raise AuthError(message="Google account has no email")
    name = info.get("name") or email.split("@")[0]
    picture = info.get("picture") or ""
    row = _user_for_google(email, name, picture)
    logger.info("Chrome extension auth: %s -> %s", email, row["id"][:8])
    return _auth_response(row)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)
    name: str = Field(default="", max_length=80)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


@router.post("/signup", response_model=AuthResponse)
async def signup(req: SignupRequest) -> AuthResponse:
    if len(req.password) < 6:
        raise InvalidInputError(message="Password must be at least 6 characters")
    # bcrypt hashes at most 72 bytes; reject longer input instead of relying on
    # implicit truncation (and newer bcrypt backends raise instead of truncate).
    if len(req.password.encode("utf-8")) > 72:
        raise InvalidInputError(message="Password must be 72 bytes or fewer")
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
    return AuthResponse(
        accessToken=token,
        uid=user_id,
        email=email,
        name=req.name or email.split("@")[0],
        creditsRemaining=remaining,
    )


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest) -> AuthResponse:
    row = get_user_by_email(req.email.lower().strip())
    if row is None or not verify_password(req.password, row["password_hash"]):
        raise AuthError(message="Invalid email or password")
    token = create_access_token(row["id"], row["email"])
    logger.info("User logged in: %s", row["email"])
    return AuthResponse(
        accessToken=token,
        uid=row["id"],
        email=row["email"],
        name=row["name"],
        picture=row["picture"],
        creditsRemaining=row["credits_remaining"],
    )


@router.get("/me", response_model=AuthResponse)
async def me(authorization: str | None = Header(default=None)) -> AuthResponse:
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthError(message="Not authenticated")
    data = decode_token(authorization[7:])
    uid = data.get("sub")
    if not isinstance(uid, str) or not uid:
        raise AuthError(message="Invalid session. Please log in again.")
    row = get_user_by_id(uid)
    if row is None:
        raise AuthError(message="User not found")
    return AuthResponse(
        accessToken=authorization[7:],
        uid=row["id"],
        email=row["email"],
        name=row["name"],
        picture=row["picture"],
        creditsRemaining=row["credits_remaining"],
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
