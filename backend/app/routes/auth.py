import logging
from functools import lru_cache
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from fastapi import APIRouter

from app.config import settings
from app.models.schemas import GoogleAuthRequest, AuthResponse
from app.exceptions import AuthError

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
