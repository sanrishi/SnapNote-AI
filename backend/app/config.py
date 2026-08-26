from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    APP_NAME: str = "SnapNote AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    FIREBASE_CREDENTIALS_PATH: str = "firebase-credentials.json"
    GEMINI_API_KEY: str = ""
    IMGBB_API_KEY: str = ""

    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "snapnote-diagrams"
    R2_PUBLIC_URL: str = "https://pub-xxxxx.r2.dev"

    FREE_CREDITS_MONTHLY: int = 50
    TEXT_CREDIT_COST: int = 1
    REVISION_CREDIT_COST: int = 1
    DIAGRAM_CREDIT_COST: int = 5
    ANONYMOUS_FREE_USES: int = 50  # overridden to 1 in production via render.yaml (prevents reload abuse)

    # Pollinations image generation (Explain Visually). Empty key = anonymous
    # tier (1 req / 15s, slower); a free registered key lifts it to 1 req / 5s.
    POLLINATIONS_API_KEY: str = ""
    POLLINATIONS_MODEL: str = "sana"
    POLLINATIONS_TIMEOUT_SECONDS: float = 25.0

    DAILY_REQ_LIMIT: int = 50
    RATE_LIMIT_PER_MIN: int = 10
    MAX_IMAGE_SIZE_MB: int = 10
    CREDITS_DB_PATH: str = "credits.db"

    # Tuned for 15-20s student SLA: smaller JPEG + tighter Gemini deadlines so
    # a slow upstream fails fast (502 in 22s) instead of 504 in 60s. See
    # extract.py / vision_service.py which both respect these.
    MAX_VISION_LONG_EDGE: int = 1280
    VISION_JPEG_QUALITY: int = 75
    GEMINI_CALL_TIMEOUT_SECONDS: float = 22.0
    DIAGRAM_TIMEOUT_SECONDS: float = 28.0

    # "semantic" = Gemini outputs a structured DiagramSpec, Python renders it
    # deterministically. "legacy" = Gemini writes the SVG directly (old path).
    DIAGRAM_RENDERER_MODE: str = "semantic"

    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""

    JWT_SECRET: str = "dev-secret-change-in-prod-please-set-JWT_SECRET-env"
    JWT_EXPIRES_HOURS: int = 168

    GOOGLE_CLIENT_ID: str = ""  # for Google Identity Services ID-token verification

    CREDIT_PACKS: dict = {
        "starter": {"credits": 50, "price_paise": 4900},
        "popular": {"credits": 120, "price_paise": 9900},
        "pro": {"credits": 300, "price_paise": 19900},
        "unlimited": {"credits": 1000, "price_paise": 49900},
    }


settings = Settings()
