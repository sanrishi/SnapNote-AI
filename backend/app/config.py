from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    APP_NAME: str = "SnapNote AI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    FIREBASE_CREDENTIALS_PATH: str = "firebase-credentials.json"
    GEMINI_API_KEY: str = ""

    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = "snapnote-diagrams"
    R2_PUBLIC_URL: str = "https://pub-xxxxx.r2.dev"

    FREE_CREDITS_MONTHLY: int = 50
    TEXT_CREDIT_COST: int = 1
    DIAGRAM_CREDIT_COST: int = 5

    DAILY_REQ_LIMIT: int = 50
    RATE_LIMIT_PER_MIN: int = 10
    MAX_IMAGE_SIZE_MB: int = 10
    CREDITS_DB_PATH: str = "credits.db"

    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""

    CREDIT_PACKS: dict = {
        "starter": {"credits": 50, "price_paise": 4900},
        "popular": {"credits": 120, "price_paise": 9900},
        "pro": {"credits": 300, "price_paise": 19900},
        "unlimited": {"credits": 1000, "price_paise": 49900},
    }


settings = Settings()
