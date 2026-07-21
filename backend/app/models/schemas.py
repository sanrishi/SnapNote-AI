from enum import Enum
from pydantic import BaseModel
from typing import Optional


class ExtractionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    DIAGRAM = "diagram"


class ExtractionResponse(BaseModel):
    type: ExtractionType
    markdown: str
    imageUrl: Optional[str] = None
    tags: list[str] = []
    creditsUsed: int


class ExtractionContext(BaseModel):
    title: str = ""
    url: str = ""
    week: str = ""


class UserCredits(BaseModel):
    creditsRemaining: int
    creditsUsed: int
    plan: str = "free"


class GoogleAuthRequest(BaseModel):
    idToken: str


class AuthResponse(BaseModel):
    accessToken: str
    uid: str
    email: str
    name: str
    creditsRemaining: int


class DiagramResult(BaseModel):
    markdown: str
