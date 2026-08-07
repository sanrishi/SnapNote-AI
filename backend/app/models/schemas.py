from enum import Enum
from pydantic import BaseModel
from typing import Optional


class ExtractionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    DIAGRAM = "diagram"


class TopicInfo(BaseModel):
    title: str = ""
    is_probable: bool = False


class VisibleContent(BaseModel):
    headings: list[str] = []
    equations: list[str] = []
    labels: list[str] = []
    statements: list[str] = []


class FormulaEntry(BaseModel):
    formula: str = ""
    explanation: str = ""
    uncertain_symbols: list[str] = []


class DiagramInterpretation(BaseModel):
    present: bool = False
    visible_elements: list[str] = []
    likely_interpretation: list[str] = []


class StudyNotes(BaseModel):
    topic: TopicInfo = TopicInfo()
    visible_content: VisibleContent = VisibleContent()
    study_notes: list[str] = []
    simple_explanation: str = ""
    formula_box: list[FormulaEntry] = []
    diagram_interpretation: DiagramInterpretation = DiagramInterpretation()
    uncertainties: list[str] = []
    key_takeaway: str = ""


class ExtractionResponse(BaseModel):
    type: ExtractionType
    markdown: str
    imageUrl: Optional[str] = None
    tags: list[str] = []
    creditsUsed: int
    studyNotes: Optional[StudyNotes] = None


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


class DeviceAuthRequest(BaseModel):
    deviceId: str


class DeviceAuthResponse(BaseModel):
    deviceId: str
    creditsRemaining: int
    creditsUsed: int
    plan: str = "free"


class RevisionGuide(BaseModel):
    why_it_matters: str = ""
    intuition: str = ""
    common_mistakes: list[str] = []
    thirty_second_revision: str = ""
    analogy: str = ""


class RevisionResponse(BaseModel):
    revision_guide: RevisionGuide
    creditsUsed: int


class DiagramResult(BaseModel):
    markdown: str
