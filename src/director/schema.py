from typing import Optional
from pydantic import BaseModel

class ReferencePiece(BaseModel):
    object_id: int
    title: str
    culture: Optional[str] = None
    period: Optional[str] = None
    artist: Optional[str] = None
    date: Optional[str] = None
    medium: Optional[str] = None
    classification: Optional[str] = None
    department: str
    tags: Optional[str] = None
    credit_line: Optional[str] = None
    link_resource: Optional[str] = None
    similarity_score: float

class DirectorDraft(BaseModel):
    title: str
    image_prompt: str
    description: str

class VisionVerdict(BaseModel):
    approved: bool
    score: int
    issues: list[str]