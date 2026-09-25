from pydantic import BaseModel
from typing import Optional

class CuratorDossier(BaseModel):
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


class SelectionDraft(BaseModel):
    selected_object_ids: list[int]
    exhibit_title: str
    narrative: str