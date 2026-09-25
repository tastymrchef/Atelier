from pydantic import BaseModel
from typing import Optional


class ArtworkDossier(BaseModel):
    """The clean, agent-facing shape of one artwork.
    Every later agent reads artworks through this contract — nothing else."""
    object_id: int
    title: str
    artist: Optional[str] = None
    date: Optional[str] = None
    medium: Optional[str] = None
    department: str
    credit_line: Optional[str] = None
    image_url: str
    is_public_domain: bool


class Artworkassessment(BaseModel):
    flag: bool
    reason: str
    confidence: str