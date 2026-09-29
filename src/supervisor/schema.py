from typing import Literal
from pydantic import BaseModel

class RouterDecision(BaseModel):
    capability: Literal["detective", "curator", "director"]
    confidence: float
    probabilities: dict[str, float]