from src.met_api import fetch_object
from src.schema import ArtworkDossier, Artworkassessment
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from langchain_openai import ChatOpenAI

import csv
import os


from typing import TypedDict, Optional
from src.schema import ArtworkDossier


class ContradictionCheck(BaseModel):
    date_conflict: bool
    department_culture_mismatch: bool
    medium_anachronism: bool
    reason: str
    confidence: str

model = ChatOpenAI(model= "gpt-4o-mini").with_structured_output(ContradictionCheck)

class DossierState(TypedDict):
    object_id: int
    raw_data: Optional[dict]
    error: Optional[str]
    dossier: Optional[ArtworkDossier]
    assessment: Optional[Artworkassessment]


REVIEW_QUEUE_PATH = "../data/processed/review_queue.csv"

def flag_for_review_node(state: DossierState) -> dict:
    dossier = state["dossier"]
    assessment = state["assessment"]
    file_exists = os.path.exists(REVIEW_QUEUE_PATH)
    with open(REVIEW_QUEUE_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["object_id", "title", "confidence", "reason"])
        writer.writerow([dossier.object_id, dossier.title, assessment.confidence, assessment.reason])
    return {}


def assess_node(state: DossierState) -> dict:
    dossier = state["dossier"]
    prompt = (
        f"Artwork record:\n{dossier.model_dump_json(indent=2)}\n\n"
        "Check this record for exactly three specific problems. For each, answer only "
        "based on what's explicitly contradictory in the data itself:\n\n"
        "1. date_conflict: does the record state two different dates/centuries for the "
        "same object that conflict with each other?\n"
        "2. department_culture_mismatch: is the department completely unrelated to the "
        "stated culture (e.g. department 'Egyptian Art' but culture 'Japan')?\n"
        "3. medium_anachronism: could the stated medium not physically have existed at "
        "the stated date (e.g. a modern synthetic material dated to the 1500s)?\n\n"
        "A missing artist, 'unidentified artist', a broad date range, or a plain/generic "
        "medium name are normal cataloging and do NOT count toward any of the three checks "
        "above. Answer each as strictly true/false based only on the specific test above."
    )
    check = model.invoke(prompt)
    flag = check.date_conflict or check.department_culture_mismatch or check.medium_anachronism
    assessment = Artworkassessment(
        flag=flag,
        reason=check.reason,
        confidence=check.confidence,
    )
    return {"assessment": assessment}


def route_after_assess(state: DossierState) -> str:
    decision = "flagged" if state["assessment"].flag else "clean"
    print(f"Object {state['object_id']}: routing -> {decision}")
    return decision

graph = StateGraph(DossierState)


def fetch_node(state: DossierState):
    response = fetch_object(state["object_id"])
    if response["status_code"] != 200 or response["error"]:
        return {"error": f"Object {state['object_id']}: API error — {response['error']}"}
    return {"raw_data": response["data"]}


def validate_node(state: DossierState) -> dict:
    if state.get("error"):
        return {}  # fetch already failed — nothing to validate, let routing skip ahead
    data = state["raw_data"]
    if not data.get("isPublicDomain"):
        return {"error": f"Object {state['object_id']}: not public domain"}
    if not data.get("primaryImage"):
        return {"error": f"Object {state['object_id']}: no usable image"}
    return {}


def format_node(state: DossierState) -> dict:
    data = state["raw_data"]
    dossier = ArtworkDossier(
        object_id=data["objectID"],
        title=data.get("title") or "Untitled",
        artist=data.get("artistDisplayName") or None,
        date=data.get("objectDate") or None,
        medium=data.get("medium") or None,
        department=data["department"],
        credit_line=data.get("creditLine") or None,
        image_url=data["primaryImage"],
        is_public_domain=data["isPublicDomain"],
    )
    return {"dossier": dossier}


graph.add_node("fetch", fetch_node)
graph.add_node("validate_node", validate_node)
graph.add_node("format_node", format_node)
graph.add_node("assessment", assess_node)
graph.add_node("flag_for_review", flag_for_review_node)



def route_after_validate(state: DossierState) -> str:
    return "format_node" if not state.get("error") else "__end__" 

graph.set_entry_point("fetch")
graph.add_edge("fetch", "validate_node")

graph.add_conditional_edges("validate_node", route_after_validate, {"format_node": "format_node", "__end__": END})

graph.add_edge("format_node", "assessment")
graph.add_conditional_edges(
    "assessment",
    route_after_assess,
    {"flagged": "flag_for_review", "clean": END},
)
graph.add_edge("flag_for_review", END)


dossier_graph = graph.compile()
