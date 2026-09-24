from src.met_api import fetch_object
from src.schema import ArtworkDossier
from langgraph.graph import StateGraph, START, END


from typing import TypedDict, Optional
from src.schema import ArtworkDossier

class DossierState(TypedDict):
    object_id: int
    raw_data: Optional[dict]
    error: Optional[str]
    dossier: Optional[ArtworkDossier]


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



def route_after_validate(state: DossierState) -> str:
    return "format_node" if not state.get("error") else "__end__" 

graph.set_entry_point("fetch")
graph.add_edge("fetch", "validate_node")

graph.add_conditional_edges("validate_node", route_after_validate, {"format_node": "format_node", "__end__": END})

dossier_graph = graph.compile()
