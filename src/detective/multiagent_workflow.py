from typing import TypedDict, Optional

from src.detective.agents import date_specialist, culture_specialist, medium_specialist, correction_specialist, DateVerdict, CultureVerdict, MediumVerdict, CorrectionProposal
from src.detective.workflow import fetch_node, validate_node, format_node, flag_for_review_node, route_after_validate
from src.detective.schema import Artworkassessment, ArtworkDossier
from langgraph.graph import StateGraph, END, START

from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

class MultiAgentState(TypedDict):
    object_id: int
    raw_data: Optional[dict]
    error: Optional[str]
    dossier: Optional["ArtworkDossier"]
    date_verdict: Optional[DateVerdict]
    culture_verdict: Optional[CultureVerdict]
    medium_verdict: Optional[MediumVerdict]
    assessment: Optional[Artworkassessment]
    correction: Optional[CorrectionProposal]
    human_decision: Optional[str]


def date_node(state: MultiAgentState) -> dict:
    return {"date_verdict": date_specialist(state["dossier"])}

def culture_node(state: MultiAgentState) -> dict:
    return {"culture_verdict": culture_specialist(state["dossier"])}

def medium_node(state: MultiAgentState) -> dict:
    return {"medium_verdict": medium_specialist(state["dossier"])}



def supervisor_node(state: MultiAgentState) -> dict:
        date_verdict = state["date_verdict"]
        culture_verdict = state["culture_verdict"]
        medium_verdict = state["medium_verdict"]

        flag = (
            date_verdict.date_conflict
            or culture_verdict.department_culture_mismatch
            or medium_verdict.medium_anachronism
        )

        reasons = []
        if date_verdict.date_conflict:
            reasons.append(f"[date] {date_verdict.reason}")
        if culture_verdict.department_culture_mismatch:
            reasons.append(f"[culture] {culture_verdict.reason}")
        if medium_verdict.medium_anachronism:
            reasons.append(f"[medium] {medium_verdict.reason}")

        reason = " ".join(reasons) if reasons else "No contradiction found by any specialist."
        return {"assessment": Artworkassessment(flag=flag, reason=reason, confidence="High")}



def route_after_supervisor(state: MultiAgentState) -> str:
    return "flagged" if state["assessment"].flag else "clean"

def propose_correction_node(state: MultiAgentState) -> dict:
    proposal = correction_specialist(state["dossier"], state["assessment"].reason)
    return {"correction": proposal}




def human_review_node(state: MultiAgentState) -> dict:
    decision = interrupt({
        "object_id": state["object_id"],
        "flag_reason": state["assessment"].reason,
        "can_propose_fix": state["correction"].can_propose_fix,
        "proposed_fix": state["correction"].proposed_fix,
        "rationale": state["correction"].rationale,
    })
    return {"human_decision": decision}


graph = StateGraph(MultiAgentState)
graph.add_node("fetch", fetch_node)
graph.add_node("validate", validate_node)
graph.add_node("format_node", format_node)
graph.add_node("date_check", date_node)
graph.add_node("culture_check", culture_node)
graph.add_node("medium_check", medium_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("flag_for_review", flag_for_review_node)
graph.add_node("propose_correction", propose_correction_node)

graph.set_entry_point("fetch")
graph.add_edge("fetch", "validate")
graph.add_conditional_edges("validate", route_after_validate, {"format_node": "format_node", "__end__": END})

graph.add_edge("format_node", "date_check")
graph.add_edge("format_node", "culture_check")
graph.add_edge("format_node", "medium_check")

graph.add_edge("date_check", "supervisor")
graph.add_edge("culture_check", "supervisor")
graph.add_edge("medium_check", "supervisor")

graph.add_conditional_edges("supervisor", route_after_supervisor, {"flagged": "propose_correction", "clean": END})
graph.add_edge("propose_correction", "flag_for_review")
graph.add_edge("flag_for_review", END)

multi_agent_graph = graph.compile()





review_graph = StateGraph(MultiAgentState)
review_graph.add_node("date_check", date_node)
review_graph.add_node("culture_check", culture_node)
review_graph.add_node("medium_check", medium_node)
review_graph.add_node("supervisor", supervisor_node)
review_graph.add_node("propose_correction", propose_correction_node)
review_graph.add_node("human_review", human_review_node)
review_graph.add_node("flag_for_review", flag_for_review_node)

review_graph.add_edge(START, "date_check")
review_graph.add_edge(START, "culture_check")
review_graph.add_edge(START, "medium_check")
review_graph.add_edge("date_check", "supervisor")
review_graph.add_edge("culture_check", "supervisor")
review_graph.add_edge("medium_check", "supervisor")
review_graph.add_conditional_edges("supervisor", route_after_supervisor, {"flagged": "propose_correction", "clean": END})
review_graph.add_edge("propose_correction", "human_review")
review_graph.add_edge("human_review", "flag_for_review")
review_graph.add_edge("flag_for_review", END)

review_graph_compiled = review_graph.compile(checkpointer=MemorySaver())