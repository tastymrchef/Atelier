from typing import TypedDict, Optional
from src.curator.schema import CuratorDossier, SelectionDraft
from src.curator.index import search, get_metadata

import pandas as pd

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class CuratorState(TypedDict):
    user_query: str
    top_k: int
    search_results: Optional[list[tuple[int, float]]]
    candidates: Optional[list[CuratorDossier]]
    selection_draft: Optional[SelectionDraft]





selection_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(SelectionDraft)

def selection_node(state: CuratorState) -> dict:
    candidates = state["candidates"]

    candidate_lines = []
    for c in candidates:
        candidate_lines.append(
            f"object_id: {c.object_id} | title: {c.title} | culture: {c.culture} | "
            f"period: {c.period} | date: {c.date} | medium: {c.medium} | "
            f"department: {c.department} | tags: {c.tags}"
        )
    candidate_block = "\n".join(candidate_lines)

    prompt = (
        f"A user wants a themed exhibition built around this idea:\n\"{state['user_query']}\"\n\n"
        f"Here are {len(candidates)} candidate artworks retrieved as possibly relevant:\n\n"
        f"{candidate_block}\n\n"
        "Select between 5 and 10 of these object_ids that genuinely belong together as a coherent "
        "exhibition — not just ones that superficially share a word, but ones that actually connect "
        "to the user's theme and to each other. It's fine, and expected, to leave out candidates "
        "that don't fit well, even if many were retrieved.\n\n"
        "Give the exhibition a short title, and write a paragraph (3-5 sentences) explaining the "
        "connection between the selected pieces — ground every claim in the actual culture, period, "
        "medium, or date fields shown above. Do not invent facts not present in the candidate list.\n\n"
        "Only select object_ids that appear in the list above."
        "Your narrative must explicitly discuss every single object_id you selected — do not select"
        "a piece and then omit it from the write-up. If you cannot say something concrete about why a"
        "candidate belongs, do not select it in the first place."
    )

    draft = selection_llm.invoke(prompt)

    valid_ids = {c.object_id for c in candidates}
    verified_ids = [oid for oid in draft.selected_object_ids if oid in valid_ids]

    return {"selection_draft": SelectionDraft(
        selected_object_ids=verified_ids,
        exhibit_title=draft.exhibit_title,
        narrative=draft.narrative,
    )}


def search_node(state: CuratorState):
    results = search(state["user_query"], top_k=state.get("top_k", 25))
    return {"search_results": results}


def _clean(value):
    return value if pd.notna(value) else None

def build_candidates_node(state: CuratorState) -> dict:
    candidates = []
    for object_id, score in state["search_results"]:
        row = get_metadata(object_id)
        title = row["Title"] if pd.notna(row["Title"]) else row["Object Name"]
        candidates.append(CuratorDossier(
            object_id=object_id,
            title=title,
            culture=_clean(row["Culture"]),
            period=_clean(row["Period"]),
            artist=_clean(row["Artist Display Name"]),
            date=_clean(row["Object Date"]),
            medium=_clean(row["Medium"]),
            classification=_clean(row["Classification"]),
            department=row["Department"],
            tags=_clean(row["Tags"]),
            credit_line=_clean(row["Credit Line"]),
            link_resource=_clean(row["Link Resource"]),
            similarity_score=score,
        ))
    return {"candidates": candidates}

graph = StateGraph(CuratorState)
graph = StateGraph(CuratorState)
graph.add_node("search", search_node)
graph.add_node("build_candidates", build_candidates_node)
graph.add_node("select", selection_node)
graph.add_edge(START, "search")
graph.add_edge("search", "build_candidates")
graph.add_edge("build_candidates", "select")
graph.add_edge("select", END)

curator_workflow_graph = graph.compile()
