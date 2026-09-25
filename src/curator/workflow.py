from typing import TypedDict, Optional
from src.curator.schema import CuratorDossier, SelectionDraft, EditorVerdict
from src.curator.index import search, get_metadata

import pandas as pd

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver


class CuratorState(TypedDict):
    user_query: str
    top_k: int
    search_results: Optional[list[tuple[int, float]]]
    candidates: Optional[list[CuratorDossier]]
    editor_verdict: Optional[EditorVerdict]
    selection_draft: Optional[SelectionDraft]
    attempt: int
    max_attempts: int
    best_draft: Optional[SelectionDraft]
    best_score: Optional[int]
    human_decision: Optional[str]


selection_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(SelectionDraft)

def writer_node(state: CuratorState) -> dict:
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



editor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(EditorVerdict)

def editor_node(state: CuratorState) -> dict:
    draft = state["selection_draft"]
    candidates = state["candidates"]
    candidate_lookup = {c.object_id: c for c in candidates}

    selected_lines = []
    for oid in draft.selected_object_ids:
        c = candidate_lookup[oid]
        selected_lines.append(
            f"object_id: {c.object_id} | title: {c.title} | culture: {c.culture} | "
            f"period: {c.period} | date: {c.date} | medium: {c.medium} | "
            f"department: {c.department} | tags: {c.tags}"
        )
    selected_block = "\n".join(selected_lines)

    prompt = (
        f"You are a strict art exhibition critic reviewing a curator's draft exhibition.\n\n"
        f"The user's requested theme: \"{state['user_query']}\"\n\n"
        f"Exhibit title: {draft.exhibit_title}\n\n"
        f"Selected pieces (ground truth data — use ONLY this to check factual claims):\n{selected_block}\n\n"
        f"Narrative written by the curator:\n{draft.narrative}\n\n"
        "Judge this draft against these fixed criteria:\n"
        "1. Grounding: every factual claim in the narrative (culture, period, medium, date, etc.) must be "
        "verifiable against the ground-truth data above. Flag any invented or unsupported claim.\n"
        "2. Completeness: every single selected object_id above must be explicitly discussed by name in "
        "the narrative. Flag any that are missing.\n"
        "3. Coherence: the selected pieces must genuinely connect to each other and to the requested theme "
        "— not just a superficial word match. Flag any piece that feels shoehorned in.\n"
        "4. Writing quality: the narrative should read as a genuine curatorial statement, not generic or "
        "cliché filler text.\n\n"
        "Give a score from 1 to 10, a boolean approved (true only if the score is 8 or higher and there are "
        "no grounding or completeness issues), and a list of specific issues (empty if approved)."
    )

    verdict = editor_llm.invoke(prompt)
    return {"editor_verdict": verdict}


def track_best_node(state: CuratorState) -> dict:
    verdict = state["editor_verdict"]
    draft = state["selection_draft"]
    attempt = state["attempt"] + 1

    best_score = state.get("best_score")
    best_draft = state.get("best_draft")

    if best_score is None or verdict.score > best_score:
        best_score = verdict.score
        best_draft = draft

    return {"attempt": attempt, "best_score": best_score, "best_draft": best_draft}

def route_after_track(state: CuratorState) -> str:
    if state["editor_verdict"].approved:
        return "human_review"
    if state["attempt"] >= state["max_attempts"]:
        return "human_review"
    return "retry"

def human_review_node(state: CuratorState) -> dict:
    payload = {
        "exhibit_title": state["best_draft"].exhibit_title,
        "narrative": state["best_draft"].narrative,
        "selected_object_ids": state["best_draft"].selected_object_ids,
        "editor_score": state["best_score"],
        "editor_approved": state["editor_verdict"].approved,
        "editor_issues": state["editor_verdict"].issues,
    }
    decision = interrupt(payload)
    return {"human_decision": decision}

graph = StateGraph(CuratorState)
graph.add_node("search", search_node)
graph.add_node("build_candidates", build_candidates_node)
graph.add_node("write", writer_node)          # renamed from selection_node
graph.add_node("edit", editor_node)
graph.add_node("track_best", track_best_node)
graph.add_node("human_review", human_review_node)

graph.add_edge(START, "search")
graph.add_edge("search", "build_candidates")
graph.add_edge("build_candidates", "write")
graph.add_edge("write", "edit")
graph.add_edge("edit", "track_best")


graph.add_conditional_edges(
    "track_best",
    route_after_track,
    {"retry": "write", "human_review": "human_review"},
)
graph.add_edge("human_review", END)


curator_multiagent_graph = graph.compile(checkpointer=MemorySaver())
