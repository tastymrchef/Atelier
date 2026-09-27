from typing import TypedDict, Optional
from src.director.schema import ReferencePiece, DirectorDraft, VisionVerdict

from langchain_openai import ChatOpenAI

import pandas as pd
from src.shared.index import search, get_metadata


from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


from langgraph.types import interrupt

import base64
import json
import boto3

class DirectorState(TypedDict):
    user_description: str
    top_k: int
    search_results: Optional[list[tuple[int, float]]]
    references: Optional[list[ReferencePiece]]
    draft: Optional[DirectorDraft]
    image_bytes: Optional[bytes]
    verdict: Optional[VisionVerdict]
    attempt: int
    max_attempts: int
    best_draft: Optional[DirectorDraft]
    best_image_bytes: Optional[bytes]
    best_score: Optional[int]
    human_decision: Optional[str]


def search_node(state: DirectorState) -> dict:
    results = search(state["user_description"], top_k=state.get("top_k", 25))
    return {"search_results": results}

def _clean(value):
    return value if pd.notna(value) else None

def build_references_node(state: DirectorState) -> dict:
    references = []
    for object_id, score in state["search_results"]:
        row = get_metadata(object_id)
        title = row["Title"] if pd.notna(row["Title"]) else row["Object Name"]
        references.append(ReferencePiece(
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
    return {"references": references}


prompt_engineer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7).with_structured_output(DirectorDraft)

def prompt_engineer_node(state: DirectorState) -> dict:
    references = state["references"]
    reference_lines = [
        f"object_id: {r.object_id} | title: {r.title} | culture: {r.culture} | "
        f"period: {r.period} | date: {r.date} | medium: {r.medium} | "
        f"department: {r.department} | tags: {r.tags}"
        for r in references
    ]
    reference_block = "\n".join(reference_lines)

    feedback_block = ""
    verdict = state.get("verdict")
    if verdict is not None and not verdict.approved:
        issues_text = "\n".join(f"- {issue}" for issue in verdict.issues)
        feedback_block = (
            f"\n\nYour previous attempt was reviewed and rejected for these specific reasons:\n"
            f"{issues_text}\n\n"
            "Revise your prompt and description to directly address every issue above — don't just "
            "repeat the same approach with minor wording changes."
        )

    prompt = (
        f"A user wants a new piece of art generated based on this description:\n\"{state['user_description']}\"\n\n"
        f"Here are {len(references)} real pieces from the Met's collection retrieved as thematically related "
        f"to this description — use them for inspiration and stylistic grounding, not as things to literally copy:\n\n"
        f"{reference_block}\n\n"
        "Write a text-to-image generation prompt for a painting or fine-art piece — not a photograph, and not a "
        "generic digital illustration. Explicitly specify an artistic medium and technique (for example: oil "
        "painting, watercolor, woodblock print, fresco, ink wash, tempera) — draw this from the mediums "
        "represented in the references above where it fits the user's theme, rather than defaulting to a "
        "generic style. The prompt should capture the user's requested mood, theme, and setting, and describe "
        "composition, lighting, and brushwork or technique concretely enough for an image model to act on.\n\n"
        "Also give the piece a short title, and write a 2-4 sentence description of what the image depicts and "
        "why you made the creative choices you did, referencing which real pieces (by title) inspired which "
        f"aspects, where applicable.{feedback_block}"
    )

    draft = prompt_engineer_llm.invoke(prompt)
    return {"draft": draft}


vision_critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(VisionVerdict)

def vision_critic_node(state: DirectorState) -> dict:
    draft = state["draft"]
    references = state["references"]
    image_b64 = base64.b64encode(state["image_bytes"]).decode("utf-8")

    reference_lines = [
        f"object_id: {r.object_id} | title: {r.title} | culture: {r.culture} | "
        f"period: {r.period} | medium: {r.medium}"
        for r in references
    ]
    reference_block = "\n".join(reference_lines)

    text_prompt = (
        f"A user requested a generated artwork based on this description:\n\"{state['user_description']}\"\n\n"
        f"Real Met collection pieces used as inspiration:\n{reference_block}\n\n"
        f"The generated piece's title: {draft.title}\n"
        f"The claimed description of the piece: {draft.description}\n\n"
        "Look at the attached generated image and judge it against these fixed criteria:\n"
        "1. Mood/theme fidelity: does the image genuinely capture the user's requested mood, theme, and setting?\n"
        "2. Collection grounding: does the image reasonably draw on the style/medium/culture of the real "
        "reference pieces listed above, rather than looking generic or unrelated to them?\n"
        "3. Description accuracy: does the claimed description actually match what's visible in the image? "
        "Flag anything the description claims that isn't actually depicted.\n"
        "4. Artistic quality: does this read as a painting/fine-art piece with real composition and technique, "
        "not a broken, malformed, or purely photographic image?\n\n"
        "Give a score from 1 to 10, a boolean approved (true only if the score is 8 or higher and there are no "
        "description-accuracy issues), and a list of specific issues (empty if approved)."
    )

    message = HumanMessage(content=[
        {"type": "text", "text": text_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    ])

    verdict = vision_critic_llm.invoke([message])
    return {"verdict": verdict}


def track_best_node(state: DirectorState) -> dict:
    verdict = state["verdict"]
    attempt = state["attempt"] + 1

    best_score = state.get("best_score")
    best_draft = state.get("best_draft")
    best_image_bytes = state.get("best_image_bytes")

    if best_score is None or verdict.score > best_score:
        best_score = verdict.score
        best_draft = state["draft"]
        best_image_bytes = state["image_bytes"]

    return {
        "attempt": attempt,
        "best_score": best_score,
        "best_draft": best_draft,
        "best_image_bytes": best_image_bytes,
    }

def route_after_track(state: DirectorState) -> str:
    if state["verdict"].approved:
        return "human_review"
    if state["attempt"] >= state["max_attempts"]:
        return "human_review"
    return "retry"



def human_review_node(state: DirectorState) -> dict:
    with open("director_output.png", "wb") as f:
        f.write(state["best_image_bytes"])

    payload = {
        "title": state["best_draft"].title,
        "description": state["best_draft"].description,
        "image_path": "director_output.png",
        "critic_score": state["best_score"],
        "critic_approved": state["verdict"].approved,
        "critic_issues": state["verdict"].issues,
    }
    decision = interrupt(payload)
    return {"human_decision": decision}


bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

def generate_image_node(state: DirectorState) -> dict:
    draft = state["draft"]

    request_body = json.dumps({
        "prompt": draft.image_prompt,
        "aspect_ratio": "1:1",
        "output_format": "png",
    })

    response = bedrock_client.invoke_model(
        modelId="stability.stable-image-core-v1:1",
        body=request_body,
    )

    response_body = json.loads(response["body"].read())

    if response_body["finish_reasons"][0] is not None:
        raise RuntimeError(f"Image generation was filtered: {response_body['finish_reasons'][0]}")

    base64_image = response_body["images"][0]
    image_bytes = base64.b64decode(base64_image)

    return {"image_bytes": image_bytes}



graph = StateGraph(DirectorState)
graph.add_node("search", search_node)
graph.add_node("build_references", build_references_node)
graph.add_node("prompt_engineer", prompt_engineer_node)
graph.add_node("generate_image", generate_image_node)
graph.add_node("vision_critic", vision_critic_node)
graph.add_node("track_best", track_best_node)
graph.add_node("human_review", human_review_node)

graph.add_edge(START, "search")
graph.add_edge("search", "build_references")
graph.add_edge("build_references", "prompt_engineer")
graph.add_edge("prompt_engineer", "generate_image")
graph.add_edge("generate_image", "vision_critic")
graph.add_edge("vision_critic", "track_best")

graph.add_conditional_edges(
    "track_best",
    route_after_track,
    {"retry": "prompt_engineer", "human_review": "human_review"},
)
graph.add_edge("human_review", END)

director_multiagent_graph = graph.compile(checkpointer=MemorySaver())