from typing import TypedDict, Optional
from laya import Router

from src.supervisor.schema import RouterDecision
from src.director.workflow import director_multiagent_graph
from src.curator.workflow import curator_multiagent_graph

from langgraph.types import interrupt


from langgraph.graph import StateGraph, START, END

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import os
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

class SupervisorState(TypedDict):
    user_input: str
    routing: Optional[RouterDecision]
    needs_clarification: Optional[bool]
    clarification_question: Optional[str]
    result: Optional[dict]

router = Router(preload=True)

ROUTING_QUESTIONS = {
    "capability": {
        "type": "choice",
        "instructions": "Which Atelier capability should handle this request?",
        "criteria": {
            "curator": "building a themed exhibition or curated selection of multiple real pieces from the Met's collection around a concept or brief",
            "director": "generating a brand new, original piece of art inspired by a mood, theme, or setting the user describes — something that doesn't exist yet",
        },
    }
}

CONFIDENCE_THRESHOLD = 0.6

def route_node(state: SupervisorState) -> dict:
    result = router.predict(state["user_input"], ROUTING_QUESTIONS)
    answer = result["answers"]["capability"]
    decision = RouterDecision(
        capability=answer["choice"],
        confidence=answer["answer_confidence"],
        probabilities=answer["probabilities"],
    )
    return {
        "routing": decision,
        "needs_clarification": decision.confidence < CONFIDENCE_THRESHOLD,
    }



def clarify_node(state: SupervisorState) -> dict:
    probs = state["routing"].probabilities
    question = (
        "Just to make sure I route this correctly — are you looking to:\n"
        "1. identify or learn about a specific existing artwork,\n"
        "2. curate a themed selection of real pieces from the collection, or\n"
        "3. generate a brand new piece of art?\n"
        "Could you say a bit more about what you're after?"
    )
    answer = interrupt({"question": question, "probabilities": probs})
    return {
        "user_input": state["user_input"] + "\n\n" + answer,
        "clarified": True,
    }


def route_after_routing(state: SupervisorState) -> str:
    if state["needs_clarification"] and not state.get("clarified"):
        return "clarify"
    return state["routing"].capability  # "detective" | "curator" | "director"


def detective_node(state): return {"result": {"capability": "detective"}}
def curator_node(state: SupervisorState, config) -> dict:
    child_config = {"configurable": {"thread_id": f"{config['configurable']['thread_id']}-curator"}}
    snapshot = curator_multiagent_graph.get_state(child_config)
    pending_interrupts = [t for t in snapshot.tasks if t.interrupts]

    if pending_interrupts:
        payload = pending_interrupts[0].interrupts[0].value
        decision = interrupt(payload)
        result = curator_multiagent_graph.invoke(Command(resume=decision), config=child_config)
    else:
        child_input = {
            "user_query": state["user_input"],
            "top_k": 25,
            "attempt": 0,
            "max_attempts": 5,
        }
        result = curator_multiagent_graph.invoke(child_input, config=child_config)
        if "__interrupt__" in result:
            decision = interrupt(result["__interrupt__"][0].value)
            result = curator_multiagent_graph.invoke(Command(resume=decision), config=child_config)

    return {"result": {"capability": "curator", **result}}



def director_node(state: SupervisorState, config) -> dict:
    child_config = {"configurable": {"thread_id": f"{config['configurable']['thread_id']}-director"}}
    snapshot = director_multiagent_graph.get_state(child_config)
    pending_interrupts = [t for t in snapshot.tasks if t.interrupts]

    if pending_interrupts:
        payload = pending_interrupts[0].interrupts[0].value
        decision = interrupt(payload)
        result = director_multiagent_graph.invoke(Command(resume=decision), config=child_config)
    else:
        child_input = {
            "user_description": state["user_input"],
            "top_k": 25,
            "attempt": 0,
            "max_attempts": 3,
        }
        result = director_multiagent_graph.invoke(child_input, config=child_config)
        if "__interrupt__" in result:
            decision = interrupt(result["__interrupt__"][0].value)
            result = director_multiagent_graph.invoke(Command(resume=decision), config=child_config)

    return {"result": {"capability": "director", **result}}

graph = StateGraph(SupervisorState)
graph.add_node("route", route_node)
graph.add_node("clarify", clarify_node)

graph.add_node("curator", curator_node)
graph.add_node("director", director_node)

graph.add_edge(START, "route")
graph.add_conditional_edges("route", route_after_routing, {
    "clarify": "clarify",  "curator": "curator", "director": "director",
})
graph.add_edge("clarify", "route")

graph.add_edge("curator", END)
graph.add_edge("director", END)



supervisor_graph = graph.compile(checkpointer=MemorySaver())


