# graph/builder.py
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from graph.nodes import (
    collect_results,
    dispatch_tasks,
    critique_answer,
    finalize,
    init,
    plan_tasks,
    research_task,
    review_critique,
    review_plan,
    route_after_plan_review,
    rewrite_query,
    route_after_critique,
    synthesize_answer,
)
from graph.state import ResearchState


def dispatch_research_tasks(state: ResearchState) -> list[Send]:
    return [
        Send(
            "research_task",
            {
                "task": t,
                "context_id": state["context_id"],
            },
        )
        for t in state["tasks"]
    ]


def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    g.add_node("init", init)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("plan_tasks", plan_tasks)
    g.add_node("review_plan", review_plan)
    g.add_node("dispatch_tasks", dispatch_tasks)
    g.add_node("research_task", research_task)
    g.add_node("collect_results", collect_results)
    g.add_node("synthesize_answer", synthesize_answer)
    g.add_node("critique_answer", critique_answer)
    g.add_node("review_critique", review_critique)
    g.add_node("finalize", finalize)

    g.add_edge(START, "init")
    g.add_edge("init", "rewrite_query")
    g.add_edge("rewrite_query", "plan_tasks")
    g.add_edge("plan_tasks", "review_plan")
    g.add_conditional_edges(
        "review_plan",
        route_after_plan_review,
        {"plan_tasks": "plan_tasks", "dispatch_tasks": "dispatch_tasks"},
    )
    g.add_conditional_edges("dispatch_tasks", dispatch_research_tasks, ["research_task"])
    g.add_edge("research_task", "collect_results")
    g.add_edge("collect_results", "synthesize_answer")
    g.add_edge("synthesize_answer", "critique_answer")
    g.add_edge("critique_answer", "review_critique")
    g.add_conditional_edges(
        "review_critique",
        route_after_critique,
        {"plan_tasks": "plan_tasks", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)

    return g
