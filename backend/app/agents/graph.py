import logging
from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.planner import planner_node
from app.agents.retrieval import retrieval_node
from app.agents.summarizer import summarizer_node
from app.agents.verification import verification_node
from app.agents.memory import memory_node

logger = logging.getLogger(__name__)

def verification_routing_edge(state: AgentState):
    """
    Conditional routing edge deciding whether to proceed to memory storage or loop back for a retry.
    """
    passed = state.get("verification_passed", True)
    retries = state.get("retries", 0)
    
    if passed:
        logger.info("Verification passed. Routing to Memory Agent.")
        return "memory"
    
    if retries >= 3:
        logger.warning("Verification failed but retry budget (3) is exhausted. Proceeding to Memory to output best effort.")
        return "memory"
        
    logger.info(f"Verification failed. Attempting loopback retry {retries + 1}/3.")
    return "planner"

# Build and compile the workflow graph
workflow = StateGraph(AgentState)

# 1. Add all functional nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("verification", verification_node)
workflow.add_node("memory", memory_node)

# 2. Configure edges
workflow.set_entry_point("planner")

workflow.add_edge("planner", "retrieval")
workflow.add_edge("retrieval", "summarizer")
workflow.add_edge("summarizer", "verification")

# Add conditional routing from the Verification node
workflow.add_conditional_edges(
    "verification",
    verification_routing_edge,
    {
        "memory": "memory",
        "planner": "planner"
    }
)

workflow.add_edge("memory", END)

# Compile into an executable graph
agent_graph = workflow.compile()
logger.info("LangGraph Agent Workflow successfully built and compiled.")
