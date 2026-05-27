import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

# System prompt directing the Planner
PLANNER_PROMPT = """
You are the **Planner Agent** for a Multi-Agent Document Assistant.
Your task is to analyze the user's current query along with the conversation history and formulate a structured search plan.
This plan must consist of 1 to 3 distinct, highly targeted search queries optimized for locating the necessary answers within the documents.
The assistant uses a hybrid search engine (ChromaDB semantic search + BM25 keyword matching), so formulate queries that cover:
1. Conceptual/semantic phrases.
2. Specific keywords, codes, dates, or numbers.

If there is verification feedback indicating that the previous search was insufficient, adjust your queries to search for different details or expand the scope.

Respond ONLY with a JSON array of strings containing the queries.
Example:
["Apple Q3 2024 revenue", "Apple Q3 financial report 2024", "Apple revenue 2024"]
"""

async def planner_node(state: AgentState) -> dict:
    """
    Node that reviews the query and generates a search plan.
    """
    logger.info("Planner Agent started execution.")
    query = state["query"]
    messages = state.get("messages", [])
    retries = state.get("retries", 0)
    feedback = state.get("verification_feedback")
    
    # Trace step
    step_msg = f"Planning search strategy (Attempt {retries + 1})."
    if feedback:
        step_msg += f" Feedback: {feedback}"
        
    current_step = {
        "agent": "Planner Agent",
        "action": step_msg,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    # Compile prompt content
    history_str = ""
    for msg in messages[-4:]:  # Include last 4 messages for context
        history_str += f"{msg['role'].upper()}: {msg['content']}\n"
        
    prompt_content = f"User Query: {query}\n"
    if history_str:
        prompt_content += f"\nConversation History:\n{history_str}"
    if feedback:
        prompt_content += f"\nVerification Feedback from last attempt:\n{feedback}"
        
    prompt_content += "\nProvide the search queries as a JSON object with a 'queries' key, which maps to a list of strings."
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=prompt_content)
        ])
        
        parsed = json.loads(response.content)
        queries = parsed.get("queries", [query])
        logger.info(f"Planner Agent formulated queries: {queries}")
        
    except Exception as e:
        logger.error(f"Planner Agent failed: {e}", exc_info=True)
        queries = [query]
        
    # We return the modifications to the state
    return {
        "retrieval_plan": queries,
        "steps": state.get("steps", []) + [current_step]
    }
