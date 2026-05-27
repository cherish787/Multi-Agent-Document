import logging
from datetime import datetime
from app.services.vector_store import hybrid_retriever
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

async def retrieval_node(state: AgentState) -> dict:
    """
    Node that runs the hybrid search service for each query in the plan
    and aggregates unique results.
    """
    logger.info("Retrieval Agent started execution.")
    plan = state.get("retrieval_plan", [])
    
    current_step = {
        "agent": "Retrieval Agent",
        "action": f"Executing hybrid search for plan: {plan}",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if not plan:
        # Fallback to user query if plan is empty
        plan = [state["query"]]
        
    aggregated_docs = {}
    
    try:
        for search_query in plan:
            # Fetch top 5 chunks per sub-query
            hits = hybrid_retriever.hybrid_search(search_query, top_n=5)
            for hit in hits:
                doc_id = hit["id"]
                # De-duplicate by keeping the highest scoring RRF hit
                if doc_id not in aggregated_docs or hit["rrf_score"] > aggregated_docs[doc_id]["rrf_score"]:
                    aggregated_docs[doc_id] = hit
                    
        # Sort aggregate docs by RRF score descending
        sorted_hits = sorted(aggregated_docs.values(), key=lambda x: x["rrf_score"], reverse=True)
        # Cap final list size at top 8 to stay within context budget
        retrieved = sorted_hits[:8]
        logger.info(f"Retrieval Agent fetched {len(retrieved)} unique chunks.")
        
    except Exception as e:
        logger.error(f"Retrieval Agent failed: {e}", exc_info=True)
        retrieved = []
        
    return {
        "retrieved_documents": retrieved,
        "steps": state.get("steps", []) + [current_step]
    }
