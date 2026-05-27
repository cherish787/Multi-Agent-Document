import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

SUMMARIZER_PROMPT = """
You are the **Summarizer Agent** for a Multi-Agent Document Assistant.
Your task is to answer the user's query comprehensively and accurately, basing your answer strictly on the provided retrieved document chunks.

Rules:
1. **Grounded Answers**: Every single claim, fact, or statistic you state MUST be supported by the provided chunks. Do not hallucinate or use external training knowledge not present in the chunks.
2. **Metadata Citations**: You MUST cite the source of every fact. Use inline bracket citations indicating the filename and page number, e.g., "[Report_2024.pdf, Page 3]" or "[Sales_Memo.txt, Page 1]".
3. **Citation Format**: Put citations at the end of the sentence or clause containing the cited fact.
4. **Insufficient Info**: If the provided chunks do not contain enough information to answer the query, clearly state: "I cannot find sufficient information in the uploaded documents to answer your question." Do not attempt to guess.

Retrieved Context Chunks:
{context}
"""

async def summarizer_node(state: AgentState) -> dict:
    """
    Node that synthesizes a detailed citation-grounded response.
    """
    logger.info("Summarizer Agent started execution.")
    query = state["query"]
    docs = state.get("retrieved_documents", [])
    
    current_step = {
        "agent": "Summarizer Agent",
        "action": f"Synthesizing answer from {len(docs)} retrieved chunks.",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if not docs:
        draft = "I cannot find any uploaded documents in the system database to search. Please upload a PDF/DOCX/TXT file first."
        return {
            "draft_answer": draft,
            "steps": state.get("steps", []) + [current_step]
        }
        
    # Format context for the LLM
    context_blocks = []
    for idx, doc in enumerate(docs):
        meta = doc["metadata"]
        filename = meta.get("filename", "Unknown File")
        page = meta.get("page", 1)
        text = doc["text"]
        context_blocks.append(f"--- Chunk {idx} (File: {filename}, Page: {page}) ---\n{text}\n")
        
    context_str = "\n".join(context_blocks)
    
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.3
    )
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=SUMMARIZER_PROMPT.format(context=context_str)),
            HumanMessage(content=f"User Question: {query}\n\nDraft your grounded answer:")
        ])
        draft = response.content
        logger.info("Summarizer Agent successfully drafted the response.")
        
    except Exception as e:
        logger.error(f"Summarizer Agent failed: {e}", exc_info=True)
        draft = f"An error occurred while generating the response: {e}"
        
    return {
        "draft_answer": draft,
        "steps": state.get("steps", []) + [current_step]
    }
