import json
import logging
from datetime import datetime
from sqlalchemy.future import select
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings
from app.database import async_session_maker
from app.models import ConversationMemory
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

MEMORY_PROMPT = """
You are the **Memory Agent** for a Multi-Agent Document Assistant.
Your task is to maintain long-term context by updating the running conversation summary and extracting key structured facts, preferences, or topics discussed during this turn.

Inputs:
- Current User Query: {query}
- Final Grounded Assistant Answer: {answer}
- Existing Memory Summary: {existing_summary}
- Existing Extracted Facts (JSON): {existing_facts}

Rules:
1. **Running Summary**: Blend the new interaction into a concise summary of the conversation's context. Keep it under 3 paragraphs.
2. **Extracted Facts**: Identify key topics, entities, companies, dates, or formulas discussed. Update the JSON dict. Keep relevant historical keys, overwrite them if updated, or add new keys.

Respond ONLY with a JSON object containing:
- "memory_summary": "updated summary string"
- "extracted_facts": {{"key1": "val1", ...}}
"""

async def memory_node(state: AgentState) -> dict:
    """
    Node that reviews the finalized Q&A turn, summarizes conversation state,
    and updates long-term memory in PostgreSQL.
    """
    logger.info("Memory Agent started execution.")
    query = state["query"]
    answer = state["draft_answer"]
    conv_id = state["conversation_id"]
    
    current_step = {
        "agent": "Memory Agent",
        "action": "Extracting key conversation facts & updating long-term memory.",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    existing_summary = ""
    existing_facts = {}
    
    # 1. Fetch existing memory from PostgreSQL
    async with async_session_maker() as session:
        try:
            result = await session.execute(
                select(ConversationMemory).where(ConversationMemory.conversation_id == conv_id)
            )
            db_mem = result.scalar_one_or_none()
            if db_mem:
                existing_summary = db_mem.memory_summary
                existing_facts = db_mem.extracted_facts
                
        except Exception as e:
            logger.warning(f"Memory Agent failed to fetch existing PG memory for conv {conv_id}: {e}")
            db_mem = None
            
    # 2. Invoke LLM to compute updated memory
    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.3,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    prompt_content = MEMORY_PROMPT.format(
        query=query,
        answer=answer,
        existing_summary=existing_summary or "None",
        existing_facts=json.dumps(existing_facts or {})
    )
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a silent memory summarization worker."),
            HumanMessage(content=prompt_content)
        ])
        
        parsed = json.loads(response.content)
        new_summary = parsed.get("memory_summary", existing_summary)
        new_facts = parsed.get("extracted_facts", existing_facts)
        
        logger.info(f"Memory Agent computed new summary length: {len(new_summary)}")
        
    except Exception as e:
        logger.error(f"Memory Agent failed to update memory context: {e}", exc_info=True)
        new_summary = existing_summary
        new_facts = existing_facts
        
    # 3. Write back memory updates to PostgreSQL
    async with async_session_maker() as session:
        try:
            # Re-fetch or create memory row
            result = await session.execute(
                select(ConversationMemory).where(ConversationMemory.conversation_id == conv_id)
            )
            db_mem = result.scalar_one_or_none()
            
            if not db_mem:
                db_mem = ConversationMemory(
                    conversation_id=conv_id,
                    memory_summary=new_summary,
                    extracted_facts=new_facts
                )
                session.add(db_mem)
            else:
                db_mem.memory_summary = new_summary
                db_mem.extracted_facts = new_facts
                
            await session.commit()
            logger.info("Successfully persisted conversation memory to PostgreSQL.")
            
        except Exception as e:
            logger.error(f"Failed to persist updated conversation memory to DB: {e}")
            await session.rollback()
            
    return {
        "steps": state.get("steps", []) + [current_step]
    }
