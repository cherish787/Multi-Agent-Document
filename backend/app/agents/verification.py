import json
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import settings
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

VERIFICATION_PROMPT = """
You are the **Verification Agent** for a Multi-Agent Document Assistant.
Your task is to inspect the drafted answer and verify its truthfulness, accuracy, and compliance with citations, using ONLY the provided retrieved document chunks.

Evaluation Criteria:
1. **Grounded Integrity (Anti-Hallucination)**: Check if the draft answer contains claims, statistics, names, or statements not directly supported by the retrieved chunks. 
2. **Citation Check**:
   - Every fact must be accompanied by a bracketed citation, e.g. `[filename, Page X]`.
   - Ensure the cited filename and page matches a real chunk in the retrieved chunks.
   - Statements lacking citations must be flagged.
3. **Loop Decision**:
   - If the answer has hallucinations, lacks necessary citations, or has invalid citations, you MUST fail the verification.
   - If the answer is completely faithful to the chunks and has valid citations, you MUST pass the verification.

Response Format:
You must reply ONLY with a JSON object containing the following keys:
- "passed": true or false (boolean)
- "feedback": null or a string explaining exactly what claims are hallucinated or what citations are missing/broken.

Retrieved Context Chunks:
{context}
"""

async def verification_node(state: AgentState) -> dict:
    """
    Node that inspects the drafted answer against the raw retrieved chunks.
    """
    logger.info("Verification Agent started execution.")
    draft = state.get("draft_answer", "")
    docs = state.get("retrieved_documents", [])
    retries = state.get("retries", 0)
    
    current_step = {
        "agent": "Verification Agent",
        "action": "Verifying drafted response for hallucinations & citations.",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if not docs:
        # No documents retrieved, nothing to verify, pass through
        return {
            "verification_passed": True,
            "verification_feedback": None,
            "steps": state.get("steps", []) + [current_step]
        }
        
    # Format context for verification agent
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
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    prompt_content = f"Drafted Answer:\n{draft}\n\nPerform verification:"
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=VERIFICATION_PROMPT.format(context=context_str)),
            HumanMessage(content=prompt_content)
        ])
        
        result = json.loads(response.content)
        passed = result.get("passed", True)
        feedback = result.get("feedback")
        
        logger.info(f"Verification Agent verdict: Passed={passed}, Feedback={feedback}")
        
    except Exception as e:
        logger.error(f"Verification Agent failed to parse result, defaulting to Pass: {e}", exc_info=True)
        passed = True
        feedback = None
        
    # If failed, increment retries
    new_retries = retries
    if not passed:
        new_retries += 1
        
    current_step["action"] = f"Verification {'passed' if passed else 'failed'}. Retries: {new_retries}/3."
    if feedback:
        current_step["action"] += f" Feedback: {feedback}"
        
    return {
        "verification_passed": passed,
        "verification_feedback": feedback,
        "retries": new_retries,
        "steps": state.get("steps", []) + [current_step]
    }
