from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    # Core Chat Logs
    messages: List[Dict[str, str]]  # Format: [{"role": "user"/"assistant", "content": "..."}]
    conversation_id: str
    
    # Task input & execution context
    query: str
    retrieval_plan: List[str]
    retrieved_documents: List[Dict[str, Any]]
    
    # Generated content
    draft_answer: str
    
    # Verification details
    verification_feedback: Optional[str]
    verification_passed: bool
    retries: int
    
    # Execution steps tracer for frontend visualization
    steps: List[Dict[str, Any]]  # Format: [{"agent": "planner", "action": "Analyzing query...", "timestamp": "..."}]
