from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# --- Document Schemas ---
class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Message & Chat Schemas ---
class Citation(BaseModel):
    filename: str
    chunk_index: int
    text: str
    score: Optional[float] = None

class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    citations: Optional[List[Citation]] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationDetailResponse(ConversationResponse):
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The message or question to send to the assistant")
    conversation_id: Optional[str] = Field(None, description="The existing conversation ID, if any")

class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: List[Citation]
    agent_steps: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detailed trace of agent execution steps")
