import os
import shutil
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from arq import create_pool
from arq.connections import RedisSettings

from app.config import settings
from app.database import engine, Base, get_db
from app.models import Document, Conversation, Message, ConversationMemory
from app.schemas import DocumentResponse, ChatRequest, ChatResponse, Citation, ConversationResponse, MessageResponse
from app.services.cache import cache_service
from app.services.vector_store import hybrid_retriever
from app.agents.graph import agent_graph

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# Setup Lifespan Event Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize PostgreSQL tables (async-safe)
    logger.info("Initializing PostgreSQL database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized successfully.")
    
    # 2. Connect Redis cache
    await cache_service.connect()
    
    # 3. Create Redis Connection Pool for Arq Worker queue
    app.state.arq_redis = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
    logger.info("Worker queue client connected.")
    
    yield
    
    # Clean up connections
    await cache_service.close()
    await app.state.arq_redis.close()
    logger.info("Shutdown connections cleaned up successfully.")

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    lifespan=lifespan
)

# Set CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production setups
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HOME ROUTE ---
@app.get("/")
def home():
    return {
        "message": "Welcome to the Multi-Agent Document Assistant API!",
        "docs_url": "/docs",
        "status": "healthy"
    }

# --- DOCUMENT ENDPOINTS ---

@app.post("/api/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Receives document, saves to filesystem, writes processing status to PG,
    and enqueues background parsing & vector-embedding task.
    """
    filename = file.filename
    _, ext = os.path.splitext(filename.lower())
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only PDF, DOCX, and TXT files are accepted."
        )

    # 1. Save uploaded file to disk
    file_id = str(datetime.utcnow().timestamp()).replace(".", "")
    safe_filename = f"{file_id}_{filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write file to storage: {e}"
        )

    # 2. Write record in DB
    db_doc = Document(
        filename=filename,
        file_path=file_path,
        status="processing"
    )
    db.add(db_doc)
    await db.commit()
    await db.refresh(db_doc)

    # 3. Enqueue Arq worker task
    try:
        await app.state.arq_redis.enqueue_job(
            "process_document_task",
            db_doc.id,
            file_path,
            filename
        )
        logger.info(f"Enqueued document ingestion for: {filename} (ID: {db_doc.id})")
    except Exception as e:
        logger.error(f"Failed to enqueue task for {filename}: {e}")
        # Mark as failed in DB since worker won't pick it up
        db_doc.status = "failed"
        db_doc.error_message = f"Queue enqueue failed: {e}"
        await db.commit()

    return db_doc

@app.get("/api/documents", response_model=List[DocumentResponse])
async def list_documents(db: AsyncSession = Depends(get_db)):
    """
    Returns list of all uploaded documents and their processing status.
    """
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return result.scalars().all()

@app.delete("/api/documents/{document_id}", status_code=status.HTTP_200_OK)
async def delete_document(document_id: str, db: AsyncSession = Depends(get_db)):
    """
    Deletes document from database, vector storage, and disk.
    """
    result = await db.execute(select(Document).where(Document.id == document_id))
    db_doc = result.scalar_one_or_none()
    if not db_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Delete from Vector Database
    hybrid_retriever.delete_document(document_id)
    
    # Delete from disk
    if os.path.exists(db_doc.file_path):
        try:
            os.remove(db_doc.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete file from disk: {db_doc.file_path}. {e}")
            
    # Delete from DB
    await db.delete(db_doc)
    await db.commit()
    
    return {"message": f"Successfully deleted document '{db_doc.filename}'"}

# --- CHAT & CONVERSATION ENDPOINTS ---

@app.get("/api/chat/conversations", response_model=List[ConversationResponse])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    """
    Retrieves all chat conversations.
    """
    result = await db.execute(select(Conversation).order_by(Conversation.created_at.desc()))
    return result.scalars().all()

@app.get("/api/chat/conversations/{conversation_id}/history", response_model=List[MessageResponse])
async def get_chat_history(conversation_id: str, db: AsyncSession = Depends(get_db)):
    """
    Fetches chronological message history for a single conversation.
    """
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        
    messages_result = await db.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.ascii())
    )
    return messages_result.scalars().all()

@app.post("/api/chat", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Main Chat Interface. Runs query through:
    1. Redis Caching Check
    2. LangGraph multi-agent processing pipeline (if cache miss)
    3. Memory extraction and PostgreSQL session logging.
    """
    # 1. Fetch or create Conversation
    conv_id = request.conversation_id
    if not conv_id:
        title = f"Doc Chat - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        db_conv = Conversation(title=title)
        db.add(db_conv)
        await db.commit()
        await db.refresh(db_conv)
        conv_id = db_conv.id
    else:
        result = await db.execute(select(Conversation).where(Conversation.id == conv_id))
        db_conv = result.scalar_one_or_none()
        if not db_conv:
            raise HTTPException(status_code=404, detail="Conversation ID not found")

    # 2. Redis Cache Lookup (Only on clean queries to showcase Redis)
    # We will key it on both query and conversation id to ensure conversation-specific contextual relevance
    cache_key = f"{conv_id}:{request.query}"
    cached = await cache_service.get(cache_key)
    if cached:
        logger.info("Serving response directly from Redis cache!")
        # Re-save user message for chronological correctness in UI
        user_msg = Message(
            conversation_id=conv_id,
            role="user",
            content=request.query
        )
        # Re-save cached assistant message
        assistant_msg = Message(
            conversation_id=conv_id,
            role="assistant",
            content=cached["answer"],
            citations=cached["citations"]
        )
        db.add(user_msg)
        db.add(assistant_msg)
        await db.commit()
        
        return ChatResponse(
            conversation_id=conv_id,
            answer=cached["answer"],
            citations=[Citation(**c) for c in cached["citations"]],
            agent_steps=[{
                "agent": "Redis Cache Service",
                "action": "Cache Hit! Instant response served.",
                "timestamp": datetime.utcnow().isoformat()
            }]
        )

    # 3. Pull previous messages from Postgres for agent context
    msg_select = await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at.asc())
    )
    db_messages = msg_select.scalars().all()
    
    agent_messages = []
    for m in db_messages:
        agent_messages.append({
            "role": m.role,
            "content": m.content
        })

    # 4. Invoke LangGraph multi-agent flow
    initial_state = {
        "messages": agent_messages,
        "conversation_id": conv_id,
        "query": request.query,
        "retrieval_plan": [],
        "retrieved_documents": [],
        "draft_answer": "",
        "verification_feedback": None,
        "verification_passed": True,
        "retries": 0,
        "steps": []
    }
    
    try:
        final_state = await agent_graph.ainvoke(initial_state)
    except Exception as e:
        logger.error(f"LangGraph execution halted with error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"LangGraph Agent workflow crashed during processing: {e}"
        )

    answer = final_state.get("draft_answer", "I encountered an error synthesizing the answer.")
    retrieved_docs = final_state.get("retrieved_documents", [])
    steps = final_state.get("steps", [])

    # Format citations
    citations = []
    for doc in retrieved_docs:
        meta = doc["metadata"]
        citations.append(Citation(
            filename=meta.get("filename", "Unknown File"),
            chunk_index=meta.get("chunk_index", 0),
            text=doc["text"]
        ))

    # 5. Persist Chat History to PostgreSQL
    user_msg = Message(
        conversation_id=conv_id,
        role="user",
        content=request.query
    )
    assistant_msg = Message(
        conversation_id=conv_id,
        role="assistant",
        content=answer,
        citations=[c.model_dump() for c in citations]
    )
    db.add(user_msg)
    db.add(assistant_msg)
    await db.commit()

    # 6. Cache response in Redis
    cached_payload = {
        "answer": answer,
        "citations": [c.model_dump() for c in citations]
    }
    await cache_service.set(cache_key, cached_payload, expire_seconds=1800)  # cache for 30 minutes

    return ChatResponse(
        conversation_id=conv_id,
        answer=answer,
        citations=citations,
        agent_steps=steps
    )
