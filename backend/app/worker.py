import logging
from arq.connections import RedisSettings
from sqlalchemy.future import select
from app.config import settings
from app.database import async_session_maker
from app.models import Document
from app.services.document import document_processor
from app.services.vector_store import hybrid_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_document_task(ctx: dict, doc_id: str, file_path: str, filename: str):
    """
    Background worker task to extract, split, embed, and store document chunks in the vector database.
    Updates the Postgres Document status when complete.
    """
    logger.info(f"Background worker starting chunking for: {filename} (ID: {doc_id})")
    
    async with async_session_maker() as session:
        try:
            # 1. Fetch document from DB to ensure it exists
            result = await session.execute(select(Document).where(Document.id == doc_id))
            db_doc = result.scalar_one_or_none()
            if not db_doc:
                logger.error(f"Document {doc_id} not found in database.")
                return
                
            # 2. Extract and chunk text
            chunks = document_processor.parse_and_chunk(file_path, filename)
            
            # 3. Add chunks to Vector Database (Chroma + BM25 index)
            # This is a synchronous call in standard langchain embeddings, so we run it directly or wrap it if needed.
            # But in arq tasks it runs fine.
            hybrid_retriever.add_chunks(chunks, doc_id)
            
            # 4. Update status to completed
            db_doc.status = "completed"
            await session.commit()
            logger.info(f"Successfully processed document: {filename} (ID: {doc_id})")
            
        except Exception as e:
            logger.error(f"Failed to process document {filename} (ID: {doc_id}): {e}", exc_info=True)
            await session.rollback()
            # Try to mark the document as failed
            try:
                result = await session.execute(select(Document).where(Document.id == doc_id))
                db_doc = result.scalar_one_or_none()
                if db_doc:
                    db_doc.status = "failed"
                    db_doc.error_message = str(e)
                    await session.commit()
            except Exception as db_err:
                logger.error(f"Failed to save failure status for document {doc_id}: {db_err}")

# Setup Arq Worker parameters
class WorkerSettings:
    functions = [process_document_task]
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    on_startup = None
    on_shutdown = None
