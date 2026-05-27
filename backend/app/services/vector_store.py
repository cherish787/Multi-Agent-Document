import uuid
import logging
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        # Initialize persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_assistant_chunks"
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], document_id: str):
        """
        Embeds and stores text chunks in ChromaDB.
        """
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = c["metadata"].copy()
            meta["document_id"] = document_id
            metadatas.append(meta)
            
        # Standardize unique IDs
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        # Get embeddings
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Successfully added {len(chunks)} chunks to vector store for document ID {document_id}")

    def delete_document(self, document_id: str):
        """
        Deletes all chunks associated with a document_id
        """
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted vector chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id}: {e}")

    def _semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search via Chroma DB
        """
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for idx in range(len(ids)):
                # Chroma distance is often L2 distance (lower is better, relevance score can be computed)
                search_results.append({
                    "id": ids[idx],
                    "text": docs[idx],
                    "metadata": metas[idx],
                    "score": float(distances[idx])
                })
        return search_results

    def _keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs keyword matching search using BM25 across all documents in the Chroma collection
        """
        # Retrieve all items in the collection
        all_data = self.collection.get(include=["documents", "metadatas"])
        if not all_data or not all_data["ids"]:
            return []
            
        corpus = all_data["documents"]
        metas = all_data["metadatas"]
        ids = all_data["ids"]
        
        # Tokenize corpus and query
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        tokenized_query = query.lower().split()
        
        # Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(tokenized_query)
        
        # Rank items by BM25 score
        scored_docs = []
        for i in range(len(ids)):
            score = float(doc_scores[i])
            if score > 0:  # Only include keyword hits with a positive score
                scored_docs.append({
                    "id": ids[i],
                    "text": corpus[i],
                    "metadata": metas[i],
                    "score": score
                })
                
        # Sort descending by score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def hybrid_search(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents using Reciprocal Rank Fusion (RRF)
        combining Semantic Search & BM25 Keyword Search.
        """
        semantic_hits = self._semantic_search(query, top_k=15)
        keyword_hits = self._keyword_search(query, top_k=15)
        
        # RRF formula: RRF_Score = SUM( 1 / (60 + rank) )
        rrf_scores = {}
        doc_details = {}
        
        # Map ID -> rank in semantic search
        for rank, hit in enumerate(semantic_hits, start=1):
            doc_id = hit["id"]
            doc_details[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))
            
        # Map ID -> rank in keyword search
        for rank, hit in enumerate(keyword_hits, start=1):
            doc_id = hit["id"]
            if doc_id not in doc_details:
                doc_details[doc_id] = hit
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))
            
        # Sort documents based on RRF scores in descending order
        sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        final_results = []
        for doc_id in sorted_ids[:top_n]:
            result = doc_details[doc_id].copy()
            # Inject RRF Score
            result["rrf_score"] = float(rrf_scores[doc_id])
            final_results.append(result)
            
        logger.info(f"Hybrid search returned {len(final_results)} chunks for query: '{query}'")
        return final_results

# Single global retriever instance
hybrid_retriever = HybridRetriever()
