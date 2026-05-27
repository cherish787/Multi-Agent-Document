import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import os
import sys

# Setup Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.document import DocumentProcessor
from app.services.vector_store import HybridRetriever
from app.agents.state import AgentState
from app.agents.graph import agent_graph, verification_routing_edge


class TestDocumentProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

    def test_chunking_logic(self):
        """Verify text recursive splitting behaves properly with constraints"""
        test_text = "This is a very long string that should be split into smaller blocks because the chunk size is set to a small value for testing."
        # Create a mock file parse
        pages = [{"text": test_text, "metadata": {"page": 1}}]
        
        # Test splitting
        chunks = []
        for page_data in pages:
            text = page_data["text"]
            page_meta = page_data["metadata"]
            page_chunks = self.processor.splitter.split_text(text)
            for i, pc in enumerate(page_chunks):
                chunks.append({
                    "text": pc,
                    "metadata": {
                        "filename": "test.txt",
                        "page": page_meta["page"],
                        "chunk_index": i
                    }
                })
        
        self.assertTrue(len(chunks) > 0)
        self.assertEqual(chunks[0]["metadata"]["filename"], "test.txt")
        self.assertEqual(chunks[0]["metadata"]["chunk_index"], 0)


class TestHybridRetrieverAndRRF(unittest.TestCase):
    @patch("app.services.vector_store.OpenAIEmbeddings")
    @patch("app.services.vector_store.chromadb.PersistentClient")
    def test_rrf_scoring(self, mock_chroma, mock_openai):
        """Ensure Reciprocal Rank Fusion (RRF) math computes scores correctly"""
        # Create a hybrid retriever instance with mocked dependencies
        retriever = HybridRetriever()
        
        # Setup mock hits
        semantic_hits = [
            {"id": "doc_1", "text": "Semantic match 1", "metadata": {"filename": "a.txt"}, "score": 0.1},
            {"id": "doc_2", "text": "Semantic match 2", "metadata": {"filename": "b.txt"}, "score": 0.2}
        ]
        keyword_hits = [
            {"id": "doc_2", "text": "Semantic match 2", "metadata": {"filename": "b.txt"}, "score": 10.0},
            {"id": "doc_3", "text": "Keyword match 3", "metadata": {"filename": "c.txt"}, "score": 5.0}
        ]
        
        retriever._semantic_search = MagicMock(return_value=semantic_hits)
        retriever._keyword_search = MagicMock(return_value=keyword_hits)
        
        # Perform hybrid fusion
        results = retriever.hybrid_search("test query", top_n=3)
        
        # doc_2 is in both, so it should rank first due to higher fused RRF score
        # RRF (doc_2) = (1 / (60 + 2)) + (1 / (60 + 1))
        # RRF (doc_1) = (1 / (60 + 1))
        # RRF (doc_3) = (1 / (60 + 2))
        
        self.assertEqual(results[0]["id"], "doc_2")
        self.assertEqual(results[1]["id"], "doc_1")
        self.assertEqual(results[2]["id"], "doc_3")
        self.assertTrue("rrf_score" in results[0])


class TestLangGraphRouting(unittest.TestCase):
    def test_routing_logic(self):
        """Ensure conditional graph edge routes based on verification and retry parameters"""
        # Case 1: Passed verification
        state_passed: AgentState = {
            "messages": [],
            "conversation_id": "test",
            "query": "hello",
            "retrieval_plan": [],
            "retrieved_documents": [],
            "draft_answer": "Fine",
            "verification_feedback": None,
            "verification_passed": True,
            "retries": 0,
            "steps": []
        }
        route = verification_routing_edge(state_passed)
        self.assertEqual(route, "memory")
        
        # Case 2: Failed verification, retries < 3
        state_failed: AgentState = {
            "messages": [],
            "conversation_id": "test",
            "query": "hello",
            "retrieval_plan": [],
            "retrieved_documents": [],
            "draft_answer": "Fine",
            "verification_feedback": "error info",
            "verification_passed": False,
            "retries": 1,
            "steps": []
        }
        route = verification_routing_edge(state_failed)
        self.assertEqual(route, "planner")
        
        # Case 3: Failed verification, but retries budget exhausted
        state_exhausted: AgentState = {
            "messages": [],
            "conversation_id": "test",
            "query": "hello",
            "retrieval_plan": [],
            "retrieved_documents": [],
            "draft_answer": "Fine",
            "verification_feedback": "error info",
            "verification_passed": False,
            "retries": 3,
            "steps": []
        }
        route = verification_routing_edge(state_exhausted)
        self.assertEqual(route, "memory")


if __name__ == "__main__":
    unittest.main()
